"""Multi-resolution data loader using batchgenerators.

Handles multiple resolution groups with weighted sampling, dynamic batch
sizing, subsampled data discovery, and super-resolution training.
"""

import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from .spacing import (
    SPACING_GRID, round_to_grid, round_spacing_to_tolerance,
    group_cases_by_spacing, get_canonical_permutation, apply_axis_permutation,
    resolve_root_parent,
)
from .batchgen_dataset import E3nnDataset, E3nnDataLoader, SuperResDataLoader


def estimate_memory_mb(
    patch_size: tuple,
    n_base_filters: int,
    batch_size: int = 2,
    n_downsample: int = 4,
    fp16: bool = False,
    spacing: tuple = None,
    mode: str = 'train',
) -> float:
    """Estimate GPU memory for training or inference (parametric model).

    This is a conservative parametric estimator used as a fallback when
    direct GPU profiling results are not available. For accurate memory
    estimation during experiment planning, use profile_memory_on_gpu()
    in train.py instead.

    Args:
        patch_size: Patch dimensions (D, H, W) in voxels
        n_base_filters: Number of base filters
        batch_size: Batch size
        n_downsample: Number of pooling levels (default: 4)
        fp16: BF16 mixed precision
        spacing: Voxel spacing in mm (unused, kept for API compat)
        mode: 'train' or 'infer'

    Returns:
        Estimated GPU memory usage in MB
    """
    return _estimate_memory_parametric(
        patch_size, n_base_filters, batch_size, n_downsample, fp16, mode
    )


def _estimate_memory_parametric(
    patch_size, n_base_filters, batch_size, n_downsample, fp16, mode='train',
):
    """Parametric fallback estimator (original conservative model)."""
    COEFFICIENTS_FP32 = {
        2: (3999, 651), 4: (19701, 2429), 8: (40000, 5000),
    }
    COEFFICIENTS_FP16 = {
        2: (24927, 461), 4: (18000, 1200), 8: (36000, 2500),
    }
    COEFFICIENTS = COEFFICIENTS_FP16 if fp16 else COEFFICIENTS_FP32
    DOWNSAMPLE_SCALE = {
        2: {3: 0.85, 4: 1.0, 5: 1.40},
        4: {3: 0.85, 4: 1.0, 5: 1.55},
        8: {3: 0.85, 4: 1.0, 5: 1.70},
    }

    coef_voxels, base = COEFFICIENTS.get(n_base_filters, (15000, 1500))
    scale_dict = DOWNSAMPLE_SCALE.get(n_base_filters, {3: 0.85, 4: 1.0, 5: 1.55})
    if n_downsample in scale_dict:
        ds_scale = scale_dict[n_downsample]
    elif n_downsample < 3:
        ds_scale = scale_dict[3] * (0.85 ** (3 - n_downsample))
    else:
        ds_scale = scale_dict[5] * (1.4 ** (n_downsample - 5))

    voxels = int(np.prod(patch_size))
    mem_for_batch2 = (coef_voxels * voxels / 1e6 + base) * ds_scale
    mem_train = mem_for_batch2 * batch_size / 2

    if mode == 'infer':
        return mem_train * 0.32
    return mem_train


def estimate_batch_size(
    patch_size: tuple,
    n_base_filters: int,
    target_memory_mb: float,
    min_batch: int = 1,
    max_batch: int = 32,
    n_downsample: int = 4,
    fp16: bool = False,
    spacing: tuple = None,
    mode: str = 'train',
) -> int:
    """Estimate batch size to fill target GPU memory (parametric fallback).

    For accurate batch sizing during experiment planning, use
    profile_memory_on_gpu() in train.py instead.

    Args:
        patch_size: Patch dimensions (D, H, W) in voxels
        n_base_filters: Number of base filters
        target_memory_mb: Target GPU memory to use
        min_batch: Minimum batch size
        max_batch: Maximum batch size
        n_downsample: Number of pooling levels (default: 4)
        fp16: BF16 mixed precision
        spacing: Voxel spacing in mm (unused, kept for API compat)
        mode: 'train' or 'infer'

    Returns:
        Estimated batch size to fill target memory
    """
    mem_bs1 = estimate_memory_mb(
        patch_size, n_base_filters, batch_size=1,
        n_downsample=n_downsample, fp16=fp16, mode=mode,
    )
    if mem_bs1 <= 0:
        return min_batch

    # Assume ~85% of bs=1 memory scales linearly with batch size
    per_item = mem_bs1 * 0.85
    overhead = mem_bs1 - per_item
    est_batch = int((target_memory_mb - overhead) / per_item)
    return max(min_batch, min(max_batch, est_batch))


# =============================================================================
# Helper Functions for Discovery
# =============================================================================

def discover_skip_files(
    preprocessed_dir: Path,
    case_ids: List[str],
    subsample_weight: float
) -> Tuple[Dict[str, dict], List[str]]:
    """Discover skip-downsampled preprocessed files.

    Args:
        preprocessed_dir: Path to preprocessed data directory
        case_ids: List of original case IDs
        subsample_weight: Only discover if > 0

    Returns:
        Tuple of:
        - properties: Dict mapping case_id (including skip variants) to metadata
        - subsampled_cases: List of skip variant case IDs
    """
    properties = {}
    subsampled_cases = []

    if subsample_weight == 0:
        return properties, subsampled_cases

    # Load properties for original cases
    for case_id in case_ids:
        pkl_path = preprocessed_dir / f"{case_id}.pkl"
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                properties[case_id] = pickle.load(f)

    # Discover skip variants
    for case_id in case_ids:
        for pkl_path in preprocessed_dir.glob(f"{case_id}_skip*.pkl"):
            sub_case_name = pkl_path.stem
            with open(pkl_path, 'rb') as f:
                sub_props = pickle.load(f)

            # Verify it's a subsampled version
            if sub_props.get('is_subsampled', False):
                properties[sub_case_name] = sub_props
                subsampled_cases.append(sub_case_name)

    return properties, subsampled_cases


def compute_steps_through_pooling(spacing, n_downsample, model_scale=2.0):
    """Compute effective spacing at each pooling level (same logic as e3nn UNet).

    Accounts for dynamic pooling: no pooling in dimensions where spacing >= scale.
    Returns list of spacings from input (level 0) through bottleneck (level n_downsample).
    """
    steps_array = [spacing]
    scales = [model_scale * (2 ** i) for i in range(n_downsample)]

    current_spacing = spacing
    for level in range(n_downsample):
        output_steps = []
        for dim in range(3):
            if current_spacing[dim] < scales[level]:
                kernel_dim = math.floor(scales[level] / current_spacing[dim])
                output_steps.append(kernel_dim * current_spacing[dim])
            else:
                output_steps.append(current_spacing[dim])
        current_spacing = tuple(output_steps)
        steps_array.append(current_spacing)

    return steps_array


def adjust_for_divisibility_per_dim(voxels, factors, min_size=8):
    """Adjust voxels to be divisible by per-dimension pooling factors.

    Parameters
    ----------
    voxels : tuple
        Input voxel counts (D, H, W)
    factors : tuple
        Pooling factor per dimension
    min_size : int
        Minimum voxel size per dimension when no pooling occurs

    Returns
    -------
    tuple
        Adjusted voxel counts
    """
    adjusted = []
    for v, f in zip(voxels, factors):
        if f > 1:
            v_int = int(v)
            f_int = int(f)
            v_adj = max(f_int, (v_int // f_int) * f_int)
        else:
            v_adj = max(min_size, int(round(v)))
        adjusted.append(v_adj)
    return tuple(adjusted)


def mm_to_voxels(patch_size_mm, spacing, n_downsample, model_scale=2.0):
    """Convert patch size from mm to voxels, accounting for dynamic pooling.

    Proper algorithm that works backwards from target receptive field:
    1. Compute how spacing evolves through pooling (per-dimension pooling factors)
    2. Compute bottleneck voxels needed for target RF: patch_size_mm / bottleneck_spacing
    3. Work backwards: input_voxels = bottleneck_voxels * pooling_factor
    4. Adjust for per-dimension divisibility constraints

    Parameters
    ----------
    patch_size_mm : tuple
        Patch size in mm (D, H, W)
    spacing : tuple
        Voxel spacing in mm (D, H, W)
    n_downsample : int
        Number of downsampling levels
    model_scale : float
        Base pooling scale (default 2.0)

    Returns
    -------
    tuple
        Patch size in voxels (D, H, W)
    """
    steps_array = compute_steps_through_pooling(spacing, n_downsample, model_scale)
    final_spacing = steps_array[-1]

    pooling_factors = []
    for dim in range(3):
        if spacing[dim] > 0:
            pool_factor = final_spacing[dim] / spacing[dim]
        else:
            pool_factor = 1.0
        pooling_factors.append(pool_factor)

    bottleneck_voxels_needed = []
    for dim in range(3):
        if final_spacing[dim] > 0:
            voxels_bn = patch_size_mm[dim] / final_spacing[dim]
        else:
            voxels_bn = patch_size_mm[dim]
        bottleneck_voxels_needed.append(voxels_bn)

    input_voxels_needed = []
    for dim in range(3):
        voxels_in = bottleneck_voxels_needed[dim] * pooling_factors[dim]
        input_voxels_needed.append(voxels_in)

    return adjust_for_divisibility_per_dim(
        tuple(input_voxels_needed),
        tuple(pooling_factors),
        min_size=8,
    )


def verify_receptive_field(input_voxels, spacing, target_rf_mm, n_downsample, model_scale=2.0):
    """Verify the receptive field after pooling matches target.

    Parameters
    ----------
    input_voxels : tuple
        Input patch voxel counts (D, H, W)
    spacing : tuple
        Voxel spacing in mm (D, H, W)
    target_rf_mm : tuple
        Target receptive field in mm (D, H, W)
    n_downsample : int
        Number of downsampling levels
    model_scale : float
        Base pooling scale (default 2.0)

    Returns
    -------
    dict with keys: target_rf, actual_rf, error, max_error
    """
    steps_array = compute_steps_through_pooling(spacing, n_downsample, model_scale)
    final_spacing = steps_array[-1]

    pooling_factors = tuple(final_spacing[d] / spacing[d] for d in range(3))
    bottleneck_voxels = tuple(input_voxels[d] / pooling_factors[d] for d in range(3))
    actual_rf = tuple(bottleneck_voxels[d] * final_spacing[d] for d in range(3))
    error = tuple(abs(actual_rf[d] - target_rf_mm[d]) for d in range(3))

    return {
        'target_rf': target_rf_mm,
        'actual_rf': actual_rf,
        'error': error,
        'max_error': max(error),
        'bottleneck_voxels': bottleneck_voxels,
        'bottleneck_spacing': final_spacing,
        'pooling_factors': pooling_factors,
    }


# =============================================================================
# MultiResolutionLoader
# =============================================================================

class MultiResolutionLoader:
    """Loader that handles multiple resolution groups with weighted sampling.

    Patch size is specified in physical units (mm) and converted to voxels
    for each resolution group based on that group's spacing.

    Supports dynamic batch sizing: automatically increases batch size for
    lower-resolution groups to better utilize GPU memory.

    Parameters
    ----------
    preprocessed_dir : str
        Path to preprocessed data directory
    case_identifiers : list of str
        List of case identifiers to load
    batch_size : int
        Base batch size (used for highest resolution group, or all groups if
        dynamic_batch_size is False)
    patch_size_mm : tuple of float
        Patch size in mm (D, H, W). Converted to voxels per resolution group.
    oversample_foreground_percent : float
        Fraction of samples guaranteed to contain foreground
    num_workers : int
        Number of data loading workers
    transforms : Compose
        Transforms to apply to batches
    pooling_factor : int
        Ensure patch sizes are divisible by this (for UNet pooling)
    dynamic_batch_size : bool
        If True, automatically scale batch size per resolution group to fill
        target_memory_mb. Lower resolution -> larger batch size.
    target_memory_mb : float
        Target GPU memory to fill when dynamic_batch_size is True.
    n_base_filters : int
        Number of base filters in the model (needed for memory estimation).
    n_downsample : int
        Number of pooling levels in the model (needed for memory estimation).
    model_scale : float
        Pooling scale in physical units for the e3nn model (default: 2.0 mm).
    min_batch_size : int
        Minimum batch size when using dynamic sizing.
    max_batch_size : int
        Maximum batch size when using dynamic sizing.
    subsample_weight : float
        Sampling weight for preprocessed subsampled data (0.0 to disable).
    """

    def __init__(
        self,
        preprocessed_dir: str,
        case_identifiers: List[str],
        batch_size: int,
        patch_size_mm: Tuple[float, ...],
        oversample_foreground_percent: float = 0.33,
        num_workers: int = 4,
        transforms=None,
        pooling_factor: int = 8,
        dynamic_batch_size: bool = False,
        target_memory_mb: float = 0,
        n_base_filters: int = 2,
        n_downsample: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        fp16: bool = False,
        subsample_weight: float = 0.0,
        model_scale: float = 2.0,
        min_spacing: float = 0.0,
        max_inplane_spacing: float = 0.0,
        min_slice_thickness: float = 0.0,
        max_slice_thickness: float = 0.0,
        min_loader_cases: int = 2,
        superres_training: bool = False,
        superres_weight: float = 0.1,
        group_balance: float = 0.0,
        planned_batch_sizes: Optional[Dict[tuple, int]] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.batch_size = batch_size
        self.dynamic_batch_size = dynamic_batch_size
        self.target_memory_mb = target_memory_mb
        self.n_base_filters = n_base_filters
        self.n_downsample = n_downsample
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.patch_size_mm = patch_size_mm
        self.transforms = transforms
        self.pooling_factor = pooling_factor
        self.fp16 = fp16
        self.subsample_weight = subsample_weight
        self.model_scale = model_scale
        self.superres_training = superres_training
        self.superres_weight = superres_weight
        self.planned_batch_sizes = planned_batch_sizes or {}

        # Load properties to get spacings
        properties = {}
        for case_name in case_identifiers:
            pkl_path = self.preprocessed_dir / f"{case_name}.pkl"
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    properties[case_name] = pickle.load(f)

        # Discover preprocessed subsampled cases if subsample_weight > 0
        subsampled_cases = []
        if subsample_weight > 0:
            for case_name in case_identifiers:
                for pkl_path in self.preprocessed_dir.glob(f"{case_name}_skip*.pkl"):
                    sub_case_name = pkl_path.stem
                    with open(pkl_path, 'rb') as f:
                        sub_props = pickle.load(f)
                    if sub_props.get('is_subsampled', False):
                        properties[sub_case_name] = sub_props
                        subsampled_cases.append(sub_case_name)

        # Group cases by spacing (with optional spacing filters)
        self.min_spacing = min_spacing
        self.max_inplane_spacing = max_inplane_spacing
        self.min_slice_thickness = min_slice_thickness
        self.max_slice_thickness = max_slice_thickness
        self.spacing_groups = group_cases_by_spacing(
            properties, min_spacing=min_spacing, max_inplane_spacing=max_inplane_spacing,
            min_slice_thickness=min_slice_thickness, max_slice_thickness=max_slice_thickness)

        # Track which cases are subsampled (for weight calculation)
        self.subsampled_cases = set(subsampled_cases)
        self._all_properties = properties

        # Create loaders for each group
        self.group_loaders: Dict[tuple, MultiThreadedAugmenter] = {}
        self.group_weights = []

        n_original = len(case_identifiers)
        n_subsampled = len(subsampled_cases)

        print(f"\nMulti-resolution loader setup:")
        print(f"  Total cases: {n_original} original + {n_subsampled} subsampled")
        print(f"  Patch size: {patch_size_mm} mm")
        print(f"  Base batch size: {batch_size}")
        print(f"  Resolution groups: {len(self.spacing_groups)}")
        if subsample_weight > 0:
            print(f"  Preprocessed subsample weight: {subsample_weight:.2f}")
        if dynamic_batch_size:
            print(f"  Dynamic batch sizing: enabled (target={target_memory_mb:.0f} MB)")

        self.group_batch_sizes: Dict[tuple, int] = {}
        self.group_patch_sizes: Dict[tuple, tuple] = {}
        self.min_loader_cases = min_loader_cases
        skipped_groups = []

        for spacing, cases in self.spacing_groups.items():
            if len(cases) < min_loader_cases:
                skipped_groups.append((spacing, f"too few cases ({len(cases)} < {min_loader_cases})"))
                continue

            patch_size_voxels = self._mm_to_voxels(patch_size_mm, spacing)

            group_batch_size = self._compute_batch_size(patch_size_voxels, spacing=spacing)
            if spacing in self.planned_batch_sizes:
                planned_bs = self.planned_batch_sizes[spacing]
                if group_batch_size != planned_bs:
                    # Should not happen since we use planned value directly,
                    # but log for safety
                    print(f"    Using planned batch size {planned_bs} for {spacing}")
            elif self.planned_batch_sizes:
                # Spacing group appeared at runtime that wasn't in the plan — skip it
                skipped_groups.append((spacing, f"not in planned batch sizes"))
                continue

            self._create_group(
                spacing, cases, preprocessed_dir, group_batch_size, patch_size_voxels,
                oversample_foreground_percent, num_workers, transforms,
                n_total_cases=len(case_identifiers),
            )

        if skipped_groups:
            print(f"  Skipped {len(skipped_groups)} groups:")
            for spacing, reason in skipped_groups:
                spacing_str = tuple(f"{s:.3f}" for s in spacing)
                print(f"    {spacing_str}: {reason}")

        # Create super-resolution subloaders if enabled
        self.superres_loaders: Dict[tuple, SingleThreadedAugmenter] = {}
        if superres_training and subsampled_cases:
            self._create_superres_loaders(
                preprocessed_dir, subsampled_cases, properties,
                oversample_foreground_percent, superres_weight,
                n_total_cases=len(case_identifiers),
            )

        # Normalize weights
        total_weight = sum(w for _, w in self.group_weights)
        self.group_weights = [(s, w / total_weight) for s, w in self.group_weights]

        # Apply group balancing: blend proportional with uniform
        # balance=0: proportional to case count, balance=1: uniform across groups
        if group_balance > 0 and len(self.group_weights) > 1:
            uniform = 1.0 / len(self.group_weights)
            self.group_weights = [
                (s, (1 - group_balance) * w + group_balance * uniform)
                for s, w in self.group_weights
            ]
            print(f"  Group balancing: {group_balance:.2f} (0=proportional, 1=uniform)")

        # Initialize iterators
        for spacing, loader in self.group_loaders.items():
            try:
                next(loader)
            except:
                pass
        for key, loader in self.superres_loaders.items():
            try:
                next(loader)
            except:
                pass

    def _adjust_for_divisibility(self, size: int, factor: int) -> int:
        """Adjust size to be divisible by factor."""
        return max(factor, (size // factor) * factor)

    def _adjust_for_divisibility_per_dim(self, voxels, factors, min_size=8):
        """Delegate to standalone adjust_for_divisibility_per_dim()."""
        return adjust_for_divisibility_per_dim(voxels, factors, min_size)

    def _compute_steps_through_pooling(self, spacing: tuple) -> list:
        """Delegate to standalone compute_steps_through_pooling()."""
        return compute_steps_through_pooling(spacing, self.n_downsample, self.model_scale)

    def _mm_to_voxels(self, patch_size_mm, spacing):
        """Delegate to standalone mm_to_voxels()."""
        return mm_to_voxels(patch_size_mm, spacing, self.n_downsample, self.model_scale)

    def _verify_receptive_field(self, input_voxels, spacing, target_rf_mm):
        """Delegate to standalone verify_receptive_field()."""
        return verify_receptive_field(
            input_voxels, spacing, target_rf_mm,
            self.n_downsample, self.model_scale
        )

    def _compute_batch_size(self, patch_size_voxels: tuple, spacing: tuple = None) -> int:
        """Compute batch size for a given patch size.

        If planned_batch_sizes contains the spacing, uses the planned value
        directly. Otherwise falls back to estimation.
        """
        if spacing is not None and spacing in self.planned_batch_sizes:
            return self.planned_batch_sizes[spacing]

        if not self.dynamic_batch_size:
            return self.batch_size

        if self.target_memory_mb <= 0:
            return self.batch_size

        return estimate_batch_size(
            patch_size=patch_size_voxels,
            n_base_filters=self.n_base_filters,
            target_memory_mb=self.target_memory_mb,
            min_batch=self.min_batch_size,
            max_batch=self.max_batch_size,
            n_downsample=self.n_downsample,
            fp16=self.fp16,
            spacing=spacing,
        )

    def _create_group(
        self,
        spacing: tuple,
        cases: List[str],
        preprocessed_dir: str,
        batch_size: int,
        patch_size: tuple,
        oversample_foreground_percent: float,
        num_workers: int,
        transforms,
        n_total_cases: int,
    ):
        """Create a loader for a resolution group."""
        dataset = E3nnDataset(preprocessed_dir, cases)

        dataloader = E3nnDataLoader(
            data=dataset,
            batch_size=batch_size,
            patch_size=patch_size,
            oversample_foreground_percent=oversample_foreground_percent,
            probabilistic_oversampling=True,
        )

        if num_workers > 0 and transforms is not None:
            augmenter = MultiThreadedAugmenter(
                dataloader, transforms,
                num_processes=num_workers,
                num_cached_per_queue=2,
                pin_memory=False
            )
        else:
            augmenter = SingleThreadedAugmenter(dataloader, transforms)

        self.group_loaders[spacing] = augmenter
        self.group_batch_sizes[spacing] = batch_size
        self.group_patch_sizes[spacing] = patch_size

        # Weight calculation
        n_original = sum(1 for c in cases if c not in self.subsampled_cases)
        n_subsampled = sum(1 for c in cases if c in self.subsampled_cases)

        if n_original > 0 and self.subsample_weight < 1.0:
            original_weight = (1.0 - self.subsample_weight) * n_original / n_total_cases
        else:
            original_weight = 0.0

        if n_subsampled > 0 and self.subsample_weight > 0:
            subsampled_weight = self.subsample_weight * n_subsampled / n_total_cases
        else:
            subsampled_weight = 0.0

        weight = (original_weight + subsampled_weight) / batch_size

        self.group_weights.append((spacing, weight))

        spacing_str = tuple(f"{s:.3f}" for s in spacing)
        if n_subsampled > 0 and n_original > 0:
            group_type = f"mixed ({n_original} orig + {n_subsampled} sub)"
        elif n_subsampled > 0:
            group_type = "subsampled"
        else:
            group_type = "real"

        print(f"    Spacing {spacing_str}: {len(cases)} cases, patch={patch_size}, batch={batch_size}, weight={weight:.3f} ({group_type})")

    def _create_superres_loaders(
        self,
        preprocessed_dir: str,
        subsampled_cases: List[str],
        properties: Dict[str, dict],
        oversample_foreground_percent: float,
        superres_weight: float,
        n_total_cases: int,
    ):
        """Create super-resolution subloaders for subsampled cases."""
        sr_groups: Dict[tuple, List[str]] = defaultdict(list)
        for sub_id in subsampled_cases:
            if sub_id not in properties:
                continue
            root_id = resolve_root_parent(sub_id, properties)
            if root_id not in properties:
                continue

            sub_sp = properties[sub_id]['spacing']
            orig_sp = properties[root_id]['spacing']
            sub_canonical = tuple(sorted(sub_sp))
            orig_canonical = tuple(sorted(orig_sp))
            sub_rounded = round_spacing_to_tolerance(sub_canonical)
            orig_rounded = round_spacing_to_tolerance(orig_canonical)
            key = ('superres', sub_rounded, orig_rounded)
            sr_groups[key].append(sub_id)

        if not sr_groups:
            return

        print(f"\n  Super-resolution subloaders:")
        n_sr_groups = len(sr_groups)
        weight_per_group = superres_weight / n_sr_groups if n_sr_groups > 0 else 0

        for key, sub_ids in sorted(sr_groups.items()):
            _, sub_sp, orig_sp = key
            patch_size_voxels = self._mm_to_voxels(self.patch_size_mm, sub_sp)

            sr_loader = SuperResDataLoader(
                preprocessed_dir=self.preprocessed_dir,
                sub_case_ids=sub_ids,
                properties=properties,
                patch_size=patch_size_voxels,
                oversample_foreground_percent=oversample_foreground_percent,
            )

            sr_transforms = Compose([NumpyToTensor(keys=['data', 'seg'], cast_to='float')])
            augmenter = SingleThreadedAugmenter(sr_loader, sr_transforms)
            self.superres_loaders[key] = augmenter

            self.group_weights.append((key, weight_per_group))

            sub_sp_str = tuple(f"{s:.2f}" for s in sub_sp)
            orig_sp_str = tuple(f"{s:.2f}" for s in orig_sp)
            print(f"    {sub_sp_str} -> {orig_sp_str}: {len(sub_ids)} cases, "
                  f"patch={patch_size_voxels}, batch=1, weight={weight_per_group:.3f}")

    def set_active_groups(self, predicate=None):
        """Restrict sampling to groups matching *predicate*.

        Parameters
        ----------
        predicate : callable((float,float,float)) -> bool, or None
            A function that receives the canonical spacing tuple and
            returns True if the group should be active.  ``None``
            restores all groups.

        The original ``group_weights`` are preserved — calling with
        ``None`` fully restores them.  Active weights are renormalised
        to sum to 1.
        """
        if not hasattr(self, '_all_group_weights'):
            self._all_group_weights = list(self.group_weights)

        if predicate is None:
            self.group_weights = list(self._all_group_weights)
        else:
            active = [(s, w) for s, w in self._all_group_weights
                      if isinstance(s, tuple) and len(s) == 3
                      and s[0] != 'superres' and predicate(s)]
            # Keep superres entries as-is if present
            superres = [(s, w) for s, w in self._all_group_weights
                        if isinstance(s, tuple) and len(s) == 3
                        and s[0] == 'superres']
            if not active:
                # Fallback: keep all groups to avoid empty loader
                self.group_weights = list(self._all_group_weights)
                return
            self.group_weights = active + superres

        total = sum(w for _, w in self.group_weights)
        if total > 0:
            self.group_weights = [(s, w / total) for s, w in self.group_weights]

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[dict, tuple]:
        """Get next batch, sampling from resolution groups by weight."""
        spacings = [s for s, _ in self.group_weights]
        weights = [w for _, w in self.group_weights]
        group_spacing = spacings[np.random.choice(len(spacings), p=weights)]

        # Check if this is a super-res group
        if isinstance(group_spacing, tuple) and len(group_spacing) == 3 and group_spacing[0] == 'superres':
            batch = next(self.superres_loaders[group_spacing])
            if 'spacings' in batch and len(batch['spacings']) > 0:
                selected_spacing = tuple(batch['spacings'][0])
            else:
                selected_spacing = group_spacing[1]
            return batch, selected_spacing

        batch = next(self.group_loaders[group_spacing])

        if 'spacings' in batch and len(batch['spacings']) > 0:
            idx = np.random.randint(len(batch['spacings']))
            selected_spacing = batch['spacings'][idx]
            selected_spacing = tuple(selected_spacing)
        else:
            selected_spacing = group_spacing

        # Store canonical group spacing for spatial_splits lookup
        batch['group_spacing'] = group_spacing

        return batch, selected_spacing

    def save_config(self, filepath: str, args=None, n_train_cases=0, n_val_cases=0):
        """Save loader configuration using shared writer.

        Parameters
        ----------
        filepath : str
            Output file path
        args : argparse.Namespace, optional
            Training arguments (required for full output)
        n_train_cases : int
            Number of training cases
        n_val_cases : int
            Number of validation cases
        """
        from irrepunet.training.utils import write_loader_config

        groups = []
        for spacing, weight in self.group_weights:
            if isinstance(spacing, tuple) and len(spacing) == 3 and spacing[0] == 'superres':
                continue
            patch_voxels = self.group_patch_sizes.get(spacing, ())
            batch_size = self.group_batch_sizes.get(spacing, self.batch_size)
            est_mem = estimate_memory_mb(
                patch_voxels, self.n_base_filters,
                batch_size=batch_size, n_downsample=self.n_downsample,
                fp16=self.fp16, spacing=spacing,
            )
            cases = self.spacing_groups.get(spacing, [])
            n_sub = sum(1 for c in cases if c in self.subsampled_cases)
            n_orig = len(cases) - n_sub
            if n_sub > 0 and n_orig > 0:
                group_type = 'mixed'
            elif n_sub > 0:
                group_type = 'subsampled'
            else:
                group_type = 'real'

            # Compute spatial splits needed if batch_size=1 exceeds memory
            n_spatial_splits = 1
            if batch_size == 1 and self.target_memory_mb > 0:
                mem_for_1 = estimate_memory_mb(
                    patch_voxels, self.n_base_filters,
                    batch_size=1, n_downsample=self.n_downsample,
                    fp16=self.fp16, spacing=spacing,
                )
                if mem_for_1 > self.target_memory_mb:
                    import math
                    test_patch = list(patch_voxels)
                    while mem_for_1 > self.target_memory_mb and n_spatial_splits <= 16:
                        n_spatial_splits *= 2
                        max_dim_idx = test_patch.index(max(test_patch))
                        test_patch[max_dim_idx] = max(
                            self.pooling_factor, test_patch[max_dim_idx] // 2
                        )
                        mem_for_1 = estimate_memory_mb(
                            tuple(test_patch), self.n_base_filters,
                            batch_size=1, n_downsample=self.n_downsample,
                            fp16=self.fp16, spacing=spacing,
                        )
                    est_mem = mem_for_1  # Show per-split memory

            groups.append({
                'spacing': spacing,
                'patch_size_voxels': patch_voxels,
                'batch_size': batch_size,
                'n_spatial_splits': n_spatial_splits,
                'estimated_memory_mb': round(est_mem, 1),
                'n_cases': len(cases),
                'group_type': group_type,
            })

        write_loader_config(
            filepath=filepath,
            args=args,
            groups=groups,
            n_train_cases=n_train_cases,
            n_val_cases=n_val_cases,
            model_scale=self.model_scale,
        )
