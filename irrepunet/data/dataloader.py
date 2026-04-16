"""Multi-resolution data loader using PyTorch DataLoader.

Simplified replacement for the batchgenerators-based MultiResolutionLoader.
Uses torch.utils.data.IterableDataset with a shared worker pool instead of
per-group MultiThreadedAugmenter instances.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .spacing import (
    group_cases_by_spacing, get_canonical_permutation, apply_axis_permutation,
)
from .multi_resolution_loader import (
    estimate_memory_mb, estimate_batch_size,
    mm_to_voxels, compute_steps_through_pooling, verify_receptive_field,
    adjust_for_divisibility_per_dim, discover_skip_files,
)
from .decimation import decimate_array, random_offsets, compute_class_locations


# =============================================================================
# Helpers
# =============================================================================

def _permute_class_locations(class_locations, perm):
    """Permute foreground coordinates to match axis permutation."""
    permuted = {}
    for cls, coords in class_locations.items():
        if len(coords) > 0:
            permuted[cls] = coords[:, perm]
        else:
            permuted[cls] = coords
    return permuted


def _get_bbox(shape, patch_size, force_fg, class_locations):
    """Compute patch bounding box with optional foreground centering."""
    dim = len(shape)
    lbs = [0] * dim
    ubs = [max(0, shape[i] - patch_size[i]) for i in range(dim)]

    if force_fg and class_locations:
        eligible = [k for k in class_locations if len(class_locations[k]) > 0]
        if eligible:
            cls = np.random.choice(eligible)
            voxels = class_locations[cls]
            voxel = voxels[np.random.choice(len(voxels))]
            bbox_lbs = []
            for i in range(dim):
                lb = max(lbs[i], voxel[i] - patch_size[i] // 2)
                lb = min(lb, ubs[i])
                bbox_lbs.append(int(lb))
            return bbox_lbs, [bbox_lbs[i] + patch_size[i] for i in range(dim)]

    bbox_lbs = [
        np.random.randint(lbs[i], ubs[i] + 1) if ubs[i] > lbs[i] else lbs[i]
        for i in range(dim)
    ]
    return bbox_lbs, [bbox_lbs[i] + patch_size[i] for i in range(dim)]


def _pad_to_size(arr, target_shape):
    """Pad array to target shape with zeros."""
    pad_width = [(0, max(0, t - c)) for c, t in zip(arr.shape, target_shape)]
    return np.pad(arr, pad_width, mode='constant', constant_values=0)


def _load_and_extract_patch(
    case_id, preprocessed_dir, properties, patch_size, num_channels,
    oversample_foreground_percent,
):
    """Load a case, apply axis permutation, and extract a random patch.

    Synthetic decimated cases load the base source's .npy and apply ``np.take``.

    Returns (data_patch, seg_patch, canonical_spacing).
    """
    props = properties.get(case_id, {})
    strides = props.get('decimation_strides')
    if strides and any(s > 1 for s in strides):
        base_id = props['decimation_base_id']
        data = np.load(preprocessed_dir / f"{base_id}.npy")
        seg_raw = np.load(preprocessed_dir / f"{base_id}_seg.npy")
        offsets = random_offsets(strides)
        data = decimate_array(data, strides, has_channel=True, offsets=offsets)
        seg_raw = decimate_array(seg_raw, strides, has_channel=False, offsets=offsets)
        # Per-sample random offset: class_locations depend on the offset and
        # are recomputed from the decimated seg (small arrays → cheap).
        class_locations = compute_class_locations(seg_raw)
        seg = seg_raw[np.newaxis]
    else:
        data = np.load(preprocessed_dir / f"{case_id}.npy")
        seg = np.load(preprocessed_dir / f"{case_id}_seg.npy")[np.newaxis]
        class_locations = props.get('class_locations', {})

    original_spacing = props.get('spacing', (1.0, 1.0, 1.0))

    perm = get_canonical_permutation(original_spacing)
    if perm != (0, 1, 2):
        data = apply_axis_permutation(data, perm, has_channel=True)
        seg = apply_axis_permutation(seg, perm, has_channel=True)
        class_locations = _permute_class_locations(class_locations, perm)
    canonical_spacing = tuple(original_spacing[i] for i in perm)

    force_fg = np.random.uniform() < oversample_foreground_percent
    shape = data.shape[1:]
    bbox_lbs, bbox_ubs = _get_bbox(shape, patch_size, force_fg, class_locations)

    slices = tuple(slice(lb, ub) for lb, ub in zip(bbox_lbs, bbox_ubs))
    data_patch = data[(slice(None),) + slices]
    seg_patch = seg[(slice(None),) + slices]

    if data_patch.shape[1:] != tuple(patch_size):
        data_patch = _pad_to_size(data_patch, (num_channels,) + tuple(patch_size))
        seg_patch = _pad_to_size(seg_patch, (1,) + tuple(patch_size))

    return data_patch, seg_patch, canonical_spacing


def _identity_collate(x):
    """No-op collate function — passes data through unchanged."""
    return x


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class _GroupConfig:
    """Configuration for one resolution group."""
    spacing: tuple
    case_ids: list
    patch_size_voxels: tuple
    batch_size: int
    weight: float
    n_original: int
    n_subsampled: int


# =============================================================================
# IterableDataset
# =============================================================================

class _MultiResIterableDataset(IterableDataset):
    """Produces complete batches from weighted resolution groups.

    Each worker independently samples groups and produces batches.
    No shared mutable state between workers.
    """

    def __init__(self, groups, preprocessed_dir, properties, transforms,
                 oversample_foreground_percent, num_channels, rank=0,
                 sync_groups=False):
        super().__init__()
        self.groups = groups
        self.preprocessed_dir = Path(preprocessed_dir)
        self.properties = properties
        self.transforms = transforms
        self.oversample_foreground_percent = oversample_foreground_percent
        self.num_channels = num_channels
        self._rank = rank
        self._sync_groups = sync_groups

        total = sum(g.weight for g in groups)
        self._weights = np.array([g.weight / total for g in groups])

    def __iter__(self):
        # Seed numpy per-worker for independent randomness.
        # batchgenerators transforms use np.random internally.
        info = torch.utils.data.get_worker_info()
        if info is not None:
            # Incorporate DDP rank so each rank draws different patches
            np.random.seed((info.seed + self._rank * 2654435761) % (2**32))
        elif self._rank > 0:
            np.random.seed((42 + self._rank * 2654435761) % (2**32))

        # Per-worker shuffled case buffers
        buffers = []
        for group in self.groups:
            ids = list(group.case_ids)
            np.random.shuffle(ids)
            buffers.append({'ids': ids, 'pos': 0})

        # Group-selection RNG.  In sync mode a separate deterministic RNG
        # makes every DDP rank pick the same group on each step.  In async
        # mode (default) the per-rank np.random produces independent choices.
        if self._sync_groups:
            group_rng = np.random.RandomState(seed=3407)
        else:
            group_rng = None

        while True:
            if group_rng is not None:
                gi = int(group_rng.choice(len(self.groups), p=self._weights))
            else:
                gi = int(np.random.choice(len(self.groups), p=self._weights))
            group = self.groups[gi]
            buf = buffers[gi]
            patch_size = np.array(group.patch_size_voxels)

            # Draw case IDs from shuffled buffer
            case_ids = []
            for _ in range(group.batch_size):
                if buf['pos'] >= len(buf['ids']):
                    np.random.shuffle(buf['ids'])
                    buf['pos'] = 0
                case_ids.append(buf['ids'][buf['pos']])
                buf['pos'] += 1

            # Build batch arrays
            data_all = np.zeros(
                (group.batch_size, self.num_channels, *group.patch_size_voxels),
                dtype=np.float32,
            )
            seg_all = np.zeros(
                (group.batch_size, 1, *group.patch_size_voxels),
                dtype=np.int16,
            )
            spacings_all = []

            for j, case_id in enumerate(case_ids):
                dp, sp, cs = _load_and_extract_patch(
                    case_id, self.preprocessed_dir, self.properties,
                    patch_size, self.num_channels,
                    self.oversample_foreground_percent,
                )
                data_all[j] = dp
                seg_all[j] = sp
                spacings_all.append(cs)

            batch = {
                'data': data_all,
                'seg': seg_all,
                'keys': case_ids,
                'spacings': spacings_all,
            }

            if self.transforms is not None:
                batch = self.transforms(**batch)

            yield batch, group.spacing


# =============================================================================
# _GroupIterator
# =============================================================================

class _GroupIterator:
    """Synchronous per-group iterator for gradient accumulation.

    Produces batches from a single resolution group in the main thread.
    Used when ``next(loader.group_loaders[spacing])`` is called directly.
    """

    def __init__(self, group, preprocessed_dir, properties, transforms,
                 oversample_foreground_percent, num_channels, rank=0):
        self.group = group
        self.preprocessed_dir = Path(preprocessed_dir)
        self.properties = properties
        self.transforms = transforms
        self.oversample_foreground_percent = oversample_foreground_percent
        self.num_channels = num_channels

        self._ids = list(group.case_ids)
        # Use rank-dependent RNG so each DDP rank shuffles differently
        rng = np.random.RandomState(seed=(7919 + rank * 104729) % (2**32))
        rng.shuffle(self._ids)
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        group = self.group
        patch_size = np.array(group.patch_size_voxels)

        case_ids = []
        for _ in range(group.batch_size):
            if self._pos >= len(self._ids):
                np.random.shuffle(self._ids)
                self._pos = 0
            case_ids.append(self._ids[self._pos])
            self._pos += 1

        data_all = np.zeros(
            (group.batch_size, self.num_channels, *group.patch_size_voxels),
            dtype=np.float32,
        )
        seg_all = np.zeros(
            (group.batch_size, 1, *group.patch_size_voxels),
            dtype=np.int16,
        )
        spacings_all = []

        for j, case_id in enumerate(case_ids):
            dp, sp, cs = _load_and_extract_patch(
                case_id, self.preprocessed_dir, self.properties,
                patch_size, self.num_channels,
                self.oversample_foreground_percent,
            )
            data_all[j] = dp
            seg_all[j] = sp
            spacings_all.append(cs)

        batch = {
            'data': data_all,
            'seg': seg_all,
            'keys': case_ids,
            'spacings': spacings_all,
        }

        if self.transforms is not None:
            batch = self.transforms(**batch)

        return batch


# =============================================================================
# MultiResolutionLoader (public API)
# =============================================================================

class MultiResolutionLoader:
    """Multi-resolution loader using PyTorch DataLoader.

    Drop-in replacement for the batchgenerators-based MultiResolutionLoader.
    Uses a single shared worker pool via torch.utils.data.DataLoader instead
    of per-group MultiThreadedAugmenter instances.

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
        group_balance: float = 0.0,
        planned_batch_sizes: Optional[Dict[tuple, int]] = None,
        planned_val_batch_sizes: Optional[Dict[tuple, int]] = None,
        planned_patch_sizes_mm: Optional[Dict[tuple, tuple]] = None,
        rank: int = 0,
        world_size: int = 1,
        sync_groups: bool = False,
        decimation_max_thickness: float = 0.0,
        decimation_max_inplane: float = 4.0,
        decimation_inplane_ratio_limit: float = 1.1,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self._rank = rank
        self._world_size = world_size
        self._sync_groups = sync_groups
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
        self.planned_batch_sizes = planned_batch_sizes or {}
        # Optional validation-specific batch sizes (sized for inference memory,
        # typically larger than training).  When non-empty, _compute_batch_size
        # prefers these over `planned_batch_sizes`.
        self.planned_val_batch_sizes = planned_val_batch_sizes or {}
        # Optional per-group patch_size_mm overrides (planner shrunk these
        # groups to fit memory).  When set, the per-group mm is used instead
        # of the global ``patch_size_mm`` for that spacing's mm_to_voxels call.
        self.planned_patch_sizes_mm = planned_patch_sizes_mm or {}

        # --- Load metadata + enumerate synthetic decimation variants ---
        # Upxy stays on disk (cubic zoom is too slow to do per-batch); skip
        # variants are enumerated synthetically and applied via np.take on
        # load.  See discover_skip_files for the full policy.
        properties, subsampled_cases_list = discover_skip_files(
            self.preprocessed_dir,
            case_identifiers,
            subsample_weight,
            decimation_max_thickness=decimation_max_thickness,
            decimation_max_inplane=decimation_max_inplane,
            decimation_inplane_ratio_limit=decimation_inplane_ratio_limit,
        )

        self.subsampled_cases = set(subsampled_cases_list)
        self._all_properties = properties

        # --- Group cases by spacing ---
        self.spacing_groups = group_cases_by_spacing(
            properties, min_spacing=min_spacing,
            max_inplane_spacing=max_inplane_spacing,
            min_slice_thickness=min_slice_thickness,
            max_slice_thickness=max_slice_thickness,
        )

        n_original = len(case_identifiers)
        n_subsampled = len(subsampled_cases_list)

        print(f"\nMulti-resolution loader setup:")
        print(f"  Total cases: {n_original} original + {n_subsampled} subsampled")
        print(f"  Patch size: {patch_size_mm} mm")
        print(f"  Base batch size: {batch_size}")
        print(f"  Resolution groups: {len(self.spacing_groups)}")
        if subsample_weight > 0:
            print(f"  Preprocessed subsample weight: {subsample_weight:.2f}")
        if dynamic_batch_size:
            print(f"  Dynamic batch sizing: enabled (target={target_memory_mb:.0f} MB)")

        # --- Build group configs ---
        self.group_batch_sizes: Dict[tuple, int] = {}
        self.group_patch_sizes: Dict[tuple, tuple] = {}
        self.group_weights: List[Tuple[tuple, float]] = []
        self.min_loader_cases = min_loader_cases
        skipped_groups = []
        groups: List[_GroupConfig] = []
        # Track groups whose effective physical patch (voxels × spacing)
        # is materially smaller than the requested patch_size_mm.  This
        # happens when mm_to_voxels has to floor the patch to the nearest
        # multiple of the cumulative pool factor.
        shrunken_groups = []  # list of (spacing, effective_mm, shortfall_mm)
        SHRINK_TOL_MM = 4.0   # absolute tolerance before warning

        for spacing, cases in self.spacing_groups.items():
            if len(cases) < min_loader_cases:
                skipped_groups.append(
                    (spacing, f"too few cases ({len(cases)} < {min_loader_cases})")
                )
                continue

            # Per-group patch_size_mm if the planner shrunk this group;
            # otherwise the global request.
            group_patch_mm = self.planned_patch_sizes_mm.get(spacing, patch_size_mm)
            patch_size_voxels = mm_to_voxels(
                group_patch_mm, spacing, n_downsample, model_scale,
            )

            group_batch_size = self._compute_batch_size(
                patch_size_voxels, spacing=spacing,
            )
            # Use val plan if present; else train plan
            ref_planned = (self.planned_val_batch_sizes
                           if spacing in self.planned_val_batch_sizes
                           else self.planned_batch_sizes)
            if spacing in ref_planned:
                planned_bs = ref_planned[spacing]
                if group_batch_size != planned_bs:
                    print(f"    Using planned batch size {planned_bs} for {spacing}")
            elif self.planned_batch_sizes:
                # Spacing not in the planned dict (e.g. exceeded the planning
                # memory budget).  Drop it rather than falling back to bs=1 +
                # full mm patch — the planner already decided it doesn't fit,
                # and running it at full patch would OOM on sample.
                skipped_groups.append(
                    (spacing, f"not in plan ({len(cases)} cases dropped)")
                )
                continue

            n_orig = sum(1 for c in cases if c not in self.subsampled_cases)
            n_sub = sum(1 for c in cases if c in self.subsampled_cases)

            # Weight calculation (same as original)
            if n_orig > 0 and subsample_weight < 1.0:
                original_weight = (1.0 - subsample_weight) * n_orig / n_original
            else:
                original_weight = 0.0

            if n_sub > 0 and subsample_weight > 0:
                subsampled_weight_val = subsample_weight * n_sub / n_original
            else:
                subsampled_weight_val = 0.0

            weight = (original_weight + subsampled_weight_val) / group_batch_size

            gc = _GroupConfig(
                spacing=spacing,
                case_ids=cases,
                patch_size_voxels=patch_size_voxels,
                batch_size=group_batch_size,
                weight=weight,
                n_original=n_orig,
                n_subsampled=n_sub,
            )
            groups.append(gc)

            self.group_batch_sizes[spacing] = group_batch_size
            self.group_patch_sizes[spacing] = patch_size_voxels
            self.group_weights.append((spacing, weight))

            # Check for silent patch-size reduction: mm_to_voxels floors to
            # the nearest multiple of the cumulative pool factor.  When the
            # requested patch_size_mm can't be represented exactly, the
            # effective physical patch is smaller than requested.
            effective_mm = tuple(v * s for v, s in zip(patch_size_voxels, spacing))
            # Compare to the per-group request (which may itself be smaller
            # than the global mm if the planner shrunk this group).
            shortfall = tuple(
                max(0.0, rmm - emm) for rmm, emm in zip(group_patch_mm, effective_mm)
            )
            is_shrunken = any(sh > SHRINK_TOL_MM for sh in shortfall)
            if is_shrunken:
                shrunken_groups.append((spacing, effective_mm, shortfall))

            # Print group info
            spacing_str = tuple(f"{s:.3f}" for s in spacing)
            if n_sub > 0 and n_orig > 0:
                group_type = f"mixed ({n_orig} orig + {n_sub} sub)"
            elif n_sub > 0:
                group_type = "subsampled"
            else:
                group_type = "real"
            shrink_marker = ""
            if is_shrunken:
                eff_mm_str = tuple(f"{e:.0f}" for e in effective_mm)
                shrink_marker = f"  [SHRUNK → patch={eff_mm_str}mm]"
            print(
                f"    Spacing {spacing_str}: {len(cases)} cases, "
                f"patch={patch_size_voxels}, batch={group_batch_size}, "
                f"weight={weight:.3f} ({group_type}){shrink_marker}"
            )

        if skipped_groups:
            print(f"  Skipped {len(skipped_groups)} groups:")
            for spacing, reason in skipped_groups:
                spacing_str = tuple(f"{s:.3f}" for s in spacing)
                print(f"    {spacing_str}: {reason}")

        if shrunken_groups:
            print(f"\n  WARNING: {len(shrunken_groups)} groups have an effective "
                  f"patch smaller than requested {tuple(patch_size_mm)}mm.")
            print(f"    mm_to_voxels floors to the nearest multiple of the "
                  f"cumulative pool factor; groups where the request doesn't land "
                  f"on a multiple see a silent shrink.")
            print(f"    {'spacing':<24} {'effective patch (mm)':<26} {'shortfall (mm)'}")
            for spacing, eff_mm, shortfall in shrunken_groups:
                sp_str = tuple(f"{s:.3f}" for s in spacing)
                eff_str = '(' + ', '.join(f'{e:.0f}' for e in eff_mm) + ')'
                sh_str = '(' + ', '.join(f'{s:.0f}' for s in shortfall) + ')'
                print(f"    {str(sp_str):<24} {eff_str:<26} {sh_str}")
            print(f"    Fix: choose a patch_size_mm that's a multiple of the "
                  f"bottleneck spacing for the affected groups, or accept the "
                  f"reduction as intended.")

        # Expected patches per step (weighted average batch size across groups).
        # Used by DDP to convert patches_per_epoch → step count so all ranks
        # call backward() the same number of times.
        if self.group_weights:
            self.mean_batch_size = sum(
                w * self.group_batch_sizes[s]
                for s, w in self.group_weights
            )
        else:
            self.mean_batch_size = float(batch_size)

        # Normalize weights
        total_weight = sum(w for _, w in self.group_weights)
        if total_weight > 0:
            self.group_weights = [
                (s, w / total_weight) for s, w in self.group_weights
            ]

        # Group balancing: blend proportional with uniform
        if group_balance > 0 and len(self.group_weights) > 1:
            uniform = 1.0 / len(self.group_weights)
            self.group_weights = [
                (s, (1 - group_balance) * w + group_balance * uniform)
                for s, w in self.group_weights
            ]
            print(f"  Group balancing: {group_balance:.2f} (0=proportional, 1=uniform)")

        # Update group configs with normalized weights
        weight_map = dict(self.group_weights)
        for gc in groups:
            gc.weight = weight_map.get(gc.spacing, gc.weight)

        # Determine num_channels from first case
        first_case = case_identifiers[0]
        sample_data = np.load(self.preprocessed_dir / f"{first_case}.npy")
        num_channels = sample_data.shape[0]
        del sample_data

        # --- Create _GroupIterator wrappers (for group_loaders dict) ---
        self.group_loaders: Dict[tuple, _GroupIterator] = {}
        for gc in groups:
            self.group_loaders[gc.spacing] = _GroupIterator(
                gc, self.preprocessed_dir, properties, transforms,
                oversample_foreground_percent, num_channels, rank=rank,
            )

        # Store groups + constructor args for curriculum filtering / DataLoader rebuild
        self._groups = groups
        self._properties = properties
        self._transforms = transforms
        self._oversample_fg = oversample_foreground_percent
        self._num_channels = num_channels
        self._num_workers = num_workers

        # --- Create dataset + DataLoader ---
        if groups:
            dataset = _MultiResIterableDataset(
                groups, self.preprocessed_dir, properties, transforms,
                oversample_foreground_percent, num_channels, rank=rank,
                sync_groups=sync_groups,
            )
            loader_kwargs = {
                'batch_size': None,
                'num_workers': num_workers,
                'collate_fn': _identity_collate,
            }
            if num_workers > 0:
                loader_kwargs['persistent_workers'] = True
                loader_kwargs['prefetch_factor'] = 2
            self._dataloader = DataLoader(dataset, **loader_kwargs)
            self._iterator = iter(self._dataloader)
        else:
            self._dataloader = None
            self._iterator = None

    def _compute_batch_size(self, patch_size_voxels: tuple, spacing: tuple = None) -> int:
        """Compute batch size for a given patch size.

        Priority:
        1. ``planned_val_batch_sizes[spacing]`` (val loader, when set)
        2. ``planned_batch_sizes[spacing]`` (training-sized)
        3. dynamic estimation
        4. ``self.batch_size`` fallback
        """
        if spacing is not None and spacing in self.planned_val_batch_sizes:
            return self.planned_val_batch_sizes[spacing]
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

    def set_active_groups(self, predicate=None):
        """Restrict sampling to groups matching *predicate*.

        Parameters
        ----------
        predicate : callable((float,float,float)) -> bool, or None
            Receives canonical spacing tuple, returns True to keep.
            None restores all groups.

        Updates both the weight list and the internal DataLoader/dataset
        weights so that worker processes see the change.
        """
        if not hasattr(self, '_all_group_weights'):
            self._all_group_weights = list(self.group_weights)
            self._all_groups = list(self._groups)

        if predicate is None:
            active_groups = list(self._all_groups)
            self.group_weights = list(self._all_group_weights)
        else:
            active_idx = [i for i, (s, _) in enumerate(self._all_group_weights)
                          if predicate(s)]
            if not active_idx:
                return  # keep current state
            active_groups = [self._all_groups[i] for i in active_idx]
            self.group_weights = [self._all_group_weights[i] for i in active_idx]

        total = sum(w for _, w in self.group_weights)
        if total > 0:
            self.group_weights = [(s, w / total) for s, w in self.group_weights]

        # Recompute mean_batch_size over the active groups (used by DDP
        # step-count calculation to keep patches/epoch ≈ patches_per_epoch).
        if self.group_weights:
            self.mean_batch_size = sum(
                w * self.group_batch_sizes[s]
                for s, w in self.group_weights
            )

        # Rebuild DataLoader so worker processes get the new groups.
        # Mutating the dataset in-place doesn't propagate to persistent workers.
        if hasattr(self, '_dataloader') and self._dataloader is not None:
            # Shut down old workers
            del self._iterator
            del self._dataloader

            dataset = _MultiResIterableDataset(
                active_groups, self.preprocessed_dir, self._properties,
                self._transforms, self._oversample_fg, self._num_channels,
                rank=self._rank, sync_groups=self._sync_groups,
            )
            loader_kwargs = {
                'batch_size': None,
                'num_workers': self._num_workers,
                'collate_fn': _identity_collate,
            }
            if self._num_workers > 0:
                loader_kwargs['persistent_workers'] = True
                loader_kwargs['prefetch_factor'] = 2
            self._dataloader = DataLoader(dataset, **loader_kwargs)
            self._iterator = iter(self._dataloader)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[dict, tuple]:
        """Get next batch, sampling from resolution groups by weight."""
        if self._iterator is None:
            raise StopIteration

        batch, group_spacing = next(self._iterator)

        if 'spacings' in batch and len(batch['spacings']) > 0:
            idx = np.random.randint(len(batch['spacings']))
            selected_spacing = batch['spacings'][idx]
            selected_spacing = tuple(selected_spacing)
        else:
            selected_spacing = group_spacing

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

        loader_groups = []
        for spacing, weight in self.group_weights:
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

            loader_groups.append({
                'spacing': spacing,
                'patch_size_voxels': patch_voxels,
                'batch_size': batch_size,
                'estimated_memory_mb': round(est_mem, 1),
                'n_cases': len(cases),
                'group_type': group_type,
            })

        write_loader_config(
            filepath=filepath,
            args=args,
            groups=loader_groups,
            n_train_cases=n_train_cases,
            n_val_cases=n_val_cases,
            model_scale=self.model_scale,
        )
