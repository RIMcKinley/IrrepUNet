#!/usr/bin/env python
"""
Training script for e3nnUNet using nnUNet-style batchgenerators.

Supports multi-resolution training by grouping cases by spacing.
Uses nnUNet's data loading and augmentation pipeline for proper foreground oversampling.
"""

import argparse
import gc
import hashlib
import json
import math
from datetime import datetime
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast

from irrepunet.models import E3nnUNet, get_model_config, spacing_independent_state_dict, load_spacing_independent_state_dict, optimize_bottleneck_kernels, compute_kernel_sizes
from irrepunet.models.layers import _pool_factor
from irrepunet.training.losses import DiceCELoss, DeepSupervisionLoss
from irrepunet.data.spacing import group_cases_by_spacing
from irrepunet.data.dataloader import MultiResolutionLoader
from irrepunet.data.multi_resolution_loader import (
    estimate_memory_mb, estimate_batch_size,
    discover_skip_files, verify_receptive_field,
    compute_steps_through_pooling, mm_to_voxels,
)
from irrepunet.data.batchgen_transforms import get_training_transforms, get_validation_transforms


def downsample_seg_for_deep_supervision(
    seg: torch.Tensor,
    output_shapes: list
) -> list:
    """Create multi-scale segmentation targets for deep supervision.

    Resizes segmentation to match the spatial dimensions of each model output.
    Uses nearest neighbor interpolation.

    Parameters
    ----------
    seg : torch.Tensor
        Segmentation labels of shape (B, D, H, W)
    output_shapes : list of tuple
        List of spatial shapes (D, H, W) for each output, from coarsest to finest

    Returns
    -------
    list of torch.Tensor
        List of segmentations matching each output shape, from coarsest to finest
    """
    multi_scale_segs = []

    for target_shape in output_shapes:
        # Skip levels where any spatial dimension is 0 (can happen with
        # spatial splitting + deep networks)
        if any(s == 0 for s in target_shape):
            continue
        if tuple(seg.shape[1:]) == tuple(target_shape):
            # Same shape, no resizing needed
            multi_scale_segs.append(seg)
        else:
            # Resize to target shape using nearest neighbor
            seg_resized = seg.float().unsqueeze(1)  # Add channel dim: (B, 1, D, H, W)
            seg_resized = torch.nn.functional.interpolate(
                seg_resized,
                size=target_shape,
                mode='nearest'
            )
            seg_resized = seg_resized.squeeze(1).long()  # Remove channel dim
            multi_scale_segs.append(seg_resized)

    return multi_scale_segs



def _profile_cache_key(model_kwargs: dict, fp16: bool, deep_supervision: bool) -> str:
    """Compute a hash key for the GPU profile cache.

    The key captures all model/training parameters that affect GPU memory.
    Peak memory is deterministic for a given model config + patch + spacing,
    independent of GPU model.
    """
    # Extract the parameters that affect memory usage
    key_params = {
        'n_classes': model_kwargs['n_classes'],
        'in_channels': model_kwargs['in_channels'],
        'diameter': model_kwargs['diameter'],
        'num_radial_basis': model_kwargs['num_radial_basis'],
        'normalization': model_kwargs['normalization'],
        'n_base_filters': model_kwargs['n_base_filters'],
        'n_downsample': model_kwargs['n_downsample'],
        'equivariance': model_kwargs['equivariance'],
        'lmax': model_kwargs.get('lmax', 2),
        'dropout_prob': model_kwargs.get('dropout_prob', 0.0),
        'deep_supervision': deep_supervision,
        'max_features': model_kwargs.get('max_features', 320),
        'irrep_ratios': list(model_kwargs.get('irrep_ratios', (4, 2, 1))),
        'fill_to_max': model_kwargs.get('fill_to_max', False),
        'kernel_trim_threshold': model_kwargs.get('kernel_trim_threshold', 1.0),
        'kernel_trim_cross_section': model_kwargs.get('kernel_trim_cross_section', 0.0),
        'kernel_growth': model_kwargs.get('kernel_growth', 2.0),
        'sc_mode': model_kwargs.get('sc_mode', 'parallel'),
        'scale': model_kwargs.get('scale', 2.0),
        'fused_gate': model_kwargs.get('fused_gate', True),
        'backend': model_kwargs.get('backend', 'e3nn'),
        'fp16': fp16,
    }
    key_str = json.dumps(key_params, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _profile_cache_path(cache_key: str) -> Path:
    """Return path to cache file for a given model config hash."""
    cache_dir = Path(__file__).parent / '.gpu_profile_cache'
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f'profile_{cache_key}.json'


def _load_profile_cache(cache_path: Path) -> dict:
    """Load cached profiling results from disk.

    Returns dict mapping "patch|spacing" string keys to result dicts.
    """
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_profile_cache(cache_path: Path, cache: dict):
    """Save profiling results to disk cache."""
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)


def _cache_entry_key(patch_voxels: tuple, spacing: tuple) -> str:
    """Create a string key for a (patch, spacing) combo."""
    return f"{patch_voxels[0]}x{patch_voxels[1]}x{patch_voxels[2]}|{spacing[0]:.4f},{spacing[1]:.4f},{spacing[2]:.4f}"


def _activation_volume(spacing, scales, patch_voxels):
    """Total channel-weighted activation volume across encoder levels.

    Used to find the worst-case spacing within a group — the spacing
    that produces the largest feature maps and highest memory usage.
    Pool kernel = floor(scale/step); when step crosses scale/2,
    kernel flips from 2 to 1 (no pooling), causing up to 4x spatial blowup.
    """
    total = 0
    spatial = list(patch_voxels)
    steps = list(spacing)
    for i, scale in enumerate(scales):
        total += spatial[0] * spatial[1] * spatial[2] * (2 ** i)
        for j in range(3):
            k = _pool_factor(scale, steps[j])
            spatial[j] //= k
            steps[j] *= k
        spatial = [max(1, s) for s in spatial]
    total += spatial[0] * spatial[1] * spatial[2] * (2 ** len(scales))
    return total


def profile_memory_on_gpu(
    model_kwargs: dict,
    patch_configs: list,
    device: torch.device,
    fp16: bool = True,
    deep_supervision: bool = True,
    n_downsample: int = 4,
) -> dict:
    """Profile GPU memory for each (patch_voxels, spacing) at bs=1 and bs=2.

    Builds the actual model with the exact config that will be used for training,
    runs forward+backward+optimizer.step(), and records peak memory.

    Results are cached to disk keyed by model config hash. Subsequent calls
    with the same model config skip already-profiled (patch, spacing) combos.

    Parameters
    ----------
    model_kwargs : dict
        E3nnUNet constructor kwargs (must include n_classes, in_channels, etc.)
    patch_configs : list of (tuple, tuple)
        Each element is (patch_voxels, spacing) where patch_voxels and spacing
        are 3-tuples.
    device : torch.device
        GPU device to profile on.
    fp16 : bool
        Whether to use BF16 mixed precision.
    deep_supervision : bool
        Whether model uses deep supervision (affects loss computation).
    n_downsample : int
        Number of downsampling levels (for deep supervision loss).

    Returns
    -------
    dict
        Mapping (patch_voxels, spacing) -> {
            'mem_bs1': float (MB),
            'mem_bs2': float or None (MB),
            'per_item': float (MB),
            'status': 'ok' or 'oom',
        }
    """
    from irrepunet.training.losses import DiceCELoss, DeepSupervisionLoss

    results = {}
    if not patch_configs:
        return results

    # Load cache
    cache_key = _profile_cache_key(model_kwargs, fp16, deep_supervision)
    cache_path = _profile_cache_path(cache_key)
    cache = _load_profile_cache(cache_path)

    # Check which configs need profiling
    configs_to_profile = []
    n_cached = 0
    for patch_voxels, spacing in patch_configs:
        entry_key = _cache_entry_key(tuple(patch_voxels), tuple(spacing))
        if entry_key in cache:
            # Restore from cache
            cached = cache[entry_key]
            results[(tuple(patch_voxels), tuple(spacing))] = cached
            n_cached += 1
        else:
            configs_to_profile.append((patch_voxels, spacing))

    if n_cached > 0:
        print(f"\n  GPU profile cache: {n_cached}/{len(patch_configs)} configs cached (key={cache_key})")

    if not configs_to_profile:
        print(f"  All configs found in cache, skipping GPU profiling")
        return results

    # Sort by total voxels ascending — profile small patches first, skip on OOM
    sorted_configs = sorted(configs_to_profile, key=lambda pc: int(np.prod(pc[0])))

    # Get GPU total memory
    gpu_total_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)

    print(f"\n  GPU profiling on {device} ({gpu_total_mb:.0f} MB total)")
    print(f"  Profiling {len(sorted_configs)} new (patch, spacing) configurations...")

    # Build model once
    model = E3nnUNet(**model_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=3e-5)

    base_criterion = DiceCELoss(ce_weight=1.0, dice_weight=1.0)
    if deep_supervision:
        criterion = DeepSupervisionLoss(base_loss=base_criterion, n_scales=n_downsample)
    else:
        criterion = base_criterion

    amp_dtype = torch.bfloat16 if fp16 else torch.float32
    n_classes = model_kwargs['n_classes']

    # Track OOM spacings to skip larger patches at same spacing
    oom_spacings = set()

    for patch_voxels, spacing in sorted_configs:
        spacing_key = tuple(spacing)

        # Skip if we already OOM'd at this spacing with a smaller patch
        if spacing_key in oom_spacings:
            entry = {
                'mem_bs1': None, 'mem_bs2': None, 'per_item': None,
                'status': 'oom_skipped',
            }
            results[(tuple(patch_voxels), spacing_key)] = entry
            cache[_cache_entry_key(tuple(patch_voxels), spacing_key)] = entry
            continue

        patch_str = f"{patch_voxels[0]}x{patch_voxels[1]}x{patch_voxels[2]}"
        sp_str = f"({spacing[0]:.2f},{spacing[1]:.2f},{spacing[2]:.2f})"

        # Profile bs=1
        mem_bs1 = _profile_single_step(
            model, optimizer, criterion, device, amp_dtype,
            patch_voxels, spacing, n_classes, batch_size=1,
            deep_supervision=deep_supervision,
        )

        if mem_bs1 is None:
            print(f"    {sp_str} patch={patch_str}: OOM at bs=1")
            oom_spacings.add(spacing_key)
            entry = {
                'mem_bs1': None, 'mem_bs2': None, 'per_item': None,
                'status': 'oom',
            }
            results[(tuple(patch_voxels), spacing_key)] = entry
            cache[_cache_entry_key(tuple(patch_voxels), spacing_key)] = entry
            continue

        # Profile bs=2 if feasible (need ~2.2x headroom)
        mem_bs2 = None
        per_item = None
        if mem_bs1 * 2.2 < gpu_total_mb:
            mem_bs2 = _profile_single_step(
                model, optimizer, criterion, device, amp_dtype,
                patch_voxels, spacing, n_classes, batch_size=2,
                deep_supervision=deep_supervision,
            )
            if mem_bs2 is not None:
                per_item = mem_bs2 - mem_bs1
            else:
                # bs=2 OOM'd — use heuristic
                per_item = mem_bs1 * 0.85
        else:
            per_item = mem_bs1 * 0.85

        entry = {
            'mem_bs1': round(mem_bs1, 1),
            'mem_bs2': round(mem_bs2, 1) if mem_bs2 is not None else None,
            'per_item': round(per_item, 1),
            'status': 'ok',
        }
        results[(tuple(patch_voxels), spacing_key)] = entry
        cache[_cache_entry_key(tuple(patch_voxels), spacing_key)] = entry

        bs2_str = f", bs2={mem_bs2:.0f}" if mem_bs2 is not None else ""
        print(f"    {sp_str} patch={patch_str}: bs1={mem_bs1:.0f} MB{bs2_str}, per_item={per_item:.0f} MB")

    # Cleanup
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

    # Save updated cache
    _save_profile_cache(cache_path, cache)
    n_profiled = sum(1 for v in results.values() if v['status'] == 'ok')
    n_new = len(sorted_configs)
    print(f"  Profiling complete: {n_new} new configs profiled, {n_cached} from cache")
    print(f"  Cache saved: {cache_path} ({len(cache)} total entries)")

    return results


def _profile_single_step(
    model, optimizer, criterion, device, amp_dtype,
    patch_voxels, spacing, n_classes, batch_size=1,
    deep_supervision=True,
):
    """Run one forward+backward+step and return peak memory in MB, or None on OOM."""
    multi_seg = None
    try:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)

        x = torch.randn(batch_size, 1, *patch_voxels, device=device)
        seg = torch.randint(0, n_classes, (batch_size, *patch_voxels), device=device)
        spacing_t = tuple(float(s) for s in spacing)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            outputs = model(x, spacing=spacing_t)

            if deep_supervision and isinstance(outputs, (list, tuple)):
                # Build multi-scale targets
                output_shapes = [o.shape[2:] for o in outputs]
                multi_seg = downsample_seg_for_deep_supervision(seg, output_shapes)
                loss = criterion(outputs, multi_seg)
            else:
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[-1]
                loss = criterion(outputs, seg)

        loss.backward()
        optimizer.step()

        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        # Cleanup tensors
        del x, seg, outputs, loss, multi_seg
        torch.cuda.empty_cache()

        return peak_mb

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise  # Re-raise non-OOM RuntimeErrors
        # Clean up after OOM
        del multi_seg
        torch.cuda.empty_cache()
        gc.collect()
        return None


class ExperimentPlanner:
    """Plans training experiments without executing them."""

    def __init__(self, args):
        self.args = args
        self.preprocessed_dir = Path(args.preprocessed_dir)
        self.output_dir = Path(args.output_dir)
        self.patch_size_mm = tuple(args.patch_size_mm)

    def load_case_metadata(self, case_ids: List[str]) -> Dict[str, dict]:
        """Load preprocessing metadata for all cases.

        Skips cases with missing metadata files.

        Returns:
            Dict mapping case_id to {'spacing': tuple, ...}
        """
        metadata = {}
        missing = []
        for case_id in case_ids:
            pkl_path = self.preprocessed_dir / f"{case_id}.pkl"
            if not pkl_path.exists():
                missing.append(case_id)
                continue
            with open(pkl_path, 'rb') as f:
                metadata[case_id] = pickle.load(f)

        if missing:
            print(f"Warning: {len(missing)} cases with missing metadata (skipped):")
            for case_id in missing[:5]:  # Show first 5
                print(f"    {case_id}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")

        return metadata

    def compute_patch_size_voxels(self, spacing: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert mm patch size to voxels for a given spacing.

        Uses the same algorithm as MultiResolutionLoader._mm_to_voxels():
        works backwards from target RF through dynamic pooling to ensure
        planner and loader compute identical patch sizes.

        Args:
            spacing: Voxel spacing in mm (D, H, W)

        Returns:
            Patch size in voxels (D, H, W)
        """
        model_scale = getattr(self.args, 'scale', 2.0)
        return mm_to_voxels(
            self.patch_size_mm, spacing,
            self.args.n_downsample, model_scale,
        )

    def compute_batch_size(self, patch_voxels: Tuple[int, int, int],
                           spacing: Tuple[float, float, float] = None) -> int:
        """Compute batch size from profiled or parametric memory estimate.

        Uses profiled GPU measurements when available, falls back to
        parametric estimation otherwise.

        Args:
            patch_voxels: Patch size in voxels (D, H, W)
            spacing: Voxel spacing in mm

        Returns:
            Batch size for this patch size
        """
        if not self.args.dynamic_batch_size:
            return self.args.batch_size

        if self.args.target_memory_mb <= 0:
            raise ValueError(
                "dynamic_batch_size requires --target_memory_mb > 0"
            )

        # Check for profiled memory first
        if hasattr(self, '_profiled_memory') and spacing is not None:
            profile_sp = getattr(self, '_profile_spacings', {}).get(tuple(spacing), tuple(spacing))
            key = (tuple(patch_voxels), tuple(profile_sp))
            profile = self._profiled_memory.get(key)
            if profile is not None and profile['status'] == 'ok':
                mem_bs1 = profile['mem_bs1']
                per_item = profile['per_item']
                overhead = mem_bs1 - per_item
                # Reserve 15% safety margin for runtime overhead
                # (CUDA context ~4.5GB, loader workers, fragmentation)
                effective_target = self.args.target_memory_mb * 0.85
                # Binary search for largest batch that fits, accounting for
                # non-linear memory scaling at high batch sizes (~1% per item)
                batch = 1
                for b in range(1, self.args.max_batch_size + 1):
                    # Non-linear correction: per-item cost grows ~1% per
                    # additional item due to allocator overhead & autograd
                    corrected_per = per_item * (1.0 + 0.01 * (b - 1))
                    est_mem = overhead + b * corrected_per
                    if est_mem <= effective_target:
                        batch = b
                    else:
                        break
                return max(1, batch)

        # Fallback to parametric estimation
        return estimate_batch_size(
            patch_size=patch_voxels,
            n_base_filters=self.args.n_base_filters,
            target_memory_mb=self.args.target_memory_mb,
            min_batch=1,
            max_batch=self.args.max_batch_size,
            n_downsample=self.args.n_downsample,
            fp16=self.args.fp16,
        )

    def _create_group_config(
        self,
        group_id: int,
        spacing: Tuple[float, float, float],
        cases: List[str],
        subsampled_cases: Optional[set] = None,
        n_spatial_splits: int = 1,
    ) -> Dict:
        """Create configuration for a single loader group."""
        # Compute patch size in voxels
        patch_voxels = self.compute_patch_size_voxels(spacing)

        # Compute batch size (uses profiled memory when available)
        batch_size = self.compute_batch_size(patch_voxels, spacing=spacing)

        # Get memory info — prefer profiled, fall back to parametric
        measured_bs1 = None
        measured_bs2 = None
        if hasattr(self, '_profiled_memory'):
            profile_sp = getattr(self, '_profile_spacings', {}).get(tuple(spacing), tuple(spacing))
            key = (tuple(patch_voxels), tuple(profile_sp))
            profile = self._profiled_memory.get(key)
            if profile is not None and profile['status'] == 'ok':
                measured_bs1 = profile['mem_bs1']
                measured_bs2 = profile['mem_bs2']
                mem_mb = measured_bs1  # Report bs=1 measured memory
            elif profile is not None and profile['status'] in ('oom', 'oom_skipped'):
                mem_mb = float('inf')
            else:
                mem_mb = estimate_memory_mb(
                    patch_size=patch_voxels,
                    n_base_filters=self.args.n_base_filters,
                    batch_size=batch_size,
                    n_downsample=self.args.n_downsample,
                    fp16=self.args.fp16,
                )
        else:
            mem_mb = estimate_memory_mb(
                patch_size=patch_voxels,
                n_base_filters=self.args.n_base_filters,
                batch_size=batch_size,
                n_downsample=self.args.n_downsample,
                fp16=self.args.fp16,
            )

        # Determine group type
        if subsampled_cases and any(c in subsampled_cases for c in cases):
            n_sub = sum(1 for c in cases if c in subsampled_cases)
            if n_sub == len(cases):
                group_type = "subsampled"
            else:
                group_type = "mixed"
        else:
            group_type = "real"

        result = {
            "group_id": group_id,
            "spacing": list(spacing),
            "patch_size_voxels": list(patch_voxels),
            "patch_size_mm": list(self.patch_size_mm),
            "batch_size": batch_size,
            "n_spatial_splits": n_spatial_splits,
            "estimated_memory_mb": round(mem_mb, 1) if mem_mb != float('inf') else None,
            "case_ids": cases,
            "group_type": group_type,
        }

        # Add profiled spacing if it differs from canonical
        profile_sp = getattr(self, '_profile_spacings', {}).get(tuple(spacing))
        if profile_sp is not None:
            result["profile_spacing"] = list(profile_sp)

        # Add profiled memory if available
        if measured_bs1 is not None:
            result["measured_memory_bs1"] = measured_bs1
        if measured_bs2 is not None:
            result["measured_memory_bs2"] = measured_bs2

        return result

    def plan_experiment(self) -> Dict:
        """Main planning logic.

        Returns:
            Dict with keys: 'config' (complete config dict), 'timestamp', 'git_commit'
        """
        # Load splits
        splits_path = self.preprocessed_dir / 'splits_final.json'
        if not splits_path.exists():
            raise FileNotFoundError(f"splits_final.json not found: {splits_path}")

        with open(splits_path) as f:
            splits = json.load(f)

        fold = self.args.fold
        if fold >= len(splits):
            raise ValueError(f"Fold {fold} not found. Available: 0-{len(splits)-1}")

        train_cases = splits[fold]['train']
        val_cases = splits[fold]['val']

        print(f"Planning experiment for fold {fold}")
        print(f"  Train cases (before filtering): {len(train_cases)}")
        print(f"  Val cases (before filtering): {len(val_cases)}")

        # Load metadata
        all_cases = train_cases + val_cases
        metadata = self.load_case_metadata(all_cases)

        # Filter to cases that have metadata
        available_cases = set(metadata.keys())
        train_cases = [c for c in train_cases if c in available_cases]
        val_cases = [c for c in val_cases if c in available_cases]

        print(f"  Train cases (after filtering): {len(train_cases)}")
        print(f"  Val cases (after filtering): {len(val_cases)}")

        # Discover skip-downsampled files if enabled
        all_properties = {}
        subsampled_cases = set()

        if self.args.subsample_weight > 0:
            all_properties, subsampled_list = discover_skip_files(
                self.preprocessed_dir,
                train_cases,
                self.args.subsample_weight
            )
            subsampled_cases = set(subsampled_list)
            print(f"  Subsampled variants found: {len(subsampled_cases)}")

            # Re-group cases with skip variants included
            min_spacing = getattr(self.args, 'min_spacing', 0.0)
            max_inplane_spacing = getattr(self.args, 'max_inplane_spacing', 0.0)
            min_slice_thickness = getattr(self.args, 'min_slice_thickness', 0.0)
            max_slice_thickness = getattr(self.args, 'max_slice_thickness', 0.0)
            spacing_groups = group_cases_by_spacing(
                all_properties, min_spacing=min_spacing, max_inplane_spacing=max_inplane_spacing,
                min_slice_thickness=min_slice_thickness, max_slice_thickness=max_slice_thickness)
        else:
            # Use original metadata only
            train_metadata = {cid: metadata[cid] for cid in train_cases}
            min_spacing = getattr(self.args, 'min_spacing', 0.0)
            max_inplane_spacing = getattr(self.args, 'max_inplane_spacing', 0.0)
            min_slice_thickness = getattr(self.args, 'min_slice_thickness', 0.0)
            max_slice_thickness = getattr(self.args, 'max_slice_thickness', 0.0)
            spacing_groups = group_cases_by_spacing(
                train_metadata, min_spacing=min_spacing, max_inplane_spacing=max_inplane_spacing,
                min_slice_thickness=min_slice_thickness, max_slice_thickness=max_slice_thickness)
            all_properties = train_metadata

        print(f"  Resolution groups: {len(spacing_groups)}")
        for spacing, cases in sorted(spacing_groups.items()):
            n_sub = sum(1 for c in cases if c in subsampled_cases)
            if n_sub > 0:
                print(f"    {spacing}: {len(cases)} cases ({n_sub} subsampled)")
            else:
                print(f"    {spacing}: {len(cases)} cases")

        # Build loader configuration
        loader_groups = []
        skipped_groups = []
        reference_spacing = None
        n_total_original_cases = len(train_cases)
        min_loader_cases = getattr(self.args, 'min_loader_cases', 2)

        # Filter groups by min_loader_cases first
        eligible_groups = []
        for spacing, cases in sorted(spacing_groups.items()):
            if len(cases) < min_loader_cases:
                skipped_groups.append((spacing, f"too few cases ({len(cases)} < {min_loader_cases})"))
            else:
                eligible_groups.append((spacing, cases))

        # Collect unique (patch_voxels, spacing) for GPU profiling
        # Profile at worst-case actual spacing per group (maximizes activation volume)
        scale = getattr(self.args, 'scale', 2.0)
        scales = [scale * (2 ** i) for i in range(self.args.n_downsample)]

        self._profile_spacings = {}  # canonical_spacing → profile_spacing
        patch_configs = []
        worst_case_info = []
        for spacing, cases in eligible_groups:
            patch_voxels = self.compute_patch_size_voxels(spacing)
            canonical_vol = _activation_volume(spacing, scales, patch_voxels)

            # Find worst-case actual spacing in this group
            worst_spacing = spacing
            worst_vol = canonical_vol
            actual_spacings = set()
            for cid in cases:
                props = all_properties.get(cid)
                if props is not None:
                    sp = tuple(sorted(props['spacing']))
                    actual_spacings.add(sp)
            for sp in actual_spacings:
                # Compute patch_voxels at canonical spacing (matches training)
                vol = _activation_volume(sp, scales, patch_voxels)
                if vol > worst_vol:
                    worst_vol = vol
                    worst_spacing = sp

            if worst_spacing != spacing:
                ratio = worst_vol / canonical_vol
                self._profile_spacings[spacing] = worst_spacing
                worst_case_info.append((spacing, worst_spacing, ratio))

            patch_configs.append((patch_voxels, worst_spacing))

        if worst_case_info:
            print(f"\n  Worst-case spacing analysis: {len(worst_case_info)}/{len(eligible_groups)} groups affected")
            # Show top 10 by ratio
            for canonical, profile, ratio in sorted(worst_case_info, key=lambda x: -x[2])[:10]:
                print(f"    {canonical}: profile at {tuple(round(s, 4) for s in profile)}, {ratio:.2f}x activation volume")
            if len(worst_case_info) > 10:
                print(f"    ... and {len(worst_case_info) - 10} more")

        # Run GPU profiling if CUDA is available
        if patch_configs and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')

            # Determine n_classes from data
            sample_seg = np.load(self.preprocessed_dir / f"{train_cases[0]}_seg.npy")
            n_classes = int(sample_seg.max()) + 1

            # Use a reference spacing for model init (doesn't matter which — buffers recomputed)
            ref_spacing = patch_configs[0][1]

            model_kwargs = dict(
                n_classes=n_classes,
                in_channels=1,
                diameter=self.args.diameter,
                num_radial_basis=self.args.num_radial_basis,
                spacing=ref_spacing,
                normalization=self.args.normalization,
                n_base_filters=self.args.n_base_filters,
                n_downsample=self.args.n_downsample,
                equivariance=self.args.equivariance,
                lmax=len(self.args.irrep_ratios) - 1,
                dropout_prob=self.args.dropout,
                cutoff=True,
                deep_supervision=self.args.deep_supervision,
                max_features=self.args.max_features,
                irrep_ratios=tuple(self.args.irrep_ratios),
                fill_to_max=self.args.fill_to_max,
                activation=getattr(self.args, 'activation', 'softplus'),
                kernel_trim_threshold=getattr(self.args, 'kernel_trim_threshold', 1.0),
                kernel_trim_cross_section=getattr(self.args, 'kernel_trim_cross_section', 0.0),
                kernel_growth=getattr(self.args, 'kernel_growth', 2.0),
                sequential_sc=getattr(self.args, 'sequential_sc', False),
                sc_mode=getattr(self.args, 'sc_mode', None),
                scale=getattr(self.args, 'scale', 2.0),
                fused_gate=getattr(self.args, 'fused_gate', True),
                backend=getattr(self.args, 'backend', 'e3nn'),
                pyramid=_build_pyramid_config(self.args) if getattr(self.args, 'pyramid', None) else False,
            )

            self._profiled_memory = profile_memory_on_gpu(
                model_kwargs=model_kwargs,
                patch_configs=patch_configs,
                device=device,
                fp16=self.args.fp16,
                deep_supervision=self.args.deep_supervision,
                n_downsample=self.args.n_downsample,
            )
        else:
            print("\n  WARNING: CUDA not available, using parametric memory estimates")
            self._profiled_memory = {}

        # Rescue groups where worst-case OOMs but canonical fits.
        # Profile at canonical spacing, filter cases by estimated memory,
        # and keep only cases whose actual spacing fits within the budget.
        effective_target = self.args.target_memory_mb * 0.85
        rescue_groups = []  # (idx, spacing, cases, patch_voxels) for groups to rescue
        for idx, (spacing, cases) in enumerate(eligible_groups):
            profile_sp = self._profile_spacings.get(spacing, spacing)
            if profile_sp == spacing:
                continue  # canonical == worst-case, no rescue possible
            patch_voxels = self.compute_patch_size_voxels(spacing)
            key = (tuple(patch_voxels), tuple(profile_sp))
            profile = self._profiled_memory.get(key)
            if profile and (profile['status'] in ('oom', 'oom_skipped')
                            or (profile['status'] == 'ok' and profile['mem_bs1'] > effective_target)):
                rescue_groups.append((idx, spacing, cases, patch_voxels))

        if rescue_groups and torch.cuda.is_available():
            # Batch-profile canonical spacings
            canonical_configs = []
            for _, spacing, _, patch_voxels in rescue_groups:
                canonical_key = (tuple(patch_voxels), tuple(spacing))
                if canonical_key not in self._profiled_memory:
                    canonical_configs.append((patch_voxels, spacing))

            if canonical_configs:
                print(f"\n  Re-profiling {len(canonical_configs)} groups at canonical spacing (worst-case OOM)...")
                canonical_profiles = profile_memory_on_gpu(
                    model_kwargs=model_kwargs,
                    patch_configs=canonical_configs,
                    device=device,
                    fp16=self.args.fp16,
                    deep_supervision=self.args.deep_supervision,
                    n_downsample=self.args.n_downsample,
                )
                self._profiled_memory.update(canonical_profiles)

            # Filter cases per rescued group
            rescued_info = []
            for idx, spacing, cases, patch_voxels in rescue_groups:
                canonical_key = (tuple(patch_voxels), tuple(spacing))
                canonical_profile = self._profiled_memory.get(canonical_key)
                if not canonical_profile or canonical_profile['status'] != 'ok':
                    continue  # canonical also OOMs, can't rescue

                per_item = canonical_profile['per_item']
                overhead = canonical_profile['mem_bs1'] - per_item
                vol_canonical = _activation_volume(spacing, scales, patch_voxels)

                # Filter: keep cases whose estimated bs=1 memory fits
                kept_cases = []
                max_kept_ratio = 1.0
                excluded = 0
                for cid in cases:
                    props = all_properties.get(cid)
                    if props is None:
                        kept_cases.append(cid)
                        continue
                    sp_actual = tuple(sorted(props['spacing']))
                    vol_actual = _activation_volume(sp_actual, scales, patch_voxels)
                    ratio = vol_actual / vol_canonical
                    est_mem = overhead + per_item * ratio
                    if est_mem <= effective_target:
                        kept_cases.append(cid)
                        max_kept_ratio = max(max_kept_ratio, ratio)
                    else:
                        excluded += 1

                if not kept_cases:
                    continue  # all cases excluded

                # Create synthetic profile scaled by worst kept ratio
                scaled_per_item = per_item * max_kept_ratio
                scaled_bs1 = overhead + scaled_per_item
                scaled_bs2 = overhead + 2 * scaled_per_item
                self._profiled_memory[canonical_key] = {
                    'mem_bs1': round(scaled_bs1, 1),
                    'mem_bs2': round(scaled_bs2, 1),
                    'per_item': round(scaled_per_item, 1),
                    'status': 'ok',
                }

                # Point lookups to canonical spacing
                if spacing in self._profile_spacings:
                    del self._profile_spacings[spacing]

                # Update eligible_groups with filtered case list
                eligible_groups[idx] = (spacing, kept_cases)
                rescued_info.append((spacing, len(cases), len(kept_cases), excluded, max_kept_ratio))

            if rescued_info:
                print(f"\n  Rescued {len(rescued_info)} groups by filtering outlier spacings:")
                for sp, n_orig, n_kept, n_excl, ratio in rescued_info:
                    print(f"    {sp}: kept {n_kept}/{n_orig} cases ({n_excl} excluded), "
                          f"worst kept ratio {ratio:.2f}x")

        # Create group configs using profiled memory
        for spacing, cases in eligible_groups:
            patch_voxels = self.compute_patch_size_voxels(spacing)
            profile_sp = self._profile_spacings.get(spacing, spacing)
            key = (tuple(patch_voxels), tuple(profile_sp))
            profile = self._profiled_memory.get(key)

            # Check if bs=1 OOMs
            if profile and profile['status'] in ('oom', 'oom_skipped'):
                # Determine spatial splits needed
                n_spatial_splits = 1
                test_patch = list(patch_voxels)
                fits = False
                while n_spatial_splits <= 16:
                    n_spatial_splits *= 2
                    max_dim_idx = test_patch.index(max(test_patch))
                    test_patch[max_dim_idx] = max(
                        self.args.pooling_factor,
                        test_patch[max_dim_idx] // 2
                    )
                    # Check if the split patch was profiled and fits
                    split_key = (tuple(test_patch), tuple(spacing))
                    split_profile = self._profiled_memory.get(split_key)
                    if split_profile and split_profile['status'] == 'ok':
                        fits = True
                        break
                    elif split_profile is None:
                        # Not profiled — use parametric fallback
                        est = estimate_memory_mb(
                            patch_size=tuple(test_patch),
                            n_base_filters=self.args.n_base_filters,
                            batch_size=1,
                            n_downsample=self.args.n_downsample,
                            fp16=self.args.fp16,
                        )
                        if est <= self.args.target_memory_mb:
                            fits = True
                            break

                if not fits:
                    skipped_groups.append((spacing, f"OOM at bs=1, needs >{16} spatial splits"))
                    continue
            elif profile and profile['status'] == 'ok':
                # Check if profiled bs=1 exceeds target memory (with 15% safety margin)
                effective_target = self.args.target_memory_mb * 0.85
                if profile['mem_bs1'] > effective_target:
                    skipped_groups.append((spacing, f"OOM: measured {profile['mem_bs1']:.0f} MB > effective target {effective_target:.0f} MB (15% safety margin)"))
                    continue
                n_spatial_splits = 1
            else:
                n_spatial_splits = 1

            real_config = self._create_group_config(
                group_id=len(loader_groups),
                spacing=spacing,
                cases=cases,
                subsampled_cases=subsampled_cases if subsampled_cases else None,
                n_spatial_splits=n_spatial_splits,
            )
            loader_groups.append(real_config)

            if reference_spacing is None:
                reference_spacing = spacing

        # Report skipped groups
        if skipped_groups:
            print(f"\nSkipped {len(skipped_groups)} loader groups:")
            for spacing, reason in skipped_groups:
                spacing_str = tuple(f"{s:.2f}" for s in spacing)
                print(f"  {spacing_str}: {reason}")

        # Bottleneck kernel optimization
        bottleneck_kernel = getattr(self.args, 'bottleneck_kernel', 0)
        bottleneck_diagnostics = None
        if bottleneck_kernel > 0:
            group_spacings = [sp for sp in spacing_groups.keys()
                              if any(sp == tuple(float(v) for v in g['spacing'])
                                     for g in loader_groups)]
            if group_spacings:
                print(f"\nOptimizing for {bottleneck_kernel}^3 bottleneck kernels...")
                bn_result = optimize_bottleneck_kernels(
                    spacings=group_spacings,
                    n_downsample=self.args.n_downsample,
                    patch_size_mm=min(self.patch_size_mm),
                    diameter=None,  # auto-derive from L0 constraint
                    target_bottleneck_kernel=bottleneck_kernel,
                    kernel_growth=getattr(self.args, 'kernel_growth', 2.0),
                )
                # Override args with optimized values
                self.args.diameter = bn_result['diameter']
                self.args.kernel_trim_threshold = bn_result['kernel_trim_threshold']
                self.args.kernel_trim_cross_section = bn_result.get('kernel_trim_cross_section', 0.0)
                self.args.scale = bn_result['scale']
                self.args.kernel_growth = bn_result['kernel_growth']
                bottleneck_diagnostics = bn_result['diagnostics']
                if not bn_result['verified']:
                    print("  WARNING: verification failed — some kernels do not "
                          f"match target {bottleneck_kernel}^3")
            else:
                print("  Warning: no loader groups available for bottleneck optimization")

        # Build complete config
        config = {
            "version": "1.0",
            "created": datetime.now().isoformat(),

            "data": {
                "preprocessed_dir": str(self.preprocessed_dir),
                "fold": fold,
                "train_cases": train_cases,
                "val_cases": val_cases,
            },

            "model": {
                "n_base_filters": self.args.n_base_filters,
                "n_downsample": self.args.n_downsample,
                "diameter": self.args.diameter,
                "num_radial_basis": self.args.num_radial_basis,
                "equivariance": self.args.equivariance,
                "normalization": self.args.normalization,
                "activation": getattr(self.args, 'activation', 'softplus'),
                "dropout": self.args.dropout,
                "max_features": self.args.max_features,
                "irrep_ratios": self.args.irrep_ratios,
                "fill_to_max": self.args.fill_to_max,
                "kernel_trim_threshold": getattr(self.args, 'kernel_trim_threshold', 1.0),
                "kernel_trim_cross_section": getattr(self.args, 'kernel_trim_cross_section', 0.0),
                "kernel_growth": getattr(self.args, 'kernel_growth', 2.0),
                "sequential_sc": getattr(self.args, 'sequential_sc', False),
                "sc_mode": getattr(self.args, 'sc_mode', 'parallel'),
                "scale": getattr(self.args, 'scale', 2.0),
                "fused_gate": getattr(self.args, 'fused_gate', True),
                "pool_mode": getattr(self.args, 'pool_mode', 'maxpool3d'),
                "backend": getattr(self.args, 'backend', 'e3nn'),
                "pyramid": getattr(self.args, 'pyramid', None),
                "pyramid_decay": getattr(self.args, 'pyramid_decay', 1.0),
            },

            "training": {
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                "learning_rate": self.args.lr,
                "weight_decay": self.args.weight_decay,
                "grad_clip": self.args.grad_clip,
                "patch_size_mm": list(self.patch_size_mm),
                "patches_per_epoch": self.args.patches_per_epoch,
                "val_patches": self.args.val_patches,
                "foreground_oversample": self.args.foreground_oversample,
                "dynamic_batch_size": self.args.dynamic_batch_size,
                "target_memory_mb": self.args.target_memory_mb,
                "min_batch_size": self.args.min_batch_size,
                "max_batch_size": self.args.max_batch_size,
                "pooling_factor": self.args.pooling_factor,
                "resolution_jitter_sigma": self.args.resolution_jitter_sigma,
                "scale_jitter_std": getattr(self.args, 'scale_jitter_std', 0.0),
                "num_workers": self.args.num_workers,
                "init_checkpoint": getattr(self.args, 'init_checkpoint', None),
                "wandb": getattr(self.args, 'wandb', False),
                "wandb_project": getattr(self.args, 'wandb_project', 'irrepunet'),
                "wandb_name": getattr(self.args, 'wandb_name', None),
            },

            "augmentation": {
                "disable_spatial": self.args.disable_spatial,
                "disable_mirroring": self.args.disable_mirroring,
                "subsample_weight": self.args.subsample_weight,
                "min_spacing": getattr(self.args, 'min_spacing', 0.0),
                "max_inplane_spacing": getattr(self.args, 'max_inplane_spacing', 0.0),
                "min_slice_thickness": getattr(self.args, 'min_slice_thickness', 0.0),
                "max_slice_thickness": getattr(self.args, 'max_slice_thickness', 0.0),
                "min_loader_cases": getattr(self.args, 'min_loader_cases', 2),
                "superres_training": getattr(self.args, 'superres_training', False),
                "superres_weight": getattr(self.args, 'superres_weight', 0.1),
                "group_balance": getattr(self.args, 'group_balance', 0.0),
                "bias_field": getattr(self.args, 'bias_field', True),
                "curriculum": getattr(self.args, 'curriculum', None),
                "curriculum_bs_tiers": getattr(self.args, 'curriculum_bs_tiers', None),
                "curriculum_phase_len": getattr(self.args, 'curriculum_phase_len', 30),
                "use_group_spacing": getattr(self.args, 'use_group_spacing', False),
            },

            "hardware": {
                "gpu": self.args.gpu,
                "fp16": self.args.fp16,
                "deep_supervision": self.args.deep_supervision,
                "no_background_dice": getattr(self.args, 'no_background_dice', False),
                "batch_dice": getattr(self.args, 'batch_dice', False),
            },

            "loader_groups": loader_groups,
            "reference_spacing": list(reference_spacing) if reference_spacing else None,
        }

        return {
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "git_commit": self._get_git_commit(),
        }

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=self.preprocessed_dir.parent,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def save_configs(self, plan_result: Dict):
        """Write config.json, loader_config.txt, run_training.sh"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config = plan_result["config"]
        timestamp = plan_result["timestamp"]
        git_commit = plan_result["git_commit"]

        # Save config.json
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config.json")

        # Save enhanced loader_config.txt
        self._save_loader_config(config, timestamp, git_commit)
        print(f"Saved loader_config.txt")

        # Surface silent patch-size reduction on stdout too
        patch_mm = tuple(self.args.patch_size_mm)
        shrunken = []
        for g in config['loader_groups']:
            raw_sp = g['spacing']
            if isinstance(raw_sp, (list, tuple)) and len(raw_sp) == 3 and raw_sp[0] == 'superres':
                continue
            sp = tuple(float(s) for s in raw_sp)
            pv = tuple(g['patch_size_voxels'])
            eff = tuple(v * s for v, s in zip(pv, sp))
            sh = tuple(max(0.0, r - e) for r, e in zip(patch_mm, eff))
            if any(x > 4.0 for x in sh):
                shrunken.append((sp, eff, sh))
        if shrunken:
            print(f"\nWARNING: {len(shrunken)} groups have an effective patch "
                  f"smaller than the requested {patch_mm}mm "
                  f"(silent floor in mm_to_voxels):")
            for sp, eff, sh in shrunken:
                sp_s = f'({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})'
                eff_s = f'({eff[0]:.0f}, {eff[1]:.0f}, {eff[2]:.0f})'
                sh_s = f'({sh[0]:.0f}, {sh[1]:.0f}, {sh[2]:.0f})'
                print(f"  {sp_s:<24}  effective={eff_s}  shortfall={sh_s}")
            print(f"  See loader_config.txt for full details.\n")

        # Generate run_training.sh
        self._generate_run_script(config)
        print(f"Saved run_training.sh")

    def _save_loader_config(self, config: Dict, timestamp: str, git_commit: Optional[str]):
        """Generate loader_config.txt using shared writer."""
        output_path = self.output_dir / "loader_config.txt"

        # Build standardized groups list
        groups = []
        for g in config['loader_groups']:
            group_info = {
                'spacing': g['spacing'],
                'patch_size_voxels': g['patch_size_voxels'],
                'batch_size': g['batch_size'],
                'estimated_memory_mb': g.get('estimated_memory_mb') or g.get('measured_memory_bs1', 0),
                'n_cases': len(g['case_ids']),
                'n_spatial_splits': g.get('n_spatial_splits', 1),
                'group_type': g.get('group_type', 'real'),
            }
            if 'measured_memory_bs1' in g:
                group_info['measured_memory_bs1'] = g['measured_memory_bs1']
            groups.append(group_info)

        # Build curriculum phases if configured
        curriculum_phases = None
        if getattr(self.args, 'curriculum_bs_tiers', None):
            curriculum_phases = _build_bs_curriculum(
                config['loader_groups'], self.args.curriculum_bs_tiers,
                getattr(self.args, 'curriculum_phase_len', 30),
                total_epochs=self.args.epochs)

        write_loader_config(
            filepath=str(output_path),
            args=self.args,
            groups=groups,
            n_train_cases=len(config['data']['train_cases']),
            n_val_cases=len(config['data']['val_cases']),
            curriculum_phases=curriculum_phases,
        )

    def _generate_run_script(self, config: Dict):
        """Generate run_training.sh script."""
        output_path = self.output_dir / "run_training.sh"

        lines = []
        lines.append("#!/bin/bash")
        lines.append("# Generated by train.py --plan_only")
        lines.append(f"# Created: {datetime.now().isoformat()}")
        lines.append("")
        lines.append("set -e  # Exit on error")
        lines.append("")
        lines.append("# Activate environment")
        lines.append("source ~/.bashrc")
        lines.append("conda activate e3nn")
        lines.append("")
        lines.append("# Change to project directory")
        lines.append(f"cd {self.output_dir.parent}")
        lines.append("")
        lines.append("# Run training with config")
        lines.append(f"python train.py --config {self.output_dir}/config.json")
        lines.append("")
        lines.append("echo \"Training complete!\"")
        lines.append("")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        # Make executable
        os.chmod(output_path, 0o755)


# =============================================================================
# Shared loader config utilities
# =============================================================================

def write_loader_config(filepath, args, groups, n_train_cases, n_val_cases, model_scale=2.0,
                        curriculum_phases=None):
    """Write unified loader_config.txt used by both --plan_only and training.

    Parameters
    ----------
    filepath : str
        Output file path
    args : argparse.Namespace
        Training arguments
    groups : list of dict
        Each dict has keys: spacing, patch_size_voxels, batch_size,
        estimated_memory_mb, n_cases, group_type
    n_train_cases : int
        Number of training cases
    n_val_cases : int
        Number of validation cases
    model_scale : float
        Model pooling scale for RF verification (default 2.0)
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT CONFIGURATION")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    # Data
    lines.append("DATA")
    lines.append("-" * 80)
    lines.append(f"Fold: {args.fold}")
    lines.append(f"Training cases: {n_train_cases}")
    lines.append(f"Validation cases: {n_val_cases}")
    lines.append(f"Preprocessed dir: {args.preprocessed_dir}")
    lines.append("")

    # Model
    lines.append("MODEL")
    lines.append("-" * 80)
    lines.append(f"Base filters: {args.n_base_filters}")
    lines.append(f"Downsampling levels: {args.n_downsample}")
    lines.append(f"Equivariance: {args.equivariance}")
    lines.append(f"Kernel diameter: {args.diameter:.4f} mm")
    lines.append(f"Pooling scale: {getattr(args, 'scale', 2.0):.4f}")
    lines.append(f"Kernel growth: {getattr(args, 'kernel_growth', 2.0):.4f}")
    lines.append(f"Kernel trim threshold: {getattr(args, 'kernel_trim_threshold', 1.0):.6f}")
    lines.append(f"Kernel trim cross-section: {getattr(args, 'kernel_trim_cross_section', 0.0):.6f}")
    lines.append(f"Radial basis functions: {args.num_radial_basis}")
    lines.append("")

    # Training
    lines.append("TRAINING")
    lines.append("-" * 80)
    lines.append(f"Epochs: {args.epochs}")
    lines.append(f"Learning rate: {args.lr}")
    lines.append(f"Batch size (base): {args.batch_size}")
    lines.append(f"Dynamic batch sizing: {args.dynamic_batch_size}")
    if args.dynamic_batch_size:
        lines.append(f"  Target memory: {args.target_memory_mb} MB")
        lines.append(f"  Min effective batch: {args.min_batch_size} (via gradient accumulation)")
        lines.append(f"  Max batch: {args.max_batch_size}")
    patch_mm = tuple(args.patch_size_mm)
    lines.append(f"Patch size (mm): {patch_mm}")
    lines.append(f"Patches per epoch: {args.patches_per_epoch}")
    lines.append(f"Validation patches: {args.val_patches}")
    lines.append(f"Foreground oversample: {args.foreground_oversample}")
    lines.append(f"Pooling factor: {args.pooling_factor}")
    lines.append(f"Workers: {args.num_workers}")
    lines.append("")

    # Augmentation
    if getattr(args, 'subsample_weight', 0) > 0:
        lines.append("AUGMENTATION")
        lines.append("-" * 80)
        lines.append(f"Subsampled data weight: {args.subsample_weight}")
        lines.append("")

    # Resolution groups - sorted by number of cases (descending)
    sorted_groups = sorted(groups, key=lambda g: g['n_cases'], reverse=True)

    lines.append("RESOLUTION GROUPS")
    lines.append("-" * 120)

    # Table header
    min_batch = getattr(args, 'min_batch_size', 1)
    lines.append(f"{'Spacing (mm)':<22} {'Patch (voxels)':<18} {'Patch (mm)':<16} {'Batch':<6} {'Split':<6} {'Accum':<6} {'Eff.BS':<7} {'Cases':<7} {'Type':<12} {'Memory':<10} {'RF Error (mm)':<20}")
    lines.append("-" * 156)

    shrunken = []
    SHRINK_TOL_MM = 4.0

    # Table rows
    for group in sorted_groups:
        raw_spacing = group['spacing']

        # Handle superres groups: ('superres', sub_sp, orig_sp)
        if isinstance(raw_spacing, (list, tuple)) and len(raw_spacing) == 3 and raw_spacing[0] == 'superres':
            sub_sp = tuple(float(s) for s in raw_spacing[1])
            orig_sp = tuple(float(s) for s in raw_spacing[2])
            spacing_str = f"SR {sub_sp[0]:.2f},{sub_sp[1]:.2f},{sub_sp[2]:.2f}"
            spacing = sub_sp
        else:
            spacing = tuple(float(s) for s in raw_spacing)
            spacing_str = f"({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f})"

        patch_voxels = tuple(group['patch_size_voxels'])

        # Format patch size
        patch_str = f"{patch_voxels[0]}x{patch_voxels[1]}x{patch_voxels[2]}"

        # Spatial splits and gradient accumulation
        n_splits = group.get('n_spatial_splits', 1)
        bs = group['batch_size']
        accum = math.ceil(min_batch / bs) if bs < min_batch else 1
        eff_bs = bs * accum

        # RF verification
        rf_info = verify_receptive_field(
            patch_voxels, spacing, patch_mm,
            args.n_downsample, model_scale
        )
        error = rf_info['error']
        rf_err_str = f"({error[0]:.1f}, {error[1]:.1f}, {error[2]:.1f})"

        # Group type
        group_type = group.get('group_type', 'real')

        # Memory (per-split if splitting)
        if 'measured_memory_bs1' in group:
            mem_str = f"{group['measured_memory_bs1']:.0f} MB*"
        else:
            mem_str = f"{group['estimated_memory_mb']:.0f} MB"

        # Effective physical patch and shrink detection (silent patch reduction
        # in mm_to_voxels when the request doesn't divide evenly by the
        # cumulative pool factor).
        effective_mm = tuple(v * s for v, s in zip(patch_voxels, spacing))
        shortfall = tuple(max(0.0, r - e) for r, e in zip(patch_mm, effective_mm))
        is_shrunken = any(sh > SHRINK_TOL_MM for sh in shortfall)
        patch_mm_str = f"{effective_mm[0]:.0f}x{effective_mm[1]:.0f}x{effective_mm[2]:.0f}"
        if is_shrunken:
            patch_mm_str += "!"
            shrunken.append((spacing_str, effective_mm, shortfall))

        lines.append(f"{spacing_str:<22} {patch_str:<18} {patch_mm_str:<16} {bs:<6} {n_splits:<6} {accum:<6} {eff_bs:<7} {group['n_cases']:<7} {group_type:<12} {mem_str:<10} {rf_err_str:<20}")

    # Footnote for measured memory
    has_measured = any('measured_memory_bs1' in g for g in groups)
    if has_measured:
        lines.append("")
        lines.append("* Memory values marked with * are directly measured on GPU (bs=1).")
        lines.append("  Batch sizes are computed from measured per-item memory cost.")

    if shrunken:
        lines.append("")
        lines.append(f"! Patch (mm) columns marked with '!' have an effective patch")
        lines.append(f"  smaller than the requested {patch_mm} mm.  This happens when")
        lines.append(f"  the request can't be divided by the cumulative pool factor;")
        lines.append(f"  mm_to_voxels floors down silently.  Affected groups:")
        lines.append(f"    {'spacing':<24} {'effective (mm)':<22} {'shortfall (mm)'}")
        for sp_str, eff_mm, shortfall in shrunken:
            eff_s = f"({eff_mm[0]:.0f}, {eff_mm[1]:.0f}, {eff_mm[2]:.0f})"
            sh_s = f"({shortfall[0]:.0f}, {shortfall[1]:.0f}, {shortfall[2]:.0f})"
            lines.append(f"    {sp_str:<24} {eff_s:<22} {sh_s}")

    lines.append("")

    # Kernel size table
    scale = getattr(args, 'scale', 2.0)
    trim_th = getattr(args, 'kernel_trim_threshold', 1.0)
    kg = getattr(args, 'kernel_growth', 2.0)
    n_down = args.n_downsample
    diam = args.diameter
    scales_list = [scale * (2 ** i) for i in range(n_down)]

    # Collect unique spacings from groups
    unique_spacings = []
    for group in sorted_groups:
        raw_spacing = group['spacing']
        if isinstance(raw_spacing, (list, tuple)) and len(raw_spacing) == 3 and raw_spacing[0] == 'superres':
            continue  # skip superres groups
        sp = tuple(float(s) for s in raw_spacing)
        if sp not in unique_spacings:
            unique_spacings.append(sp)

    if unique_spacings:
        lines.append("KERNEL SIZE TABLE")
        lines.append("-" * 120)
        # Header
        hdr = f"{'Spacing':<22s}"
        for k in range(n_down + 1):
            label = f"L{k}" + (" (bottleneck)" if k == n_down else "")
            hdr += f" {label:>14s}"
        lines.append(hdr)
        lines.append("-" * 120)

        for sp in unique_spacings:
            trim_cs = getattr(args, 'kernel_trim_cross_section', 0.0)
            sizes = compute_kernel_sizes(diam, sp, scales_list, trim_th, kg, trim_cs)
            sp_str = f"({sp[0]:.2f},{sp[1]:.2f},{sp[2]:.2f})"
            row = f"{sp_str:<22s}"
            for k in range(n_down + 1):
                ks = sizes[k]
                ks_str = f"{ks[0]}x{ks[1]}x{ks[2]}"
                row += f" {ks_str:>14s}"
            lines.append(row)

        lines.append("")

    # Curriculum schedule
    if curriculum_phases:
        lines.append("CURRICULUM SCHEDULE (batch-size tiers)")
        lines.append("-" * 60)
        lines.append(f"{'Phase':<6} {'Epoch':<7} {'Type':<16} {'BS >=':<7} {'Groups':<8} {'Cases':<7}")
        lines.append("-" * 60)
        for i, p in enumerate(curriculum_phases):
            lines.append(
                f"{i:<6} {p['epoch']:<7} {p['type']:<16} {p['bs_threshold']:<7} "
                f"{p['n_groups']:<8} {p['n_cases']:<7}"
            )
        lines.append("")

    # Hardware
    lines.append("HARDWARE")
    lines.append("-" * 80)
    lines.append(f"GPU: {args.gpu}")
    lines.append(f"Mixed precision (BF16): {args.fp16}")
    lines.append(f"Deep supervision: {getattr(args, 'deep_supervision', False)}")
    lines.append("")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))



# =============================================================================
# Pyramid config helper
# =============================================================================

def _build_pyramid_config(args):
    """Build pyramid config dict from args."""
    cfg = {'pyramid_mode': args.pyramid}
    if hasattr(args, 'pyramid_decay') and args.pyramid_decay is not None:
        cfg['pyramid_decay'] = args.pyramid_decay
    return cfg


# =============================================================================
# Batch-size curriculum
# =============================================================================

def _build_bs_curriculum(loader_groups, tiers, phase_len, total_epochs=None):
    """Build loader-onboarding phases from batch-size tiers.

    Each tier lowers the batch-size threshold, activating more spacing
    groups (finer resolutions have smaller batches).

    If *total_epochs* is given, *phase_len* is ignored and phases are
    spaced so the last tier starts at the midpoint of training, leaving
    the second half for all-groups training.

    Parameters
    ----------
    loader_groups : list of dict
        Each dict has keys: spacing, batch_size, case_ids (or n_cases).
    tiers : list of int
        Descending batch-size thresholds, e.g. [12, 8, 4, 2, 1].
    phase_len : int
        Epochs per phase (used only when total_epochs is None).
    total_epochs : int, optional
        Total training epochs.  When provided, phases are evenly
        distributed over the first half of training.

    Returns
    -------
    list of dict
        Phase dicts with keys: epoch, type, bs_threshold, n_groups, n_cases.
    """
    tiers = sorted(tiers, reverse=True)
    n_tiers = len(tiers)

    if total_epochs is not None and n_tiers > 0:
        # Equal-length phases: total_epochs / n_tiers each
        step = total_epochs // n_tiers
    else:
        step = phase_len

    phases = []
    for ti, threshold in enumerate(tiers):
        active = [g for g in loader_groups if g['batch_size'] >= threshold]
        n_groups = len(active)
        n_cases = sum(
            len(g['case_ids']) if 'case_ids' in g else g.get('n_cases', 0)
            for g in active
        )
        phases.append({
            'epoch': ti * step,
            'type': 'initial' if ti == 0 else 'loader_add',
            'bs_threshold': threshold,
            'n_groups': n_groups, 'n_cases': n_cases,
        })

    return phases


# =============================================================================
# Resolution tracking and visualization
# =============================================================================

def _extract_resolutions(resolutions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract in-plane and slice thickness from resolution entries."""
    in_plane_res = []
    slice_thickness = []

    for entry in resolutions:
        spacing = entry['spacing']
        batch_size = entry['batch_size']
        # Sort spacing to get (in_plane, in_plane, slice) order
        sorted_spacing = tuple(sorted(spacing))
        ip = sorted_spacing[0]  # Smallest = in-plane
        st = sorted_spacing[2]  # Largest = slice thickness

        # Add one point per sample in batch
        for _ in range(batch_size):
            in_plane_res.append(ip)
            slice_thickness.append(st)

    return np.array(in_plane_res), np.array(slice_thickness)


def plot_resolution_density(
    train_resolutions: List[Dict],
    val_resolutions: List[Dict],
    output_dir: Path,
    epoch: int
):
    """Plot 2D density of training and validation samples by resolution.

    Parameters
    ----------
    train_resolutions : list of dict
        Training resolution entries, each with 'spacing' and 'batch_size'
    val_resolutions : list of dict
        Validation entries from last epoch, each with 'spacing' and 'dice'
    output_dir : Path
        Directory to save plots
    epoch : int
        Current epoch number
    """
    if not train_resolutions and not val_resolutions:
        return

    # Extract training resolutions
    train_ip, train_st = _extract_resolutions(train_resolutions) if train_resolutions else (np.array([]), np.array([]))

    # Extract validation resolutions and dice
    val_ip, val_st, val_dice = np.array([]), np.array([]), np.array([])
    if val_resolutions:
        ip_list, st_list, dice_list = [], [], []
        for entry in val_resolutions:
            spacing = entry['spacing']
            sorted_spacing = tuple(sorted(spacing))
            ip_list.append(sorted_spacing[0])
            st_list.append(sorted_spacing[2])
            dice_list.append(entry['dice'])
        val_ip = np.array(ip_list)
        val_st = np.array(st_list)
        val_dice = np.array(dice_list)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_train, ax_val = axes[0]
    ax_ip, ax_st = axes[1]

    # Common plot settings
    xlim = (0, 2.5)
    ylim = (0.4, 15)

    # Top Left: Training 2D density (cumulative)
    if len(train_ip) > 0:
        hb_train = ax_train.hexbin(train_ip, train_st, gridsize=30, cmap='Blues',
                                    mincnt=1, yscale='log')
        cb_train = fig.colorbar(hb_train, ax=ax_train)
        cb_train.set_label('Count')
    ax_train.set_xlabel('In-plane Resolution (mm)', fontsize=11)
    ax_train.set_ylabel('Slice Thickness (mm)', fontsize=11)
    ax_train.set_title(f'Training Samples (Cumulative)\nEpochs 0-{epoch} ({len(train_ip)} samples)', fontsize=12)
    ax_train.set_xlim(xlim)
    ax_train.set_ylim(ylim)

    # Top Right: Validation hexbin colored by mean pseudo-dice (last epoch only)
    if len(val_ip) > 0:
        hb_val = ax_val.hexbin(val_ip, val_st, C=val_dice, gridsize=20,
                                reduce_C_function=np.mean, cmap='RdYlGn',
                                mincnt=1, yscale='log', vmin=0, vmax=1)
        cb_val = fig.colorbar(hb_val, ax=ax_val)
        cb_val.set_label('Mean Dice')
    ax_val.set_xlabel('In-plane Resolution (mm)', fontsize=11)
    ax_val.set_ylabel('Slice Thickness (mm)', fontsize=11)
    ax_val.set_title(f'Validation Pseudo-Dice (Epoch {epoch})\n{len(val_ip)} patches', fontsize=12)
    ax_val.set_xlim(xlim)
    ax_val.set_ylim(ylim)

    # Bottom Left: In-plane marginal comparison (proportions)
    bins_ip = np.linspace(0, 2.5, 30)
    if len(train_ip) > 0:
        weights_train = np.ones_like(train_ip) / len(train_ip)
        ax_ip.hist(train_ip, bins=bins_ip, weights=weights_train, alpha=0.6, color='#3498db', label=f'Train ({len(train_ip)})')
    if len(val_ip) > 0:
        weights_val = np.ones_like(val_ip) / len(val_ip)
        ax_ip.hist(val_ip, bins=bins_ip, weights=weights_val, alpha=0.6, color='#e67e22', label=f'Val ({len(val_ip)})')
    ax_ip.set_xlabel('In-plane Resolution (mm)', fontsize=11)
    ax_ip.set_ylabel('Proportion', fontsize=11)
    ax_ip.set_title('In-plane Resolution Distribution', fontsize=12)
    ax_ip.legend(loc='upper right')

    # Bottom Right: Slice thickness marginal comparison (proportions)
    bins_st = np.logspace(np.log10(0.5), np.log10(12), 30)
    if len(train_st) > 0:
        weights_train = np.ones_like(train_st) / len(train_st)
        ax_st.hist(train_st, bins=bins_st, weights=weights_train, alpha=0.6, color='#3498db', label=f'Train ({len(train_st)})')
    if len(val_st) > 0:
        weights_val = np.ones_like(val_st) / len(val_st)
        ax_st.hist(val_st, bins=bins_st, weights=weights_val, alpha=0.6, color='#e67e22', label=f'Val ({len(val_st)})')
    ax_st.set_xlabel('Slice Thickness (mm)', fontsize=11)
    ax_st.set_ylabel('Proportion', fontsize=11)
    ax_st.set_xscale('log')
    ax_st.set_title('Slice Thickness Distribution', fontsize=12)
    ax_st.legend(loc='upper right')

    plt.suptitle(f'Resolution Distribution (Epoch {epoch})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot (overwritten each epoch, same directory as progress.png)
    plt.savefig(output_dir / 'resolution.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Progress plot (nnUNet-style)
# =============================================================================

def plot_progress(train_losses, val_losses, pseudo_dice, ema_pseudo_dice,
                  lrs, epoch_times, output_dir):
    """Generate nnUNet-style 3-panel progress plot.

    Panel 1: Loss (left y) and Dice (right y) vs epoch
    Panel 2: Epoch duration vs epoch
    Panel 3: Learning rate vs epoch
    """
    if not train_losses:
        return

    epochs = list(range(len(train_losses)))

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # Panel 1: Loss + Dice (dual y-axis)
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', label='train loss', linewidth=1.5)
    ax1.plot(epochs, val_losses, 'r-', label='val loss', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    ax1_right = ax1.twinx()
    ax1_right.plot(epochs, pseudo_dice, 'g:', label='pseudo dice', linewidth=1.0, alpha=0.7)
    ax1_right.plot(epochs, ema_pseudo_dice, 'g-', label='EMA pseudo dice', linewidth=1.5)
    ax1_right.set_ylabel('Pseudo Dice')
    ax1_right.legend(loc='upper right')
    ax1.set_title('Training Progress')

    # Panel 2: Epoch duration
    ax2 = axes[1]
    ax2.plot(epochs, epoch_times, 'b-', linewidth=1.0)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Time [s]')
    ax2.set_title('Epoch Duration')

    # Panel 3: Learning rate
    ax3 = axes[2]
    ax3.plot(epochs, lrs, 'b-', linewidth=1.0)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'progress.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Plan validation
# =============================================================================

def _write_plan_validation_log(args, output_dir: Path, config_hash: str = None):
    """Compare runtime args against planned config.json and log results.

    Writes plan_validation.log confirming match or listing divergences.
    Raises RuntimeError if model/training parameters diverge.
    """
    config_path = output_dir / 'config.json'
    if not config_path.exists():
        # No planned config — legacy mode, just log runtime args
        log_lines = [
            f"Plan validation: {datetime.now().isoformat()}",
            "No config.json found (legacy mode) — skipping validation.",
        ]
        (output_dir / 'plan_validation.log').write_text('\n'.join(log_lines))
        return

    with open(config_path) as f:
        config = json.load(f)

    # Parameters to validate (config section, key, args attribute name)
    checks = [
        # Model params (hard fail on mismatch)
        ('model', 'n_base_filters', 'n_base_filters', True),
        ('model', 'n_downsample', 'n_downsample', True),
        ('model', 'diameter', 'diameter', True),
        ('model', 'num_radial_basis', 'num_radial_basis', True),
        ('model', 'equivariance', 'equivariance', True),
        ('model', 'normalization', 'normalization', True),
        ('model', 'activation', 'activation', True),
        ('model', 'dropout', 'dropout', True),
        ('model', 'max_features', 'max_features', True),
        ('model', 'kernel_trim_threshold', 'kernel_trim_threshold', True),
        ('model', 'kernel_trim_cross_section', 'kernel_trim_cross_section', True),
        ('model', 'kernel_growth', 'kernel_growth', True),
        ('model', 'sc_mode', 'sc_mode', True),
        ('model', 'scale', 'scale', True),
        ('model', 'fused_gate', 'fused_gate', True),
        ('model', 'backend', 'backend', True),
        # Training params (hard fail on mismatch)
        ('training', 'epochs', 'epochs', True),
        ('training', 'learning_rate', 'lr', True),
        ('training', 'weight_decay', 'weight_decay', True),
        ('training', 'grad_clip', 'grad_clip', True),
        ('training', 'patch_size_mm', 'patch_size_mm', True),
        ('training', 'patches_per_epoch', 'patches_per_epoch', True),
        ('training', 'dynamic_batch_size', 'dynamic_batch_size', True),
        ('training', 'target_memory_mb', 'target_memory_mb', True),
        # Hardware (hard fail)
        ('hardware', 'fp16', 'fp16', True),
        ('hardware', 'deep_supervision', 'deep_supervision', True),
        # Data (warn only)
        ('data', 'preprocessed_dir', 'preprocessed_dir', False),
        ('data', 'fold', 'fold', False),
    ]

    log_lines = [
        f"Plan validation: {datetime.now().isoformat()}",
        f"Config: {config_path}",
    ]
    if config_hash:
        log_lines.append(f"Config hash (SHA256): {config_hash}")
    log_lines.append("")

    divergences = []
    warnings = []

    for section, key, attr, is_hard in checks:
        if section not in config or key not in config[section]:
            continue
        planned = config[section][key]
        actual = getattr(args, attr, None)
        # Normalize for comparison (lists vs tuples)
        if isinstance(planned, list):
            planned_cmp = tuple(planned)
        else:
            planned_cmp = planned
        if isinstance(actual, list):
            actual_cmp = tuple(actual)
        else:
            actual_cmp = actual

        if planned_cmp != actual_cmp:
            entry = f"  {section}.{key}: planned={planned} vs runtime={actual}"
            if is_hard:
                divergences.append(entry)
            else:
                warnings.append(entry)

    if not divergences and not warnings:
        log_lines.append("All parameters match planned config. OK")
    else:
        if warnings:
            log_lines.append("WARNINGS (non-critical):")
            log_lines.extend(warnings)
            log_lines.append("")
        if divergences:
            log_lines.append("ERRORS (critical divergences):")
            log_lines.extend(divergences)

    (output_dir / 'plan_validation.log').write_text('\n'.join(log_lines))

    if divergences:
        print("ERROR: Runtime parameters diverge from planned config.json:")
        for d in divergences:
            print(d)
        raise RuntimeError(
            f"Plan validation failed: {len(divergences)} parameter(s) diverge from config.json. "
            f"See {output_dir / 'plan_validation.log'}"
        )
    elif warnings:
        print(f"Plan validation: OK ({len(warnings)} non-critical warning(s), see plan_validation.log)")
    else:
        print("Plan validation: all parameters match config.json.")


# =============================================================================
# Training loop
# =============================================================================

def train(args, config_hash: str = None, planned_batch_sizes: dict = None):
    # Enable TF32 for matmuls — ~35% speedup on Ampere+ GPUs with negligible precision loss
    torch.set_float32_matmul_precision('high')

    # --- DDP setup (auto-detected from torchrun env vars) ---
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1
    is_main = rank == 0  # only rank 0 logs, saves checkpoints, runs validation

    if distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if is_main:
        if distributed:
            mode = "sync" if getattr(args, 'ddp_sync_groups', False) else "async"
            print(f"Using device: {device} (DDP: {world_size} GPUs, {mode} groups)")
        else:
            print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B (optional, rank 0 only)
    wandb_run = None
    if is_main and getattr(args, 'wandb', False):
        try:
            import wandb
            run_name = getattr(args, 'wandb_name', None) or output_dir.name
            wandb_run = wandb.init(
                project=getattr(args, 'wandb_project', 'irrepunet'),
                name=run_name,
                config=vars(args),
                dir=str(output_dir),
                resume='allow' if getattr(args, 'resume', False) else None,
            )
            print(f"W&B logging enabled: {wandb_run.url}")
        except ImportError:
            print("WARNING: --wandb specified but wandb is not installed. pip install wandb")
        except Exception as e:
            print(f"WARNING: W&B init failed: {e}")

    # Load splits
    preprocessed_dir = Path(args.preprocessed_dir)
    with open(preprocessed_dir / 'splits_final.json') as f:
        splits = json.load(f)

    train_cases = splits[args.fold]['train']
    val_cases = splits[args.fold]['val']

    if is_main:
        print(f"Fold {args.fold}: {len(train_cases)} train, {len(val_cases)} val")

    # Get patch size in mm
    patch_size_mm = tuple(args.patch_size_mm)

    # Get reference spacing to compute representative voxel patch size for transforms
    with open(preprocessed_dir / f"{train_cases[0]}.pkl", 'rb') as f:
        ref_props = pickle.load(f)
    ref_spacing = ref_props['spacing']
    # Convert mm to voxels for transform configuration (approximate)
    ref_patch_voxels = tuple(
        max(args.pooling_factor, int(round(mm / sp)) // args.pooling_factor * args.pooling_factor)
        for mm, sp in zip(patch_size_mm, ref_spacing)
    )

    # Create transforms (using reference voxel size)
    train_transforms = get_training_transforms(
        ref_patch_voxels,
        disable_mirroring=args.disable_mirroring,
        disable_spatial=args.disable_spatial,
        bias_field=getattr(args, 'bias_field', True),
    )
    val_transforms = get_validation_transforms()

    # Create data loaders (with optional virtual low-resolution groups)
    train_loader = MultiResolutionLoader(
        preprocessed_dir=str(preprocessed_dir),
        case_identifiers=train_cases,
        batch_size=args.batch_size,
        patch_size_mm=patch_size_mm,
        oversample_foreground_percent=args.foreground_oversample,
        num_workers=args.num_workers,
        transforms=train_transforms,
        pooling_factor=args.pooling_factor,
        dynamic_batch_size=args.dynamic_batch_size,
        target_memory_mb=args.target_memory_mb,
        n_base_filters=args.n_base_filters,
        n_downsample=args.n_downsample,
        min_batch_size=1,  # Allow batch_size=1; gradient accumulation handles the rest
        max_batch_size=args.max_batch_size,
        fp16=args.fp16,
        subsample_weight=args.subsample_weight,
        model_scale=getattr(args, 'scale', 2.0),  # Pooling scale from e3nn model (physical units)
        min_spacing=getattr(args, 'min_spacing', 0.0),
        max_inplane_spacing=getattr(args, 'max_inplane_spacing', 0.0),
        min_slice_thickness=getattr(args, 'min_slice_thickness', 0.0),
        max_slice_thickness=getattr(args, 'max_slice_thickness', 0.0),
        min_loader_cases=getattr(args, 'min_loader_cases', 2),
        superres_training=getattr(args, 'superres_training', False),
        superres_weight=getattr(args, 'superres_weight', 0.1),
        group_balance=getattr(args, 'group_balance', 0.0),
        planned_batch_sizes=planned_batch_sizes,
        rank=rank,
        world_size=world_size,
        sync_groups=distributed and getattr(args, 'ddp_sync_groups', False),
    )

    # Save runtime loader configuration (rank 0 only)
    if is_main:
        print("Saving runtime loader config...", flush=True)
        runtime_loader_path = output_dir / 'loader_config_runtime.txt'
        train_loader.save_config(
            str(runtime_loader_path),
            args=args,
            n_train_cases=len(train_cases),
            n_val_cases=len(val_cases),
        )
        # Compare with planned loader_config.txt if it exists
        planned_loader_path = output_dir / 'loader_config.txt'
        if planned_loader_path.exists():
            planned_lines = planned_loader_path.read_text().splitlines()
            runtime_lines = runtime_loader_path.read_text().splitlines()
            if planned_lines != runtime_lines:
                import difflib
                diff = list(difflib.unified_diff(
                    planned_lines, runtime_lines,
                    fromfile='loader_config.txt (planned)',
                    tofile='loader_config_runtime.txt (runtime)',
                    lineterm=''
                ))
                if diff:
                    print(f"  WARNING: Runtime loader config differs from planned version.")
                    print(f"  See {output_dir / 'loader_config_diff.txt'} for details.")
                    (output_dir / 'loader_config_diff.txt').write_text('\n'.join(diff))
            else:
                print("  Runtime loader config matches planned version.")
        print("Runtime loader config saved.", flush=True)

    # Spatial splitting disabled: estimate_memory_mb overestimates for n_downsample>=6
    # (~4x), causing unnecessary splits that slow training dramatically. With batch=1
    # and conservative target_memory_mb, the largest patches (192x192x96) fit in ~22GB.
    spatial_splits = {}

    if is_main:
        print("Setting up validation loader...", flush=True)
    val_loader = MultiResolutionLoader(
        preprocessed_dir=str(preprocessed_dir),
        case_identifiers=val_cases,
        batch_size=args.batch_size,  # Fixed batch size for validation
        patch_size_mm=patch_size_mm,
        oversample_foreground_percent=0.33,
        num_workers=0,  # SingleThreadedAugmenter for val (no forked workers)
        transforms=val_transforms,
        model_scale=getattr(args, 'scale', 2.0),  # Pooling scale from e3nn model (physical units)
        n_downsample=args.n_downsample,
        # No resolution filtering for validation — evaluate on all resolutions
        min_loader_cases=getattr(args, 'min_loader_cases', 2),
        planned_batch_sizes=planned_batch_sizes,
    )
    if is_main:
        print("Validation loader ready.", flush=True)

    # Determine number of classes
    sample_seg = np.load(preprocessed_dir / f"{train_cases[0]}_seg.npy")
    n_classes = int(sample_seg.max()) + 1
    if is_main:
        print(f"Number of classes: {n_classes}", flush=True)

    # Create model (ref_spacing already loaded above for transforms)
    if is_main:
        print(f"Building E3nnUNet model...", flush=True)
    model = E3nnUNet(
        n_classes=n_classes,
        in_channels=1,
        diameter=args.diameter,
        num_radial_basis=args.num_radial_basis,
        spacing=ref_spacing,
        normalization=args.normalization,
        n_base_filters=args.n_base_filters,
        n_downsample=args.n_downsample,
        equivariance=args.equivariance,
        lmax=len(args.irrep_ratios) - 1,
        dropout_prob=args.dropout,
        cutoff=True,
        deep_supervision=args.deep_supervision,
        max_features=args.max_features,
        irrep_ratios=tuple(args.irrep_ratios),
        fill_to_max=args.fill_to_max,
        activation=getattr(args, 'activation', 'softplus'),
        kernel_trim_threshold=getattr(args, 'kernel_trim_threshold', 1.0),
        kernel_trim_cross_section=getattr(args, 'kernel_trim_cross_section', 0.0),
        kernel_growth=getattr(args, 'kernel_growth', 2.0),
        sequential_sc=getattr(args, 'sequential_sc', False),
        sc_mode=getattr(args, 'sc_mode', None),
        scale=getattr(args, 'scale', 2.0),
        fused_gate=getattr(args, 'fused_gate', True),
        pool_mode=getattr(args, 'pool_mode', 'maxpool3d'),
        backend=getattr(args, 'backend', 'e3nn'),
        pyramid=_build_pyramid_config(args) if getattr(args, 'pyramid', None) else False,
    )
    if is_main:
        print(f"Model built, moving to {device}...", flush=True)
    model = model.to(device)
    if is_main:
        print(f"Model on device.", flush=True)

    # Keep a reference to the unwrapped model for attribute access
    raw_model = model

    # Wrap with DDP
    if distributed:
        model = DDP(raw_model, device_ids=[local_rank])
        if is_main:
            print(f"Model wrapped with DistributedDataParallel", flush=True)

    # Load pretrained weights if requested (before optimizer, fresh training)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_path}")
        if is_main:
            print(f"Loading pretrained weights from: {init_path}", flush=True)
        init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
        init_sd = init_ckpt.get('model_state_dict', {})
        load_spacing_independent_state_dict(raw_model, init_sd)
        if is_main:
            print(f"Loaded pretrained weights ({len(init_sd)} keys)", flush=True)

    # Count parameters
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        print(f"Model parameters: {n_params:,}")

    # Loss function
    base_criterion = DiceCELoss(
        ce_weight=1.0,
        dice_weight=1.0,
        include_background=not getattr(args, 'no_background_dice', False),
        batch_dice=getattr(args, 'batch_dice', False),
    )
    if args.deep_supervision:
        criterion = DeepSupervisionLoss(
            base_loss=base_criterion,
            n_scales=args.n_downsample
        )
        if is_main:
            print(f"Using deep supervision with {args.n_downsample} scales")
    else:
        criterion = base_criterion

    # Optimizer (use raw_model params — DDP wraps the same underlying parameters)
    optimizer = AdamW(raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # Plan validation: compare runtime args against planned config.json
    if is_main:
        _write_plan_validation_log(args, output_dir, config_hash)

    # Training state
    start_epoch = 0
    best_val_dice = 0

    # BF16 mixed precision (same dynamic range as FP32, no GradScaler needed)
    use_amp = args.fp16
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    if use_amp and is_main:
        print("Using BF16 mixed precision training")

    # Training history for plotting (nnUNet-style)
    train_losses_history = []
    val_losses_history = []
    pseudo_dice_history = []
    ema_pseudo_dice_history = []
    lr_history = []
    epoch_times_history = []

    # Safety: refuse to overwrite existing checkpoint without --resume
    if not args.resume and (output_dir / 'checkpoint.pt').exists():
        raise RuntimeError(
            f"checkpoint.pt already exists in {output_dir}. "
            f"Use --resume to continue training, or remove the checkpoint to start fresh."
        )

    # Resume if requested (all ranks load the same checkpoint)
    if args.resume and (output_dir / 'checkpoint.pt').exists():
        checkpoint = torch.load(output_dir / 'checkpoint.pt', map_location=device, weights_only=False)
        # Verify config hash matches (catches accidental config edits)
        saved_hash = checkpoint.get('config_hash')
        if saved_hash and config_hash and saved_hash != config_hash:
            raise RuntimeError(
                f"Config hash mismatch on resume! "
                f"Checkpoint was trained with config hash {saved_hash}, "
                f"but current config.json has hash {config_hash}. "
                f"This suggests config.json was modified after training started."
            )
        load_spacing_independent_state_dict(raw_model, checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint.get('best_val_dice', 0)
        # Restore logging history
        if 'logging' in checkpoint:
            log_data = checkpoint['logging']
            train_losses_history = log_data['train_losses']
            val_losses_history = log_data['val_losses']
            pseudo_dice_history = log_data['pseudo_dice']
            ema_pseudo_dice_history = log_data['ema_pseudo_dice']
            lr_history = log_data['lrs']
            epoch_times_history = log_data['epoch_times']
        if is_main:
            print(f"Resumed from epoch {start_epoch}")

    # Training loop
    if is_main:
        print(f"\nStarting training...", flush=True)
        if args.resolution_jitter_sigma > 0:
            print(f"Resolution jitter enabled: sigma={args.resolution_jitter_sigma}")
        if args.scale_jitter_std > 0:
            print(f"Scale jitter enabled: std={args.scale_jitter_std} (smooths architecture boundaries)")
    log_file = open(output_dir / 'train.log', 'a') if is_main else None

    # Structured JSON log: one dict per epoch, append-friendly
    json_log_path = output_dir / 'training_log.json'
    if json_log_path.exists():
        with open(json_log_path) as f:
            json_log = json.load(f)
    else:
        json_log = []

    # Cumulative resolution tracking across all epochs
    all_train_resolutions = []
    last_val_resolutions = []  # Per-patch dice + spacing from last epoch

    # Rolling 10-epoch window of per-group TP/FP/FN for stable dice reporting
    from collections import deque, defaultdict
    _group_stats_window = deque(maxlen=10)  # each entry: dict[group_key -> (tp, fp, fn)]

    # If using init_checkpoint, run validation-only at epoch 0 to establish baseline
    validate_only_first = args.init_checkpoint and start_epoch == 0

    _logged_group_spacing = False

    # Curriculum: staged introduction of resolution groups.
    # Two modes: legacy (--curriculum) and batch-size tiers (--curriculum_bs_tiers).
    _curriculum_phases = []
    if getattr(args, 'curriculum_bs_tiers', None):
        # Build loader_groups from train_loader's runtime data
        _runtime_groups = [
            {'spacing': list(sp), 'batch_size': train_loader.group_batch_sizes[sp],
             'n_cases': len(cases)}
            for sp, cases in train_loader.spacing_groups.items()
            if sp in train_loader.group_batch_sizes
        ]
        _curriculum_phases = _build_bs_curriculum(
            _runtime_groups, args.curriculum_bs_tiers,
            getattr(args, 'curriculum_phase_len', 30),
            total_epochs=args.epochs)
        _bs_lookup = {sp: train_loader.group_batch_sizes[sp]
                      for sp in train_loader.group_batch_sizes}

    # Legacy resolution-based curriculum
    _curriculum_boundaries = getattr(args, 'curriculum', None) or []
    _curriculum_stages = [
        (1.0, 3.0, "aniso: in-plane>=1mm, slice>=3mm"),
        (1.0, 0.0, "1mm iso: in-plane>=1mm"),
        (0.75, 0.0, "0.75mm: in-plane>=0.75mm"),
        (0.5, 0.0, "0.5mm: in-plane>=0.5mm"),
        (0.0, 0.0, "all groups"),
    ]
    _curriculum_stage = -1  # force update on first epoch
    _active_phase = None

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Update curriculum: batch-size tiers (new) or legacy resolution boundaries
        if _curriculum_phases:
            phase = _curriculum_phases[0]
            for p in _curriculum_phases:
                if epoch >= p['epoch']:
                    phase = p
            if phase is not _active_phase:
                _active_phase = phase
                bs_min = phase['bs_threshold']
                train_loader.set_active_groups(
                    lambda sp, _m=bs_min: _bs_lookup.get(tuple(sp), 0) >= _m
                )
                n_active = len(train_loader.group_weights)
                if is_main:
                    print(f"  [curriculum] {phase['type']}: bs>={bs_min}, "
                          f"{n_active} groups active", flush=True)
        elif _curriculum_boundaries:
            stage = 0
            for boundary in _curriculum_boundaries:
                if epoch >= boundary:
                    stage += 1
            stage = min(stage, len(_curriculum_stages) - 1)
            if stage != _curriculum_stage:
                _curriculum_stage = stage
                min_ip, min_sl, desc = _curriculum_stages[stage]
                if min_ip > 0 or min_sl > 0:
                    train_loader.set_active_groups(
                        lambda sp, ip=min_ip, sl=min_sl: (
                            min(sp[0], sp[1]) >= ip and max(sp) >= sl
                        )
                    )
                else:
                    train_loader.set_active_groups(None)
                n_active = len(train_loader.group_weights)
                if is_main:
                    print(f"  [curriculum] Stage {stage}: {desc} "
                          f"({n_active} groups active)",
                          flush=True)

        # Skip training on epoch 0 when using pretrained weights (baseline validation)
        if validate_only_first and epoch == 0:
            train_loss = 0.0
            epoch_peak_mb = 0.0
            if is_main:
                print(f"Epoch {epoch:4d} | Skipping training (baseline validation of pretrained weights)")
        else:
            model.train()

            train_loss = 0
            n_train_batches = 0
            n_train_patches = 0
            epoch_peak_mb = 0.0
            epoch_peak_spacing = None
            epoch_peak_patch = None
            epoch_peak_bs = 0
            nan_skip_count = 0
            torch.cuda.reset_peak_memory_stats(device)

            # Loop termination.  In DDP async mode each rank draws different
            # groups with different batch sizes, so we fix the step count to
            # keep backward() calls in sync across ranks.  In sync mode (or
            # single-GPU) we count patches as before.
            _ddp_async = distributed and not getattr(args, 'ddp_sync_groups', False)
            if _ddp_async:
                _steps_per_epoch = max(1, round(args.patches_per_epoch / train_loader.mean_batch_size))
                _stop = lambda: n_train_batches < _steps_per_epoch
            else:
                _stop = lambda: n_train_patches < args.patches_per_epoch
            while _stop():
                batch, spacing = next(train_loader)

                # Override with group spacing if requested (replicates JAX JIT behavior)
                if getattr(args, 'use_group_spacing', False):
                    native_sp = spacing
                    spacing = batch.get('group_spacing', spacing)
                    if not _logged_group_spacing and is_main:
                        print(f"  [use_group_spacing] native={native_sp} -> group={spacing}")
                        _logged_group_spacing = True

                # Use FP16 for input if enabled
                input_dtype = amp_dtype
                images = batch['data'].to(device, dtype=input_dtype)
                labels = batch['seg'].to(device).long().squeeze(1)  # (B, D, H, W)
                batch_size = images.shape[0]
                is_superres = batch.get('is_superres', False)

                # Gradient accumulation for small-batch groups.
                # Disabled in DDP async mode: ranks process different groups
                # with different batch sizes, so all must call backward()
                # exactly once per step to avoid deadlocking the all-reduce.
                if _ddp_async:
                    accum_steps = 1
                elif not is_superres and batch_size < args.min_batch_size and spacing in train_loader.group_loaders:
                    accum_steps = math.ceil(args.min_batch_size / batch_size)
                else:
                    accum_steps = 1

                # Spatial splits for groups that exceed GPU memory
                # Use canonical group spacing for lookup (actual per-case spacing may differ)
                group_sp = batch.get('group_spacing', spacing)
                n_splits = spatial_splits.get(group_sp, 1) if not is_superres else 1
                total_micro = accum_steps * n_splits

                optimizer.zero_grad()
                accum_loss = 0.0

                for accum_idx in range(accum_steps):
                    if accum_idx > 0:
                        batch = next(train_loader.group_loaders[spacing])
                        images = batch['data'].to(device, dtype=input_dtype)
                        labels = batch['seg'].to(device).long().squeeze(1)

                    # Apply resolution jitter if enabled (shared across splits)
                    if args.resolution_jitter_sigma > 0:
                        jitter = np.random.normal(0, args.resolution_jitter_sigma, 3)
                        jittered_spacing = tuple(s + j for s, j in zip(spacing, jitter))
                        jittered_spacing = tuple(max(0.1, s) for s in jittered_spacing)
                    else:
                        jittered_spacing = spacing

                    # Track resolution for this batch (cumulative across epochs)
                    all_train_resolutions.append({
                        'spacing': jittered_spacing,
                        'batch_size': batch_size
                    })

                    # Apply scale jitter if enabled (shared across splits)
                    if args.scale_jitter_std > 0:
                        base_scales = [model.scale * (2 ** i) for i in range(model.n_downsample)]
                        jittered_scales = [
                            s * (1 + np.random.normal(0, args.scale_jitter_std))
                            for s in base_scales
                        ]
                    else:
                        jittered_scales = None

                    # Split images/labels spatially if needed
                    if n_splits > 1:
                        # Distribute splits across dimensions, ensuring no dim
                        # goes below min_dim (needed for n_downsample pooling levels)
                        min_dim = 2 ** args.n_downsample  # e.g. 64 for n_downsample=6
                        spatial = list(images.shape[2:])  # [D, H, W]
                        splits_per_dim = [1, 1, 1]
                        remaining = n_splits
                        while remaining > 1:
                            # Find largest dimension that can still be split
                            best_dim = -1
                            best_size = 0
                            for d in range(3):
                                chunk_size = spatial[d] // splits_per_dim[d]
                                if chunk_size // 2 >= min_dim and chunk_size > best_size:
                                    best_dim = d
                                    best_size = chunk_size
                            if best_dim < 0:
                                break  # can't split further without going below min
                            splits_per_dim[best_dim] *= 2
                            remaining //= 2

                        total_chunks = splits_per_dim[0] * splits_per_dim[1] * splits_per_dim[2]
                        if total_chunks > 1:
                            # Multi-dim chunking: split sequentially per dim
                            img_chunks_list = [images]
                            lbl_chunks_list = [labels]
                            for d in range(3):
                                if splits_per_dim[d] > 1:
                                    new_img = []
                                    new_lbl = []
                                    for ic in img_chunks_list:
                                        new_img.extend(torch.chunk(ic, splits_per_dim[d], dim=d+2))
                                    for lc in lbl_chunks_list:
                                        new_lbl.extend(torch.chunk(lc, splits_per_dim[d], dim=d+1))
                                    img_chunks_list = new_img
                                    lbl_chunks_list = new_lbl
                            image_chunks = img_chunks_list
                            label_chunks = lbl_chunks_list
                            # Update n_splits/total_micro to reflect actual chunk count
                            n_splits = len(image_chunks)
                            total_micro = accum_steps * n_splits
                        else:
                            image_chunks = [images]
                            label_chunks = [labels]
                            n_splits = 1
                            total_micro = accum_steps
                    else:
                        image_chunks = [images]
                        label_chunks = [labels]

                    for img_chunk, lbl_chunk in zip(image_chunks, label_chunks):
                        # Forward pass
                        with autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                            if is_superres:
                                orig_spacing = tuple(batch['orig_spacings'][0])
                                outputs = raw_model.forward_superres(
                                    img_chunk, sub_spacing=jittered_spacing, orig_spacing=orig_spacing
                                )
                                if outputs.shape[2:] != lbl_chunk.shape[1:]:
                                    outputs = nn.functional.interpolate(
                                        outputs, size=lbl_chunk.shape[1:],
                                        mode='trilinear', align_corners=True
                                    )
                                loss = base_criterion(outputs, lbl_chunk)
                            elif args.deep_supervision:
                                outputs = model(img_chunk, spacing=jittered_spacing, scales=jittered_scales)
                                # Filter out outputs with 0-size spatial dims
                                # (can happen with spatial splitting + deep networks)
                                valid_outputs = [o for o in outputs if all(s > 0 for s in o.shape[2:])]
                                output_shapes = [out.shape[2:] for out in valid_outputs]
                                multi_scale_labels = downsample_seg_for_deep_supervision(
                                    lbl_chunk, output_shapes=output_shapes
                                )
                                loss = criterion(valid_outputs, multi_scale_labels)
                            else:
                                outputs = model(img_chunk, spacing=jittered_spacing, scales=jittered_scales)
                                loss = criterion(outputs, lbl_chunk)

                        # Backward pass with loss scaled by total micro-batches
                        (loss / total_micro).backward()
                        accum_loss += loss.item()

                    n_train_patches += batch_size

                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip)
                else:
                    grad_norm = torch.tensor(0.0, device=device)

                # NaN guard: check loss + grad norm for non-finite values and
                # synchronise the decision across DDP ranks (any rank NaN ⇒
                # all ranks skip).  Log the circumstances before skipping so
                # the bad batch can be investigated.
                loss_val = accum_loss / max(1, total_micro)
                bad_local = (not math.isfinite(loss_val)) or (not torch.isfinite(grad_norm).item())
                if distributed:
                    bad_flag = torch.tensor(1 if bad_local else 0,
                                            device=device, dtype=torch.int)
                    dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
                    bad = bool(bad_flag.item())
                else:
                    bad = bad_local

                if bad:
                    nan_skip_count += 1
                    # Each rank logs its own circumstances (who saw what)
                    print(
                        f"  [NaN GUARD] epoch={epoch} step={n_train_batches} "
                        f"rank={rank} loss={loss_val} grad_norm={grad_norm.item():.3e} "
                        f"spacing={tuple(f'{s:.3f}' for s in spacing)} "
                        f"group_spacing={tuple(f'{s:.3f}' for s in batch.get('group_spacing', spacing))} "
                        f"patch={tuple(images.shape[2:])} bs={batch_size} "
                        f"is_superres={is_superres} "
                        f"bad_local={bad_local}",
                        flush=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()

                # Lightweight memory monitoring (reads counter, no GPU sync)
                step_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if step_peak_mb > epoch_peak_mb:
                    if step_peak_mb > args.target_memory_mb and is_main:
                        print(f"  [MEM WARNING] new peak {step_peak_mb:.0f} MB > target {args.target_memory_mb} MB | "
                              f"spacing={tuple(f'{s:.3f}' for s in spacing)} "
                              f"patch={tuple(images.shape[2:])} bs={batch_size}", flush=True)
                    epoch_peak_mb = step_peak_mb
                    epoch_peak_spacing = spacing
                    epoch_peak_patch = tuple(images.shape[2:])
                    epoch_peak_bs = batch_size

                # Only count non-NaN steps in the epoch loss; always count the
                # step itself so the DDP step-count loop terminates deterministically.
                if not bad:
                    train_loss += accum_loss / total_micro
                n_train_batches += 1

            train_loss /= max(1, n_train_batches - nan_skip_count)

            # Final safety: if the model parameters themselves have gone NaN,
            # abort training on all ranks rather than continue silently.
            param_bad = any(not torch.isfinite(p).all() for p in raw_model.parameters())
            if distributed:
                pb = torch.tensor(1 if param_bad else 0, device=device, dtype=torch.int)
                dist.all_reduce(pb, op=dist.ReduceOp.MAX)
                param_bad = bool(pb.item())
            if param_bad:
                raise RuntimeError(
                    f"Model parameters contain NaN/Inf at epoch {epoch} "
                    f"(rank {rank}, nan_skips this epoch: {nan_skip_count}). "
                    f"Inspect [NaN GUARD] log lines for the triggering batch."
                )

        # Step LR scheduler (all ranks must stay in sync)
        is_baseline = validate_only_first and epoch == 0
        if not is_baseline:
            scheduler.step()

        # Validation + logging + checkpointing: rank 0 only in DDP
        # Non-main ranks skip to the barrier at the end.
        if not is_main:
            if distributed:
                dist.barrier()
            continue

        # Validation - nnUNet-style global Dice calculation
        # Accumulate TP, FP, FN across all batches, then compute Dice at the end
        # This is much more stable than averaging per-batch Dice scores
        raw_model.eval()
        tp_hard = [0] * (n_classes - 1)  # Per-class TP (excluding background)
        fp_hard = [0] * (n_classes - 1)  # Per-class FP
        fn_hard = [0] * (n_classes - 1)  # Per-class FN
        epoch_val_resolutions = []  # Per-patch dice + spacing for this epoch
        epoch_group_counts = defaultdict(lambda: [0, 0, 0])  # group_key -> [tp, fp, fn]
        val_loss = 0.0
        n_val_batches = 0

        val_patches_per_group = getattr(args, 'val_patches_per_group', 0)
        max_vbs = getattr(args, 'max_val_batch_size', 0)
        n_val_patches = 0
        with torch.no_grad():
            if val_patches_per_group > 0:
                # Fixed number of patches per resolution group
                group_iter = val_loader.group_loaders.items()
            else:
                # Legacy: sample proportionally by weight
                group_iter = None

            def _val_forward(images, labels, spacing):
                """Run val forward on images, handling max_val_batch_size slicing."""
                all_preds = []
                total_loss = 0.0
                n_sub = 0
                bs = images.shape[0]
                sub_bs = max_vbs if max_vbs > 0 else bs
                for start in range(0, bs, sub_bs):
                    end = min(start + sub_bs, bs)
                    sub_img = images[start:end]
                    sub_lbl = labels[start:end]
                    with autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                        out = raw_model(sub_img, spacing=spacing)
                        if args.deep_supervision:
                            total_loss += criterion(out, downsample_seg_for_deep_supervision(sub_lbl, [o.shape[2:] for o in out])).item()
                            out = out[-1]
                        else:
                            total_loss += base_criterion(out, sub_lbl).item()
                    all_preds.append(out.argmax(dim=1))
                    n_sub += 1
                return torch.cat(all_preds, dim=0), total_loss / n_sub

            if group_iter is not None:
                for spacing, loader in group_iter:
                    group_patches = 0
                    while group_patches < val_patches_per_group:
                        batch = next(loader)

                        input_dtype = amp_dtype
                        images = batch['data'].to(device, dtype=input_dtype)
                        labels = batch['seg'].to(device).long().squeeze(1)
                        batch_size = images.shape[0]

                        preds, val_batch_loss = _val_forward(images, labels, spacing)
                        val_loss += val_batch_loss
                        n_val_batches += 1

                        # Use canonical group spacing for resolution tracking
                        gsp = batch.get('group_spacing', spacing)
                        gkey = (f'{gsp[0]:.3f}', f'{gsp[1]:.3f}', f'{gsp[2]:.3f}')

                        for b in range(batch_size):
                            patch_dice_scores = []
                            for c in range(1, n_classes):
                                pred_c = (preds[b] == c)
                                label_c = (labels[b] == c)
                                tp_b = (pred_c & label_c).sum().item()
                                fp_b = (pred_c & ~label_c).sum().item()
                                fn_b = (~pred_c & label_c).sum().item()
                                denom = 2 * tp_b + fp_b + fn_b
                                patch_dice_scores.append(2 * tp_b / denom if denom > 0 else 0.0)
                                epoch_group_counts[gkey][0] += tp_b
                                epoch_group_counts[gkey][1] += fp_b
                                epoch_group_counts[gkey][2] += fn_b
                            epoch_val_resolutions.append({
                                'spacing': spacing,
                                'dice': float(np.mean(patch_dice_scores))
                            })

                        for c in range(1, n_classes):
                            pred_c = (preds == c)
                            label_c = (labels == c)
                            tp_hard[c - 1] += (pred_c & label_c).sum().item()
                            fp_hard[c - 1] += (pred_c & ~label_c).sum().item()
                            fn_hard[c - 1] += (~pred_c & label_c).sum().item()

                        group_patches += batch_size
                        n_val_patches += batch_size
            else:
                while n_val_patches < args.val_patches:
                    batch, spacing = next(val_loader)

                    # Override with group spacing if requested
                    if getattr(args, 'use_group_spacing', False):
                        spacing = batch.get('group_spacing', spacing)

                    input_dtype = amp_dtype
                    images = batch['data'].to(device, dtype=input_dtype)
                    labels = batch['seg'].to(device).long().squeeze(1)
                    batch_size = images.shape[0]

                    preds, val_batch_loss = _val_forward(images, labels, spacing)
                    val_loss += val_batch_loss
                    n_val_batches += 1

                    # Use canonical group spacing for resolution tracking
                    gsp = batch.get('group_spacing', spacing)
                    gkey = (f'{gsp[0]:.3f}', f'{gsp[1]:.3f}', f'{gsp[2]:.3f}')

                    for b in range(batch_size):
                        patch_dice_scores = []
                        for c in range(1, n_classes):
                            pred_c = (preds[b] == c)
                            label_c = (labels[b] == c)
                            tp_b = (pred_c & label_c).sum().item()
                            fp_b = (pred_c & ~label_c).sum().item()
                            fn_b = (~pred_c & label_c).sum().item()
                            denom = 2 * tp_b + fp_b + fn_b
                            patch_dice_scores.append(2 * tp_b / denom if denom > 0 else 0.0)
                            epoch_group_counts[gkey][0] += tp_b
                            epoch_group_counts[gkey][1] += fp_b
                            epoch_group_counts[gkey][2] += fn_b
                        epoch_val_resolutions.append({
                            'spacing': spacing,
                            'dice': float(np.mean(patch_dice_scores))
                        })

                    for c in range(1, n_classes):
                        pred_c = (preds == c)
                        label_c = (labels == c)
                        tp_hard[c - 1] += (pred_c & label_c).sum().item()
                        fp_hard[c - 1] += (pred_c & ~label_c).sum().item()
                        fn_hard[c - 1] += (~pred_c & label_c).sum().item()

                    n_val_patches += batch_size

        # Compute global Dice per class: 2*TP / (2*TP + FP + FN)
        dice_per_class = []
        for c in range(n_classes - 1):
            denom = 2 * tp_hard[c] + fp_hard[c] + fn_hard[c]
            if denom > 0:
                dice_per_class.append(2 * tp_hard[c] / denom)

        # Mean foreground Dice (like nnUNet's mean_fg_dice)
        val_dice = np.mean(dice_per_class) if dice_per_class else 0.0

        # Average validation loss
        if n_val_batches > 0:
            val_loss /= n_val_batches

        # EMA pseudo-dice (nnUNet-style smoothing)
        if ema_pseudo_dice_history:
            ema_pseudo_dice = ema_pseudo_dice_history[-1] * 0.9 + 0.1 * val_dice
        else:
            ema_pseudo_dice = val_dice

        # Super-resolution validation (separate dice for SR task)
        sr_val_dice = None
        sr_group_stats = {}  # Per-group TP/FP/FN for breakdown logging
        if train_loader.superres_loaders:
            sr_tp = [0] * (n_classes - 1)
            sr_fp = [0] * (n_classes - 1)
            sr_fn = [0] * (n_classes - 1)
            sr_patches = 0
            sr_target = max(20, args.val_patches // 5)
            with torch.no_grad():
                sr_keys = list(train_loader.superres_loaders.keys())
                patches_per_key = max(1, sr_target // len(sr_keys))
                for sr_key in sr_keys:
                    sr_loader = train_loader.superres_loaders[sr_key]
                    # Per-group accumulators
                    g_tp = [0] * (n_classes - 1)
                    g_fp = [0] * (n_classes - 1)
                    g_fn = [0] * (n_classes - 1)
                    g_n = 0
                    for _ in range(patches_per_key):
                        sr_batch = next(sr_loader)
                        input_dtype = amp_dtype
                        sr_images = sr_batch['data'].to(device, dtype=input_dtype)
                        sr_labels = sr_batch['seg'].to(device).long().squeeze(1)
                        sr_spacing = tuple(sr_batch['spacings'][0])
                        sr_orig_spacing = tuple(sr_batch['orig_spacings'][0])

                        with autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                            sr_outputs = raw_model.forward_superres(
                                sr_images, sub_spacing=sr_spacing, orig_spacing=sr_orig_spacing
                            )
                            if sr_outputs.shape[2:] != sr_labels.shape[1:]:
                                sr_outputs = nn.functional.interpolate(
                                    sr_outputs, size=sr_labels.shape[1:],
                                    mode='trilinear', align_corners=True
                                )

                        sr_preds = sr_outputs.argmax(dim=1)
                        for c in range(1, n_classes):
                            pred_c = (sr_preds == c)
                            label_c = (sr_labels == c)
                            tp_val = (pred_c & label_c).sum().item()
                            fp_val = (pred_c & ~label_c).sum().item()
                            fn_val = (~pred_c & label_c).sum().item()
                            sr_tp[c - 1] += tp_val
                            sr_fp[c - 1] += fp_val
                            sr_fn[c - 1] += fn_val
                            g_tp[c - 1] += tp_val
                            g_fp[c - 1] += fp_val
                            g_fn[c - 1] += fn_val
                        sr_patches += 1
                        g_n += 1

                    # Compute per-group dice
                    g_dice_vals = []
                    for c in range(n_classes - 1):
                        denom = 2 * g_tp[c] + g_fp[c] + g_fn[c]
                        if denom > 0:
                            g_dice_vals.append(2 * g_tp[c] / denom)
                    g_dice = np.mean(g_dice_vals) if g_dice_vals else 0.0
                    # Build readable key: sub_sp -> orig_sp
                    if isinstance(sr_key, tuple) and len(sr_key) == 3 and sr_key[0] == 'superres':
                        sub_sp = sr_key[1]
                        orig_sp = sr_key[2]
                        label = (f"({sub_sp[0]:.2f},{sub_sp[1]:.2f},{sub_sp[2]:.2f})"
                                 f" -> ({orig_sp[0]:.2f},{orig_sp[1]:.2f},{orig_sp[2]:.2f})")
                    else:
                        label = str(sr_key)
                    sr_group_stats[label] = {'dice': g_dice, 'n': g_n}

            sr_dice_per_class = []
            for c in range(n_classes - 1):
                denom = 2 * sr_tp[c] + sr_fp[c] + sr_fn[c]
                if denom > 0:
                    sr_dice_per_class.append(2 * sr_tp[c] / denom)
            sr_val_dice = np.mean(sr_dice_per_class) if sr_dice_per_class else 0.0

        epoch_time = time.time() - epoch_start
        lr = scheduler.get_last_lr()[0]

        sr_str = f" | sr_dice: {sr_val_dice:.4f}" if sr_val_dice is not None else ""
        if is_baseline:
            log_msg = (f"Epoch {epoch:4d} | BASELINE (pretrained) | val_loss: {val_loss:.4f} "
                       f"| pseudo_dice: {val_dice:.4f} | ema_dice: {ema_pseudo_dice:.4f}{sr_str} "
                       f"| time: {epoch_time:.1f}s")
        else:
            mem_str = f" | peak_mem: {epoch_peak_mb:.0f}MB" if epoch_peak_mb > 0 else ""
            nan_str = f" | nan_skips: {nan_skip_count}" if nan_skip_count > 0 else ""
            log_msg = (f"Epoch {epoch:4d} | loss: {train_loss:.4f} | val_loss: {val_loss:.4f} "
                       f"| pseudo_dice: {val_dice:.4f} | ema_dice: {ema_pseudo_dice:.4f}{sr_str} "
                       f"| lr: {lr:.2e}{mem_str}{nan_str} | time: {epoch_time:.1f}s")
        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()

        # Append to history lists
        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)
        pseudo_dice_history.append(val_dice)
        ema_pseudo_dice_history.append(ema_pseudo_dice)
        lr_history.append(lr)
        epoch_times_history.append(epoch_time)

        # W&B per-epoch logging
        if wandb_run is not None:
            wandb_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'pseudo_dice': val_dice,
                'lr': lr,
                'epoch_time': epoch_time,
            }
            if epoch_peak_mb > 0:
                wandb_metrics['peak_memory_mb'] = epoch_peak_mb
            if sr_val_dice is not None:
                wandb_metrics['sr_dice'] = sr_val_dice
            wandb_run.log(wandb_metrics, step=epoch)

        # Structured JSON log entry
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'pseudo_dice': val_dice,
            'ema_dice': ema_pseudo_dice,
            'lr': lr,
            'epoch_time': epoch_time,
        }
        if epoch_peak_mb > 0:
            epoch_log['peak_memory_mb'] = epoch_peak_mb
        if sr_val_dice is not None:
            epoch_log['sr_dice'] = sr_val_dice
        json_log.append(epoch_log)
        with open(json_log_path, 'w') as f:
            json.dump(json_log, f, indent=1)

        # Store last epoch's val resolutions for plotting
        last_val_resolutions = epoch_val_resolutions

        # Push this epoch's per-group TP/FP/FN into rolling window
        _group_stats_window.append(dict(epoch_group_counts))

        # Log rolling-window per-group dice every epoch (after 10 epochs accumulated)
        if len(_group_stats_window) >= 10:
            # Sum TP/FP/FN across the window for each group
            window_totals = defaultdict(lambda: [0, 0, 0])
            for epoch_counts in _group_stats_window:
                for gkey, (tp, fp, fn) in epoch_counts.items():
                    window_totals[gkey][0] += tp
                    window_totals[gkey][1] += fp
                    window_totals[gkey][2] += fn

            # Compute dice from aggregated counts
            wandb_group_metrics = {}
            group_dice_log = {}
            for gkey in sorted(window_totals.keys()):
                tp, fp, fn = window_totals[gkey]
                denom = 2 * tp + fp + fn
                dice_val = 2 * tp / denom if denom > 0 else 0.0
                wandb_group_metrics[f'group_dice/{gkey}'] = dice_val
                group_dice_log[str(gkey)] = round(dice_val, 4)

            if wandb_group_metrics:
                if wandb_run is not None:
                    wandb_run.log(wandb_group_metrics, step=epoch)

            # Print every 10 epochs to avoid log spam
            if epoch % 10 == 0 and group_dice_log:
                print(f'  Val dice by resolution group (epoch {epoch}, 10-epoch window):')
                log_file.write(f'  Val dice by resolution group (epoch {epoch}, 10-epoch window):\n')
                for gkey in sorted(window_totals.keys()):
                    tp, fp, fn = window_totals[gkey]
                    denom = 2 * tp + fp + fn
                    dice_val = 2 * tp / denom if denom > 0 else 0.0
                    print(f'    {gkey}: {dice_val:.4f} (tp={tp}, fp={fp}, fn={fn})')
                    log_file.write(f'    {gkey}: {dice_val:.4f} (tp={tp}, fp={fp}, fn={fn})\n')
                log_file.flush()

            # Always append to JSON log
            epoch_log['group_dice'] = group_dice_log
            with open(json_log_path, 'w') as f:
                json.dump(json_log, f, indent=1)

            # SR dice by resolution group
            if sr_group_stats:
                print(f'  SR dice by resolution group (epoch {epoch}):')
                log_file.write(f'  SR dice by resolution group (epoch {epoch}):\n')
                for label in sorted(sr_group_stats.keys()):
                    stats = sr_group_stats[label]
                    print(f'    {label}: {stats["dice"]:.4f} (n={stats["n"]})')
                    log_file.write(f'    {label}: {stats["dice"]:.4f} (n={stats["n"]})\n')
                log_file.flush()

            plot_resolution_density(all_train_resolutions, last_val_resolutions, output_dir, epoch)

        # Plot progress every epoch (overwritten each time)
        plot_progress(train_losses_history, val_losses_history,
                      pseudo_dice_history, ema_pseudo_dice_history,
                      lr_history, epoch_times_history, output_dir)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': spacing_independent_state_dict(raw_model),
            'model_config': get_model_config(raw_model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_dice': val_dice,
            'best_val_dice': best_val_dice,
            'logging': {
                'train_losses': train_losses_history,
                'val_losses': val_losses_history,
                'pseudo_dice': pseudo_dice_history,
                'ema_pseudo_dice': ema_pseudo_dice_history,
                'lrs': lr_history,
                'epoch_times': epoch_times_history,
            },
            'config_hash': config_hash,
        }
        torch.save(checkpoint, output_dir / 'checkpoint.pt')

        # Save best model (using EMA pseudo-dice, like nnUNet)
        if ema_pseudo_dice > best_val_dice:
            best_val_dice = ema_pseudo_dice
            checkpoint['best_val_dice'] = best_val_dice
            torch.save(checkpoint, output_dir / 'model_best.pt')
            print(f"  New best EMA pseudo dice: {ema_pseudo_dice:.4f}")
            if wandb_run is not None:
                wandb_run.log({'best_pseudo_dice': best_val_dice}, step=epoch)

        # DDP barrier: all ranks wait for rank 0 to finish checkpointing
        if distributed:
            dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()
    if log_file is not None:
        log_file.close()
    if is_main:
        print(f"\nTraining complete. Best EMA pseudo dice: {best_val_dice:.4f}")

    # Run full-volume validation with NIfTI saving (rank 0 only)
    if is_main and not getattr(args, 'skip_validation', False):
        print(f"\n{'='*60}")
        print("Running post-training validation...")
        print(f"{'='*60}")
        try:
            from validate import run_validation
            gpu_id = local_rank if distributed else args.gpu
            run_validation(
                experiment_dir=output_dir,
                checkpoint='model_best.pt',
                gpu=gpu_id,
                save_predictions=True,
            )
        except Exception as e:
            print(f"WARNING: Post-training validation failed: {e}")
            print("Run manually: python validate.py", str(output_dir), "--save_predictions")

    # DDP cleanup
    if distributed:
        dist.destroy_process_group()


# =============================================================================
# Config File Support (for experiment planning workflow)
# =============================================================================

def load_experiment_config(config_path: Path) -> Dict:
    """Load experiment configuration from config.json.

    Args:
        config_path: Path to config.json

    Returns:
        Dict with complete experiment configuration
    """
    with open(config_path) as f:
        return json.load(f)


def args_from_config(config: Dict, config_path: Path = None, cli_resume: bool = False) -> argparse.Namespace:
    """Convert config dict to argparse.Namespace.

    Args:
        config: Configuration dictionary from config.json
        config_path: Path to config.json (used to derive output_dir if not in config)

    Returns:
        argparse.Namespace with all training arguments
    """
    # Extract flat arguments
    args_dict = {}

    # Data arguments
    args_dict['preprocessed_dir'] = config['data']['preprocessed_dir']
    args_dict['fold'] = config['data']['fold']

    # Model arguments
    model = config['model']
    args_dict['n_base_filters'] = model['n_base_filters']
    args_dict['n_downsample'] = model['n_downsample']
    args_dict['diameter'] = model['diameter']
    args_dict['num_radial_basis'] = model['num_radial_basis']
    args_dict['equivariance'] = model['equivariance']
    args_dict['normalization'] = model['normalization']
    args_dict['activation'] = model.get('activation', 'softplus')
    args_dict['dropout'] = model['dropout']
    args_dict['max_features'] = model['max_features']
    args_dict['irrep_ratios'] = model['irrep_ratios']
    args_dict['fill_to_max'] = model['fill_to_max']
    args_dict['kernel_trim_threshold'] = model.get('kernel_trim_threshold', 1.0)
    args_dict['kernel_trim_cross_section'] = model.get('kernel_trim_cross_section', 0.0)
    args_dict['kernel_growth'] = model.get('kernel_growth', 2.0)
    args_dict['sequential_sc'] = model.get('sequential_sc', False)
    args_dict['sc_mode'] = model.get('sc_mode', 'parallel')
    args_dict['fused_gate'] = model.get('fused_gate', True)
    args_dict['scale'] = model.get('scale', 2.0)
    args_dict['pool_mode'] = model.get('pool_mode', 'maxpool3d')
    args_dict['backend'] = model.get('backend', 'e3nn')
    args_dict['pyramid'] = model.get('pyramid', None)
    args_dict['pyramid_decay'] = model.get('pyramid_decay', 1.0)
    args_dict['bottleneck_kernel'] = model.get('bottleneck_kernel', 0)

    # Training arguments
    training = config['training']
    # Derive output_dir from config file location (parent directory)
    if config_path:
        args_dict['output_dir'] = str(config_path.parent)
    else:
        args_dict['output_dir'] = str(Path(config['data']['preprocessed_dir']).parent.parent / 'experiments')
    args_dict['epochs'] = training['epochs']
    args_dict['batch_size'] = training['batch_size']
    args_dict['learning_rate'] = training['learning_rate']
    args_dict['weight_decay'] = training['weight_decay']
    args_dict['grad_clip'] = training['grad_clip']
    args_dict['patch_size_mm'] = tuple(training['patch_size_mm'])
    args_dict['patches_per_epoch'] = training['patches_per_epoch']
    args_dict['val_patches'] = training['val_patches']
    args_dict['foreground_oversample'] = training['foreground_oversample']
    args_dict['dynamic_batch_size'] = training.get('dynamic_batch_size', True)
    args_dict['target_memory_mb'] = training['target_memory_mb']
    args_dict['min_batch_size'] = training['min_batch_size']
    args_dict['max_batch_size'] = training['max_batch_size']
    args_dict['pooling_factor'] = training['pooling_factor']
    args_dict['resolution_jitter_sigma'] = training['resolution_jitter_sigma']
    args_dict['scale_jitter_std'] = training.get('scale_jitter_std', 0.0)
    args_dict['num_workers'] = training['num_workers']
    args_dict['lr'] = args_dict.pop('learning_rate')

    # Augmentation arguments
    aug = config['augmentation']
    args_dict['disable_spatial'] = aug.get('disable_spatial', True)
    args_dict['disable_mirroring'] = aug['disable_mirroring']
    args_dict['subsample_weight'] = aug['subsample_weight']
    args_dict['min_spacing'] = aug.get('min_spacing', 0.0)
    args_dict['max_inplane_spacing'] = aug.get('max_inplane_spacing', 0.0)
    args_dict['min_slice_thickness'] = aug.get('min_slice_thickness', 0.0)
    args_dict['max_slice_thickness'] = aug.get('max_slice_thickness', 0.0)
    args_dict['min_loader_cases'] = aug.get('min_loader_cases', 2)
    args_dict['superres_training'] = aug.get('superres_training', False)
    args_dict['superres_weight'] = aug.get('superres_weight', 0.1)
    args_dict['group_balance'] = aug.get('group_balance', 0.0)
    args_dict['bias_field'] = aug.get('bias_field', True)
    args_dict['curriculum'] = aug.get('curriculum', None)
    args_dict['curriculum_bs_tiers'] = aug.get('curriculum_bs_tiers', None)
    args_dict['curriculum_phase_len'] = aug.get('curriculum_phase_len', 30)
    args_dict['use_group_spacing'] = aug.get('use_group_spacing', False)

    # Hardware arguments
    hw = config['hardware']
    args_dict['gpu'] = hw['gpu']
    args_dict['fp16'] = hw.get('fp16', True)
    args_dict['deep_supervision'] = hw.get('deep_supervision', True)
    args_dict['no_background_dice'] = hw.get('no_background_dice', False)
    args_dict['batch_dice'] = hw.get('batch_dice', False)

    # Resume flag: honour CLI --resume when loading from config
    args_dict['resume'] = cli_resume
    args_dict['plan_only'] = False
    args_dict['init_checkpoint'] = training.get('init_checkpoint', config.get('init_checkpoint', None))

    # W&B arguments (default off so existing configs work unchanged)
    args_dict['wandb'] = training.get('wandb', False)
    args_dict['wandb_project'] = training.get('wandb_project', 'irrepunet')
    args_dict['wandb_name'] = training.get('wandb_name', None)

    return argparse.Namespace(**args_dict)


def main():
    parser = argparse.ArgumentParser(description='Train e3nnUNet with nnUNet-style batchgenerators')

    # Config file argument (optional)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.json from --plan_only. '
                            'If provided, other arguments are ignored.')

    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--preprocessed_dir', type=str, required=False, default=None,
                           help='Path to preprocessed irrepunet directory (required if --config not provided)')
    data_group.add_argument('--fold', type=int, default=0,
                           help='Cross-validation fold (default: 0)')

    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--n_base_filters', type=int, default=2,
                            help='Base number of filters (default: 2)')
    model_group.add_argument('--n_downsample', type=int, default=4,
                            help='Number of downsampling steps (default: 4)')
    model_group.add_argument('--diameter', type=float, default=5.0,
                            help='Convolution kernel diameter in mm (default: 5.0)')
    model_group.add_argument('--num_radial_basis', type=int, default=5,
                            help='Number of radial basis functions (default: 5)')
    model_group.add_argument('--equivariance', type=str, default='SO3', choices=['SO3', 'O3'],
                            help='Equivariance type (default: SO3)')
    model_group.add_argument('--normalization', type=str, default='instance',
                            choices=['batch', 'instance', 'layer', 'None'],
                            help='Normalization type (default: instance)')
    model_group.add_argument('--activation', type=str, default='softplus',
                            choices=['softplus', 'selu', 'relu', 'silu', 'gelu'],
                            help='Scalar activation function for Gate layers (default: softplus)')
    model_group.add_argument('--dropout', type=float, default=0.0,
                            help='Dropout probability (default: 0.0)')
    model_group.add_argument('--max_features', type=int, default=320,
                            help='Max features per level, like nnUNet (default: 320)')
    model_group.add_argument('--irrep_ratios', type=int, nargs='+', default=[4, 2, 1],
                            help='Irrep multipliers for l=0,1,2 (default: 4 2 1)')
    model_group.add_argument('--fill_to_max', action='store_true',
                            help='Top up capped levels with scalar irreps to reach max_features')
    model_group.add_argument('--kernel_trim_threshold', type=float, default=1.0,
                            help='Trim outermost kernel shell when s*step/r > threshold (default: 1.0 = no trim)')
    model_group.add_argument('--kernel_trim_cross_section', type=float, default=0.0,
                            help='Cross-section fraction trim threshold (default: 0.0 = disabled). '
                                 'Trims shells where sqrt(1-(s*step/r)^2) < threshold. Meaningful values: 0.3-0.5. '
                                 'Takes priority over --kernel_trim_threshold when > 0.')
    model_group.add_argument('--kernel_growth', type=float, default=2.0,
                            help='Per-level diameter growth factor (default: 2.0). '
                                 'Lower values (e.g. 1.9) reduce bottleneck kernel radius.')
    model_group.add_argument('--sequential_sc', action='store_true',
                            help='(Legacy) Alias for --sc_mode sc_first')
    model_group.add_argument('--sc_mode', type=str, default='parallel',
                            choices=['parallel', 'sc_first', 'conv_first', 'sc_first_res', 'conv_first_res', 'none'],
                            help='Self-connection mode: parallel (default, sc+conv), sc_first (conv(sc(x))), '
                                 'conv_first (sc(conv(x))), sc_first_res (sc(x)+conv(sc(x))), '
                                 'conv_first_res (conv(x)+sc(conv(x))), '
                                 'none (no SC, cutoff=False so center weight is learnable)')
    model_group.add_argument('--fused_gate', action=argparse.BooleanOptionalAction, default=True,
                            help='Use FusedGate (derive gates from l=0 via linear) instead of e3nn Gate '
                                 '(default: True). Use --no-fused_gate for legacy Gate.')
    model_group.add_argument('--bottleneck_kernel', type=int, default=0,
                            help='Optimize diameter/scale/trim for this kernel size at bottleneck '
                                 '(0=disabled, 3 or 5). Overrides --diameter and --kernel_trim_threshold.')
    model_group.add_argument('--pool_mode', type=str, default='maxpool3d',
                            choices=['maxpool3d', 'average', 's2d'],
                            help='Pooling mode: maxpool3d (default), average, or s2d (SH space-to-depth)')
    model_group.add_argument('--scale', type=float, default=2.0,
                            help='Base pooling scale in mm (default: 2.0). Physical unit for first pool level; '
                                 'subsequent levels use scale * 2^i. Normally set automatically by --bottleneck_kernel.')
    model_group.add_argument('--backend', type=str, default='e3nn',
                            choices=['e3nn'],
                            help='Convolution backend (only e3nn is supported).')
    model_group.add_argument('--pyramid', type=str, default=None, nargs='?', const='scatter',
                            choices=['scatter', 'interp'],
                            help='Enable pyramid kernel convolution. Optional mode: scatter (default) or interp.')
    model_group.add_argument('--pyramid_decay', type=float, default=1.0,
                            help='Pyramid level decay factor (default: 1.0 = equal weights). '
                                 'Level k gets weight decay^k.  Weights are always non-learnable.')

    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--output_dir', type=str, required=False, default=None,
                            help='Output directory for checkpoints and logs (required if --config not provided)')
    train_group.add_argument('--epochs', type=int, default=1000,
                            help='Number of epochs (default: 1000)')
    train_group.add_argument('--batch_size', type=int, default=2,
                            help='Batch size (default: 2)')
    train_group.add_argument('--dynamic_batch_size', action=argparse.BooleanOptionalAction, default=True,
                            help='Automatically scale batch size per resolution group to fill GPU memory '
                                 '(default: True). Use --no-dynamic_batch_size to disable.')
    train_group.add_argument('--target_memory_mb', type=float, default=8000,
                            help='Target GPU memory in MB for dynamic batch sizing (required if --dynamic_batch_size)')
    train_group.add_argument('--min_batch_size', type=int, default=1,
                            help='Minimum effective batch size (default: 1). Groups with smaller '
                                 'GPU batch size accumulate gradients to reach this.')
    train_group.add_argument('--max_batch_size', type=int, default=24,
                            help='Maximum batch size for dynamic sizing (default: 32)')
    train_group.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate (default: 0.01)')
    train_group.add_argument('--weight_decay', type=float, default=3e-5,
                            help='Weight decay (default: 3e-5)')
    train_group.add_argument('--grad_clip', type=float, default=12.0,
                            help='Gradient clipping (default: 12.0)')
    train_group.add_argument('--patch_size_mm', type=float, nargs=3, default=[80, 80, 80],
                            help='Patch size in mm (default: 51.2 51.2 51.2)')
    train_group.add_argument('--patches_per_epoch', type=int, default=500,
                            help='Training patches per epoch (default: 500)')
    train_group.add_argument('--val_patches', type=int, default=100,
                            help='Validation patches per epoch (default: 100)')
    train_group.add_argument('--val_patches_per_group', type=int, default=0,
                            help='Fixed number of val patches per resolution group. '
                                 'If >0, overrides --val_patches with equal sampling per group.')
    train_group.add_argument('--max_val_batch_size', type=int, default=0,
                            help='Max batch size during validation (0=use training batch sizes). '
                                 'Reduces GPU memory pressure at the cost of slower validation.')
    train_group.add_argument('--foreground_oversample', type=float, default=0.33,
                            help='Foreground oversampling ratio (default: 0.33)')
    train_group.add_argument('--subsample_weight', type=float, default=0.0,
                            help='Weight for preprocessed subsampled data (default: 0.0, disabled). '
                                 'Use preprocess.py subsample to create subsampled versions first.')
    train_group.add_argument('--group_balance', type=float, default=0.0,
                            help='Balance sampling across resolution groups (default: 0.0). '
                                 '0=proportional to case count, 1=uniform across groups.')
    train_group.add_argument('--curriculum', type=int, nargs='+', default=None, metavar='EPOCH',
                            help='Staged resolution introduction at these epoch boundaries. '
                                 'E.g., --curriculum 10 20 30 introduces groups in 4 stages: '
                                 'stage 0 (epochs 0-9): ~1mm iso only, '
                                 'stage 1 (10-19): add 0.75mm in-plane, '
                                 'stage 2 (20-29): add 0.5mm in-plane, '
                                 'stage 3 (30+): all groups.')
    train_group.add_argument('--curriculum_bs_tiers', type=int, nargs='+', default=None, metavar='BS',
                            help='Batch-size curriculum: descending batch-size thresholds. '
                                 'Each tier activates spacing groups with batch_size >= threshold. '
                                 'E.g., --curriculum_bs_tiers 12 8 4 2 1 with --curriculum_phase_len 112 '
                                 'onboards coarse-to-fine resolution groups over 5 phases.')
    train_group.add_argument('--curriculum_phase_len', type=int, default=30,
                            help='Epochs per phase for --curriculum_bs_tiers (default: 30)')
    train_group.add_argument('--superres_training', action='store_true',
                            help='Enable super-resolution training: model receives low-res input, '
                                 'produces high-res output using original-resolution labels')
    train_group.add_argument('--superres_weight', type=float, default=0.1,
                            help='Sampling weight for super-resolution batches (default: 0.1)')
    train_group.add_argument('--min_spacing', type=float, default=0.0,
                            help='Exclude cases with finest spacing below this value in mm (default: 0.0, include all)')
    train_group.add_argument('--max_inplane_spacing', type=float, default=0.0,
                            help='Exclude cases with in-plane spacing above this value in mm (default: 0.0, include all)')
    train_group.add_argument('--min_slice_thickness', type=float, default=0.0,
                            help='Exclude cases with slice thickness below this value in mm (default: 0.0, include all)')
    train_group.add_argument('--max_slice_thickness', type=float, default=0.0,
                            help='Exclude cases with slice thickness above this value in mm (default: 0.0, include all)')
    train_group.add_argument('--min_loader_cases', type=int, default=2,
                            help='Minimum cases per loader group (groups with fewer cases are skipped)')
    train_group.add_argument('--pooling_factor', type=int, default=8,
                            help='Ensure patch sizes divisible by this (default: 8)')
    train_group.add_argument('--resolution_jitter_sigma', type=float, default=0.0,
                            help='Std dev of Gaussian jitter applied to spacing (default: 0.0, disabled)')
    train_group.add_argument('--scale_jitter_std', type=float, default=0.0,
                            help='Std dev of multiplicative Gaussian jitter applied to pooling scales '
                                 '(default: 0.0, disabled). Smooths architecture boundaries. '
                                 'Recommended: 0.05 (5%% jitter).')
    train_group.add_argument('--num_workers', type=int, default=1,
                            help='Number of dataloader workers per resolution group (default: 1)')
    train_group.add_argument('--resume', action='store_true',
                            help='Resume from checkpoint')
    train_group.add_argument('--skip_validation', action='store_true',
                            help='Skip post-training validation')
    train_group.add_argument('--init_checkpoint', type=str, default=None,
                            help='Initialize model weights from a pretrained checkpoint '
                                 '(fresh optimizer/scheduler, epoch 0)')

    # Augmentation arguments
    aug_group = parser.add_argument_group('Augmentation')
    aug_group.add_argument('--disable_spatial', action=argparse.BooleanOptionalAction, default=True,
                          help='Disable spatial augmentation (default: True). '
                               'Use --no-disable_spatial to enable spatial augs.')
    aug_group.add_argument('--disable_mirroring', action='store_true',
                          help='Disable mirroring augmentation')
    aug_group.add_argument('--no_bias_field', dest='bias_field', action='store_false',
                          help='Disable random MRI bias field augmentation')
    parser.set_defaults(bias_field=True)
    aug_group.add_argument('--deep_supervision', action=argparse.BooleanOptionalAction, default=True,
                          help='Enable deep supervision (nnUNet-style multi-scale loss). '
                               'Default: True.  Use --no-deep_supervision to disable.')
    aug_group.add_argument('--use_group_spacing', action='store_true',
                          help='Use canonical group spacing for model instead of per-case native spacing. '
                               'Replicates JAX JIT behavior for controlled comparison.')
    aug_group.add_argument('--no_background_dice', action='store_true',
                          help='Exclude background class from Dice loss (foreground only)')
    aug_group.add_argument('--batch_dice', action='store_true',
                          help='Use batch dice (sum over batch before ratio) instead of '
                               'per-sample dice (default). Per-sample gives equal weight to '
                               'each sample regardless of lesion volume.')

    # Hardware arguments
    hw_group = parser.add_argument_group('Hardware')
    hw_group.add_argument('--gpu', type=int, default=0,
                         help='GPU device ID (default: 0)')
    hw_group.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True,
                         help='Use BF16 mixed precision training (default: True). '
                              'Flag kept as --fp16 for config compat. Use --no-fp16 for FP32.')
    hw_group.add_argument('--ddp_sync_groups', action='store_true',
                         help='In multi-GPU DDP, force all ranks to train on the same spacing group '
                              'each step (synchronized). Default is async: each rank independently '
                              'picks groups, giving better resolution diversity per step.')
    parser.add_argument('--plan_only', action='store_true',
                        help='Generate config files without training')

    # W&B tracking arguments
    wandb_group = parser.add_argument_group('Weights & Biases')
    wandb_group.add_argument('--wandb', action='store_true',
                             help='Enable W&B logging (default: off)')
    wandb_group.add_argument('--wandb_project', type=str, default='irrepunet',
                             help='W&B project name (default: irrepunet)')
    wandb_group.add_argument('--wandb_name', type=str, default=None,
                             help='W&B run name (default: output_dir basename)')

    args = parser.parse_args()

    # Resolve sc_mode from --sequential_sc legacy flag
    if getattr(args, 'sc_mode', None) is None:
        if getattr(args, 'sequential_sc', False):
            args.sc_mode = 'sc_first'
        else:
            args.sc_mode = 'parallel'

    # Plan-only mode: generate configs and exit
    if getattr(args, 'plan_only', False):
        if args.config:
            print("Error: --plan_only and --config are mutually exclusive", file=sys.stderr)
            sys.exit(1)
        if not args.preprocessed_dir:
            print("Error: --preprocessed_dir is required with --plan_only", file=sys.stderr)
            sys.exit(1)
        if not args.output_dir:
            print("Error: --output_dir is required with --plan_only", file=sys.stderr)
            sys.exit(1)
        planner = ExperimentPlanner(args)
        try:
            plan_result = planner.plan_experiment()
            planner.save_configs(plan_result)
            print(f"\nExperiment planned successfully!")
            print(f"  Config: {args.output_dir}/config.json")
            print(f"  Loader: {args.output_dir}/loader_config.txt")
            print(f"  Run:    python train.py --config {args.output_dir}/config.json")
        except Exception as e:
            print(f"Error during planning: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)

    # Load from config file if provided
    config_hash = None
    planned_batch_sizes = None

    if args.config:
        print(f"Loading experiment configuration from {args.config}")
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        # Compute config hash for integrity checking
        config_bytes = config_path.read_bytes()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        print(f"Config hash (SHA256): {config_hash[:16]}...")

        config = load_experiment_config(config_path)
        cli_resume = args.resume
        cli_wandb = args.wandb
        cli_wandb_project = getattr(args, 'wandb_project', 'irrepunet')
        cli_wandb_name = getattr(args, 'wandb_name', None)
        args = args_from_config(config, config_path=config_path, cli_resume=cli_resume)
        # CLI W&B flags override config (allow enabling W&B without editing config)
        if cli_wandb:
            args.wandb = True
            args.wandb_project = cli_wandb_project
            if cli_wandb_name:
                args.wandb_name = cli_wandb_name

        # Extract planned batch sizes from loader_groups
        if 'loader_groups' in config:
            planned_batch_sizes = {}
            for group in config['loader_groups']:
                spacing_key = tuple(float(s) for s in group['spacing'])
                planned_batch_sizes[spacing_key] = group['batch_size']
            print(f"Loaded planned batch sizes for {len(planned_batch_sizes)} spacing groups")

        print(f"Configuration loaded successfully")
    else:
        # Legacy mode: validate required arguments
        if not args.preprocessed_dir:
            print("Error: --preprocessed_dir is required (or provide --config)", file=sys.stderr)
            sys.exit(1)
        if not args.output_dir:
            print("Error: --output_dir is required (or provide --config)", file=sys.stderr)
            sys.exit(1)

    train(args, config_hash=config_hash, planned_batch_sizes=planned_batch_sizes)


if __name__ == '__main__':
    main()
