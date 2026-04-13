#!/usr/bin/env python
"""
Full-volume validation with distilled inference and per-group analysis.

Loads a trained e3nn model, distills to pure PyTorch per architecture group,
runs sliding window inference (same method as infer_case.py) with Gaussian
weighting and reflect-padding, then displays results grouped by resolution
and imaging plane.

Optionally saves NIfTI predictions (--save_predictions).

Usage:
    python validate.py experiments/plan_80mm_maxip1.1
    python validate.py experiments/plan_80mm_maxip1.1 --checkpoint model_best.pt
    python validate.py experiments/plan_80mm_maxip1.1 --gpu 1
    python validate.py experiments/plan_80mm_maxip1.1 --overlap 0.69
    python validate.py experiments/plan_80mm_maxip1.1 --save_predictions
    python validate.py experiments/plan_80mm_maxip1.1 --mirror_tta
"""

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
from nibabel import orientations as nib_orient
import numpy as np
import torch

from irrepunet.data.multi_resolution_loader import estimate_batch_size
from irrepunet.inference import sliding_window_inference as _sw_inference
from irrepunet.models import (
    project_to_spacing,
    architecture_spacing_range,
    update_projected_weights,
)
from irrepunet.models.unet import E3nnUNet


# ---------------------------------------------------------------------------
# Inference helper — wraps irrepunet.inference.sliding_window_inference
# with per-architecture-group model caching and weight updates.
# ---------------------------------------------------------------------------

def run_inference(image, model, spacing, patch_mm, overlap,
                  device_str, mirror_axes, sw_batch_size,
                  native_e3nn=False):
    """Run sliding_window_inference from irrepunet.inference on preprocessed data.

    Uses the same reflect-padding, Gaussian weighting, and batched patch
    extraction as infer_case.py / predict_nifti.

    Parameters
    ----------
    image : np.ndarray
        (C, D, H, W) preprocessed (cropped, normalized) image.
    model : nn.Module
        Distilled Conv3d model or native e3nn model.
    spacing : tuple
        Voxel spacing in mm.
    patch_mm : float
        Patch size in mm.
    overlap : float
        Overlap fraction.
    device_str : str
        CUDA device string.
    mirror_axes : tuple or None
        Axes for mirror TTA.
    sw_batch_size : int
        Patches per forward pass.
    native_e3nn : bool
        If True, skip projection and pass spacing to model forward call.

    Returns
    -------
    np.ndarray
        Softmax probabilities (C, D, H, W).
    """
    return _sw_inference(
        model=model,
        image=image,
        spacing=spacing,
        patch_size_mm=patch_mm,
        overlap=overlap,
        device=device_str,
        use_fp16=True,
        mirror_axes=mirror_axes,
        sw_batch_size=sw_batch_size,
        native_e3nn=native_e3nn,
    )


def dice_score(pred, target):
    """Compute foreground dice score."""
    pred_fg = (pred == 1).astype(float)
    tgt_fg = (target == 1).astype(float)
    intersection = (pred_fg * tgt_fg).sum()
    union = pred_fg.sum() + tgt_fg.sum()
    if union == 0:
        return 1.0 if tgt_fg.sum() == 0 else 0.0
    return (2.0 * intersection) / union


# ---------------------------------------------------------------------------
# NIfTI saving — reconstruct from preprocessed data + pkl metadata
# ---------------------------------------------------------------------------

def save_prediction_nifti(pred_cropped, props, output_path):
    """Save a cropped prediction array as a NIfTI in original image space.

    Requires augmented pkl metadata (original_affine, original_nifti_shape).
    Run augment_pkl_metadata.py to add this metadata if missing.

    Parameters
    ----------
    pred_cropped : np.ndarray
        (D, H, W) integer prediction in cropped RAS space.
    props : dict
        Case properties from pkl file.
    output_path : Path
        Output NIfTI path.
    """
    original_affine = props['original_affine']
    original_nifti_shape = props['original_nifti_shape']
    original_shape = props['original_shape']  # RAS shape before crop
    bbox = props['bbox']

    # Uncrop: place prediction into full RAS volume
    full_ras = np.zeros(original_shape, dtype=np.int16)
    slices = tuple(slice(b[0], b[1]) for b in bbox)
    full_ras[slices] = pred_cropped

    # Reverse RAS reorientation to original orientation
    original_ornt = nib.io_orientation(original_affine)
    ras_ornt = nib_orient.axcodes2ornt(('R', 'A', 'S'))
    reverse_transform = nib_orient.ornt_transform(ras_ornt, original_ornt)
    output_data = nib_orient.apply_orientation(full_ras, reverse_transform).astype(np.int16)

    # Save with original affine
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_nii = nib.Nifti1Image(output_data, original_affine)
    out_nii.header.set_data_dtype(np.int16)
    nib.save(out_nii, str(output_path))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(experiment_dir, checkpoint_name, device):
    """Load model from experiment config + checkpoint."""
    config_path = experiment_dir / 'config.json'
    checkpoint_path = experiment_dir / checkpoint_name

    with open(config_path) as f:
        config = json.load(f)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, checkpoint = E3nnUNet.load_checkpoint(checkpoint_path, device=device)
    print(f"Loaded {type(model).__name__}")

    ckpt_epoch = checkpoint.get('epoch', '?')
    ckpt_val_dice = checkpoint.get('val_dice', '?')
    print(f"Checkpoint epoch: {ckpt_epoch}, val_dice: {ckpt_val_dice}")

    return model, config, ckpt_epoch, ckpt_val_dice


# ---------------------------------------------------------------------------
# Architecture grouping
# ---------------------------------------------------------------------------

def build_architecture_groups(model, spacings):
    """Group spacings by architecture equivalence.

    Returns (arch_groups, spacing_to_rep) where arch_groups maps
    representative spacing -> list of equivalent spacings.
    """
    unique_spacings = sorted(set(spacings))
    arch_groups = {}
    spacing_to_rep = {}

    for sp in unique_spacings:
        found = False
        for rep in arch_groups:
            try:
                ranges = architecture_spacing_range(model, rep)
                in_range = all(lo < sp[d] <= hi for d, (lo, hi) in enumerate(ranges))
                if in_range:
                    arch_groups[rep].append(sp)
                    spacing_to_rep[sp] = rep
                    found = True
                    break
            except Exception:
                pass
        if not found:
            arch_groups[sp] = [sp]
            spacing_to_rep[sp] = sp

    return arch_groups, spacing_to_rep


# ---------------------------------------------------------------------------
# Analysis / display
# ---------------------------------------------------------------------------

def classify_plane(spacing):
    """Classify imaging plane from spacing: ax, cor, sag, or iso."""
    s = list(spacing)
    max_idx = s.index(max(s))
    max_val, min_val = max(s), min(s)
    if max_val / (min_val + 1e-9) < 1.5:
        return "iso"
    return ["ax", "cor", "sag"][max_idx]


def quantize_spacing(spacing):
    """Quantize to (in-plane, slice_thickness) resolution group."""
    s = sorted(spacing)
    if s[2] / (s[0] + 1e-9) < 1.5:
        mean_sp = round(round(np.mean(s) / 0.25) * 0.25, 2)
        return (mean_sp, mean_sp)
    else:
        inplane = [s[0], s[1]]
        inp = round(round(np.mean(inplane) / 0.25) * 0.25, 2)
        thk = round(round(s[2] / 1.0) * 1.0, 1)
        return (inp, thk)


def count_training_cases(preprocessed_dir, train_cases):
    """Count real and synthetic training cases by (resolution group, plane)."""
    real = defaultdict(lambda: defaultdict(int))
    synth = defaultdict(lambda: defaultdict(int))

    for case in train_cases:
        pkl = preprocessed_dir / f"{case}.pkl"
        if not pkl.exists():
            continue
        with open(pkl, 'rb') as f:
            props = pickle.load(f)
        sp = tuple(float(s) for s in props['spacing'])
        res = quantize_spacing(sp)
        plane = classify_plane(sp)
        real[res][plane] += 1

    for pkl in preprocessed_dir.glob("*_skip*.pkl"):
        base = pkl.stem.split('_skip')[0]
        if base not in train_cases:
            continue
        with open(pkl, 'rb') as f:
            props = pickle.load(f)
        sp = tuple(float(s) for s in props['spacing'])
        res = quantize_spacing(sp)
        plane = classify_plane(sp)
        synth[res][plane] += 1

    return real, synth


def print_results_table(results, preprocessed_dir=None, fold=0):
    """Print results grouped by resolution and imaging plane."""
    # Group validation cases
    val_groups = defaultdict(lambda: defaultdict(list))
    for c in results:
        sp = tuple(c['spacing'])
        res = quantize_spacing(sp)
        plane = classify_plane(sp)
        val_groups[res][plane].append(c['dice'])

    # Training counts
    show_train = False
    real_counts = defaultdict(lambda: defaultdict(int))
    synth_counts = defaultdict(lambda: defaultdict(int))

    if preprocessed_dir is not None:
        splits_path = preprocessed_dir / 'splits_final.json'
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
            train_cases = set(splits[fold]['train'])
            real_counts, synth_counts = count_training_cases(preprocessed_dir, train_cases)
            show_train = True

    all_res = sorted(val_groups.keys())

    if show_train:
        header = f"{'In-plane':>8} {'Slice':>6} {'Plane':>5}  {'N val':>5}  {'Real':>5} {'Synth':>5}  {'Dice':>6} {'Std':>6}"
        sep = "-" * 62
    else:
        header = f"{'In-plane':>8} {'Slice':>6} {'Plane':>5}  {'N val':>5}  {'Dice':>6} {'Std':>6}"
        sep = "-" * 48

    print()
    print(header)
    print(sep)

    plane_all = defaultdict(list)

    for res in all_res:
        inp, thk = res
        is_iso = abs(inp - thk) < 0.3
        res_label = f"{inp:>6.2f}mm"
        thk_label = "  iso" if is_iso else f"{thk:>4.0f}mm"

        planes_here = [p for p in ['ax', 'cor', 'sag', 'iso'] if p in val_groups[res]]

        for pi, plane in enumerate(planes_here):
            dices = val_groups[res][plane]
            n = len(dices)
            mean = np.mean(dices)
            std = np.std(dices) if n > 1 else 0.0
            std_str = f"{std:.4f}" if n > 1 else "   -  "
            plane_all[plane].extend(dices)

            res_str = f"{res_label} {thk_label}" if pi == 0 else " " * 15

            if show_train:
                r = real_counts[res].get(plane, 0)
                s = synth_counts[res].get(plane, 0)
                tr = f"{r:>5}" if r > 0 else "    -"
                sy = f"{s:>5}" if s > 0 else "    -"
                print(f"{res_str} {plane:>5}  {n:>5}  {tr} {sy}  {mean:.4f} {std_str}")
            else:
                print(f"{res_str} {plane:>5}  {n:>5}  {mean:.4f} {std_str}")

        if len(planes_here) > 1:
            all_d = []
            for plane in planes_here:
                all_d.extend(val_groups[res][plane])
            std_str = f"{np.std(all_d):.4f}" if len(all_d) > 1 else "   -  "
            if show_train:
                total_r = sum(real_counts[res].get(p, 0) for p in planes_here)
                total_s = sum(synth_counts[res].get(p, 0) for p in planes_here)
                tr = f"{total_r:>5}" if total_r > 0 else "    -"
                sy = f"{total_s:>5}" if total_s > 0 else "    -"
                print(f"{'':>15}   ALL  {len(all_d):>5}  {tr} {sy}  {np.mean(all_d):.4f} {std_str}")
            else:
                print(f"{'':>15}   ALL  {len(all_d):>5}  {np.mean(all_d):.4f} {std_str}")
        print()

    # Summary by plane
    print("=" * len(sep))
    print("SUMMARY BY PLANE:")
    for plane in ['ax', 'cor', 'sag', 'iso']:
        if plane in plane_all:
            d = plane_all[plane]
            print(f"  {plane:>5}: n={len(d):>3}, mean={np.mean(d):.4f}, "
                  f"median={np.median(d):.4f}, std={np.std(d):.4f}")
    all_dices = [d for v in plane_all.values() for d in v]
    print(f"  {'TOTAL':>5}: n={len(all_dices):>3}, mean={np.mean(all_dices):.4f}, "
          f"median={np.median(all_dices):.4f}, std={np.std(all_dices):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(
    experiment_dir,
    checkpoint='model_best.pt',
    gpu=0,
    overlap=0.69,
    sw_batch_size=4,
    output=None,
    save_predictions=True,
    pred_dir=None,
    no_train_counts=False,
    mirror_tta=False,
    native_e3nn=False,
):
    """Run full-volume validation on all validation cases.

    Can be called programmatically (e.g. from train.py) or via CLI.

    Parameters
    ----------
    experiment_dir : str or Path
        Experiment directory with config.json and checkpoint.
    checkpoint : str
        Checkpoint filename.
    gpu : int
        GPU device ID.
    overlap : float
        Sliding window overlap fraction.
    sw_batch_size : int
        Patches per forward pass.
    output : str, optional
        Output JSON filename (default: val_<checkpoint_stem>.json).
    save_predictions : bool
        Save NIfTI predictions.
    pred_dir : str or Path, optional
        Custom prediction output directory.
    no_train_counts : bool
        Skip training case count analysis in table.
    mirror_tta : bool
        Enable mirror TTA.
    native_e3nn : bool
        Native e3nn inference mode.

    Returns
    -------
    dict
        Validation results including per-case dice scores.
    """
    torch.set_float32_matmul_precision('high')

    experiment_dir = Path(experiment_dir)
    device_str = f'cuda:{gpu}'

    if output is None:
        stem = Path(checkpoint).stem
        suffixes = []
        if native_e3nn:
            suffixes.append('native')
        if mirror_tta:
            suffixes.append('mirror')
        tta_suffix = '_' + '_'.join(suffixes) if suffixes else ''
        output = f'val_{stem}{tta_suffix}.json'

    # Prediction output directory
    if save_predictions:
        pred_dir = Path(pred_dir) if pred_dir else experiment_dir / 'val_predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving NIfTI predictions to: {pred_dir}")
    else:
        pred_dir = None

    # Load model
    model, config, ckpt_epoch, ckpt_val_dice = load_model(
        experiment_dir, checkpoint, torch.device(device_str))

    # Support both flat and nested config formats
    data_config = config.get('data', config)
    preprocessed_dir = Path(data_config['preprocessed_dir'])
    fold = data_config.get('fold', 0)

    # Use val_cases from config if available, else load from splits
    if 'val_cases' in data_config:
        val_cases = data_config['val_cases']
    else:
        with open(preprocessed_dir / 'splits_final.json') as f:
            splits = json.load(f)
        val_cases = splits[fold]['val']

    training_config = config.get('training', config)
    patch_mm = training_config.get('patch_size_mm', [80, 80, 80])
    if isinstance(patch_mm, list):
        patch_mm = patch_mm[0]

    # Model params for dynamic sw_batch_size estimation
    model_config = config.get('model', config)
    n_base_filters = model_config.get('n_base_filters', 2)
    n_downsample = model_config.get('n_downsample', 4)
    gpu_mem_mb = torch.cuda.get_device_properties(gpu).total_memory / 1024**2
    target_memory_mb = gpu_mem_mb * 0.9

    # Load case metadata
    print("\nLoading case metadata...")
    case_props = {}
    nifti_ready = True
    for case in val_cases:
        with open(preprocessed_dir / f"{case}.pkl", 'rb') as f:
            props = pickle.load(f)
        case_props[case] = props
        if save_predictions and 'original_affine' not in props:
            nifti_ready = False

    if save_predictions and not nifti_ready:
        print("WARNING: pkl files missing NIfTI metadata (original_affine).")
        print("  Run: python augment_pkl_metadata.py --preprocessed_dir ... --raw_dir ...")
        print("  Continuing without NIfTI saving.")
        pred_dir = None

    case_spacings = {
        case: tuple(float(s) for s in props['spacing'])
        for case, props in case_props.items()
    }

    # Native e3nn mode: prepare model once
    if native_e3nn:
        native_model = model.to(torch.device(device_str)).eval()
        print("Native e3nn mode: skipping projection, spacing passed at forward time")

    # Build architecture groups
    all_spacings = list(case_spacings.values())
    arch_groups, spacing_to_rep = build_architecture_groups(model, all_spacings)

    unique_spacings = len(set(all_spacings))
    print(f"{unique_spacings} unique spacings -> {len(arch_groups)} architecture groups")

    # Mirror TTA setup
    mirror_axes = (0, 1, 2) if mirror_tta else None
    if mirror_tta:
        equivariance = config.get('equivariance', 'SO3')
        if equivariance == 'O3':
            print("WARNING: Mirror TTA with O3 equivariance — the model already handles "
                  "reflections via odd-parity irreps. Mirror TTA may provide little benefit "
                  "and will add 8x inference cost.")
        print("Mirror TTA: 8 passes (flips across all 3 axes)")

    # Sort cases by voxel count (smallest first) so GPU-light cases run first
    def _case_voxels(case):
        shape = case_props[case]['shape']
        return shape[0] * shape[1] * shape[2]

    val_cases_sorted = sorted(val_cases, key=_case_voxels)
    print(f"Sorted cases by voxel count: {_case_voxels(val_cases_sorted[0]):,} "
          f"-> {_case_voxels(val_cases_sorted[-1]):,}")

    # Run inference
    results = []
    skipped = []
    t0 = time.time()
    projected_cache = {}
    projection_times = {}
    gpu_rep = None  # which rep is currently on GPU

    for i, case in enumerate(val_cases_sorted):
        tc = time.time()
        spacing = case_spacings[case]
        rep = spacing_to_rep[spacing]

        if not native_e3nn:
            # Project or reuse (only one model on GPU at a time)
            try:
                if rep not in projected_cache:
                    # Evict current GPU model before projecting a new one
                    if gpu_rep is not None and gpu_rep in projected_cache:
                        projected_cache[gpu_rep].cpu()
                        torch.cuda.empty_cache()
                        gpu_rep = None
                    tp = time.time()
                    projected = project_to_spacing(model, rep)
                    projected.to(torch.device(device_str))
                    projected.eval()
                    proj_time = time.time() - tp
                    projection_times[rep] = proj_time
                    projected_cache[rep] = projected
                    gpu_rep = rep
                    print(f"  Projected ({rep[0]:.3f}, {rep[1]:.3f}, {rep[2]:.3f}) [{proj_time:.1f}s]")
                else:
                    projected = projected_cache[rep]
                    # Move to GPU if not already there
                    if gpu_rep != rep:
                        if gpu_rep is not None and gpu_rep in projected_cache:
                            projected_cache[gpu_rep].cpu()
                            torch.cuda.empty_cache()
                        projected.to(torch.device(device_str))
                        gpu_rep = rep

                # Update weights for exact spacing if needed
                if spacing != projected._current_spacing:
                    update_projected_weights(projected, model, spacing, verify=True)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if 'out of memory' not in str(e).lower() and 'CUDA' not in str(e):
                    raise
                torch.cuda.empty_cache()
                print(f"[{i+1}/{len(val_cases_sorted)}] {case}: OOM during projection — skipped")
                skipped.append({'case': case, 'spacing': list(spacing),
                                'shape': list(case_props[case]['shape']), 'reason': 'OOM_projection'})
                continue

        # Load preprocessed data
        img = np.load(preprocessed_dir / f"{case}.npy")
        seg = np.load(preprocessed_dir / f"{case}_seg.npy")
        if seg.ndim == 4:
            seg = seg[0]

        # Dynamic sw_batch_size: estimate from patch voxels and GPU memory
        patch_voxels = tuple(max(1, int(round(patch_mm / s))) for s in spacing)
        case_sw_batch = estimate_batch_size(
            patch_voxels, n_base_filters, target_memory_mb,
            min_batch=1, max_batch=sw_batch_size,
            n_downsample=n_downsample, fp16=True, mode='infer',
        )

        # Select model for inference
        inference_model = native_model if native_e3nn else projected

        # Inference with OOM protection (catches both torch.cuda.OutOfMemoryError
        # and RuntimeError from TorchScript OOM)
        try:
            probs = run_inference(
                img, inference_model, spacing, patch_mm, overlap,
                device_str, mirror_axes, case_sw_batch,
                native_e3nn=native_e3nn)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' not in str(e).lower() and 'CUDA' not in str(e):
                raise
            torch.cuda.empty_cache()
            voxels = _case_voxels(case)
            print(f"[{i+1}/{len(val_cases_sorted)}] {case}: OOM (voxels={voxels:,}, "
                  f"spacing=({spacing[0]:.2f},{spacing[1]:.2f},{spacing[2]:.2f})) — skipped")
            skipped.append({'case': case, 'spacing': list(spacing),
                            'shape': list(img.shape[1:]), 'reason': 'OOM'})
            continue

        pred_np = probs.argmax(axis=0).astype(np.int16)
        d = dice_score(pred_np, seg)
        elapsed = time.time() - tc

        # Save NIfTI prediction
        if pred_dir is not None:
            props = case_props[case]
            nifti_path = pred_dir / f"{case}.nii.gz"
            save_prediction_nifti(pred_np, props, nifti_path)

        results.append({
            'case': case,
            'spacing': list(spacing),
            'shape': list(img.shape[1:]),
            'dice': d,
            'time': round(elapsed, 2),
        })

        print(f"[{i+1}/{len(val_cases_sorted)}] {case}: dice={d:.4f}, "
              f"spacing=({spacing[0]:.2f},{spacing[1]:.2f},{spacing[2]:.2f}), "
              f"time={elapsed:.1f}s")

    total_time = time.time() - t0
    dices = [r['dice'] for r in results]

    # Save JSON
    output_data = {
        'experiment': str(experiment_dir),
        'checkpoint': checkpoint,
        'checkpoint_epoch': ckpt_epoch,
        'checkpoint_val_dice': ckpt_val_dice,
        'method': ('native_e3nn' if native_e3nn else 'distilled')
                  + ('_mirror' if mirror_tta else ''),
        'native_e3nn': native_e3nn,
        'overlap': overlap,
        'sw_batch_size': sw_batch_size,
        'weighting': 'gaussian',
        'padding': 'reflect',
        'mirror_tta': mirror_tta,
        'n_cases': len(results),
        'mean_dice': float(np.mean(dices)),
        'std_dice': float(np.std(dices)),
        'median_dice': float(np.median(dices)),
        'n_architecture_groups': len(arch_groups),
        'total_projection_time_s': sum(projection_times.values()),
        'total_time_s': total_time,
        'per_case': results,
    }
    if skipped:
        output_data['skipped'] = skipped
        output_data['n_skipped'] = len(skipped)
    if pred_dir is not None:
        output_data['predictions_dir'] = str(pred_dir)

    out_path = experiment_dir / output
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results saved to: {out_path}")
    if pred_dir is not None:
        print(f"NIfTI predictions saved to: {pred_dir}")
    print(f"Mean dice: {np.mean(dices):.4f} (median: {np.median(dices):.4f}, std: {np.std(dices):.4f})")
    print(f"Completed: {len(results)}/{len(val_cases_sorted)} cases "
          f"({len(skipped)} skipped due to OOM)")
    print(f"Total time: {total_time:.0f}s ({total_time/len(results):.1f}s/case)")
    print(f"Projection: {len(arch_groups)} groups, {sum(projection_times.values()):.0f}s total")

    # Print grouped results table
    ppd = preprocessed_dir if not no_train_counts else None
    print_results_table(results, preprocessed_dir=ppd, fold=fold)

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Full-volume validation with distilled inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('experiment_dir', type=str,
                        help='Experiment directory with config.json and checkpoint')
    parser.add_argument('--checkpoint', type=str, default='model_best.pt',
                        help='Checkpoint filename')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--overlap', type=float, default=0.69,
                        help='Sliding window overlap fraction')
    parser.add_argument('--sw_batch_size', type=int, default=4,
                        help='Patches per forward pass')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON filename (default: val_<checkpoint_stem>.json)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save NIfTI predictions to experiment_dir/val_predictions/')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Custom prediction output directory (default: experiment_dir/val_predictions)')
    parser.add_argument('--no_train_counts', action='store_true',
                        help='Skip training case count analysis in table')
    parser.add_argument('--mirror_tta', action='store_true',
                        help='Enable mirror (flip) test-time augmentation across all 3 axes (8 passes)')
    parser.add_argument('--native_e3nn', action='store_true',
                        help='Native e3nn inference: skip projection, pass spacing at forward time. '
                             'Required for runtime pyramid cap control.')
    args = parser.parse_args()

    run_validation(
        experiment_dir=args.experiment_dir,
        checkpoint=args.checkpoint,
        gpu=args.gpu,
        overlap=args.overlap,
        sw_batch_size=args.sw_batch_size,
        output=args.output,
        save_predictions=args.save_predictions,
        pred_dir=args.pred_dir,
        no_train_counts=args.no_train_counts,
        mirror_tta=args.mirror_tta,
        native_e3nn=args.native_e3nn,
    )


if __name__ == '__main__':
    main()
