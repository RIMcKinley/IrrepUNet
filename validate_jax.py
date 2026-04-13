#!/usr/bin/env python
"""
Full-volume validation inference for JAX E3nnUNet.

Loads a trained JAX model checkpoint, runs sliding window inference on all
validation cases, computes per-case dice, and optionally saves NIfTI predictions.

Usage:
    python validate_jax.py experiments/jax_targeted_pooling_450ep
    python validate_jax.py experiments/jax_targeted_pooling_450ep --checkpoint model_best.pkl
    python validate_jax.py experiments/jax_targeted_pooling_450ep --save_predictions
    python validate_jax.py experiments/jax_targeted_pooling_450ep --mirror_tta --overlap 0.75
    python validate_jax.py experiments/jax_targeted_pooling_450ep --gpu 1
"""

import argparse
import json
import pickle
import threading
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import nibabel as nib
from nibabel import orientations as nib_orient
import numpy as np

from irrepunet.models_jax import E3nnUNet
from irrepunet.training.utils import args_from_config
from irrepunet.data.spacing import round_spacing_to_tolerance


# =============================================================================
# Model loading
# =============================================================================

def load_jax_model(experiment_dir, checkpoint_name='model_best.pkl'):
    """Load JAX E3nnUNet from experiment config + checkpoint.

    Returns (model, config, checkpoint_metadata).
    """
    config_path = experiment_dir / 'config.json'
    checkpoint_path = experiment_dir / checkpoint_name

    with open(config_path) as f:
        config = json.load(f)

    # Build model from config
    args = args_from_config(config)
    ref_spacing = tuple(config.get('reference_spacing', [1.0, 1.0, 1.0]))
    n_classes = config.get('n_classes', 2)

    irrep_ratios = tuple(args.irrep_ratios) if isinstance(args.irrep_ratios, list) else args.irrep_ratios
    lmax = len(args.irrep_ratios) - 1 if isinstance(args.irrep_ratios, (list, tuple)) else 2

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
        lmax=lmax,
        pool_mode=getattr(args, 'pool_mode', 'maxpool3d'),
        scale=getattr(args, 'scale', 2.0),
        dropout_prob=args.dropout,
        scalar_upsampling=False,
        cutoff=True,
        deep_supervision=args.deep_supervision,
        max_features=args.max_features,
        irrep_ratios=irrep_ratios,
        fill_to_max=args.fill_to_max,
        kernel_trim_threshold=getattr(args, 'kernel_trim_threshold', 1.0),
        kernel_trim_cross_section=getattr(args, 'kernel_trim_cross_section', 0.0),
        sphere_norm=True,
        activation=getattr(args, 'activation', 'softplus'),
        rngs=nnx.Rngs(42),
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    model_state = jax.tree.map(jnp.array, data['model_state'])
    nnx.update(model, model_state)

    epoch = data.get('epoch', '?')
    best_dice = data.get('best_val_dice', '?')
    print(f"Loaded epoch {epoch}, best_val_dice={best_dice}")

    return model, config, data


# =============================================================================
# Sliding window inference (JAX)
# =============================================================================

def make_gaussian_importance_map(patch_size, sigma_scale=0.125):
    """3D Gaussian importance map (sigma = 1/8 patch size, matching nnUNet)."""
    coords = []
    for s in patch_size:
        sigma = s * sigma_scale
        c = np.arange(s, dtype=np.float32) - (s - 1) / 2.0
        g = np.exp(-0.5 * (c / max(sigma, 1e-6)) ** 2)
        coords.append(g)
    importance = coords[0][:, None, None] * coords[1][None, :, None] * coords[2][None, None, :]
    importance = importance / importance.max()
    importance = np.clip(importance, 1e-4, 1.0)
    return importance


def _compute_padding(dim, patch, step):
    """Compute reflect-padding for a perfectly regular patch grid."""
    if dim <= patch:
        return 0
    n_steps = -(-(dim - patch) // step)  # ceil division
    return patch + n_steps * step - dim


def _mm_to_voxels(patch_size_mm, spacing, img_shape):
    """Convert mm patch size to voxels, clamped to image size."""
    if isinstance(patch_size_mm, (int, float)):
        patch_size_mm = (float(patch_size_mm),) * 3
    return tuple(
        min(max(8, int(round(mm / sp))), s)
        for mm, sp, s in zip(patch_size_mm, spacing, img_shape)
    )


@jax.jit
def _forward_jit(graphdef, state, x):
    """JIT-compiled forward pass for inference (module-level for cache reuse)."""
    m = nnx.merge(graphdef, state)
    out = m(x, deterministic=True, use_running_average=True)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return jax.nn.softmax(out, axis=1)


def sliding_window_inference_jax(
    model, image, spacing, patch_size_mm=128.0, overlap=0.69,
    mirror_axes=None, sw_batch_size=4,
):
    """Batched sliding window inference with Gaussian weighting in JAX.

    Parameters
    ----------
    model : E3nnUNet (JAX)
        Model with spacing already set via update_spacing().
    image : np.ndarray
        (C, D, H, W) preprocessed image.
    spacing : tuple
        Voxel spacing in mm.
    patch_size_mm : float or tuple
        Patch size in mm.
    overlap : float
        Overlap fraction between patches.
    mirror_axes : tuple, optional
        Axes for test-time augmentation.
    sw_batch_size : int
        Number of patches per forward pass.

    Returns
    -------
    np.ndarray
        Softmax probabilities (C, D, H, W).
    """
    img_shape = image.shape[1:]  # (D, H, W)
    patch_voxels = _mm_to_voxels(patch_size_mm, spacing, img_shape)
    pd, ph, pw = patch_voxels
    D, H, W = img_shape

    # Compute step sizes and padding for regular grid
    step_d = max(1, int(pd * (1 - overlap)))
    step_h = max(1, int(ph * (1 - overlap)))
    step_w = max(1, int(pw * (1 - overlap)))

    total_pad_d = _compute_padding(D, pd, step_d)
    total_pad_h = _compute_padding(H, ph, step_h)
    total_pad_w = _compute_padding(W, pw, step_w)

    # Symmetric reflect-pad
    pad_d0, pad_d1 = total_pad_d // 2, total_pad_d - total_pad_d // 2
    pad_h0, pad_h1 = total_pad_h // 2, total_pad_h - total_pad_h // 2
    pad_w0, pad_w1 = total_pad_w // 2, total_pad_w - total_pad_w // 2

    if total_pad_d > 0 or total_pad_h > 0 or total_pad_w > 0:
        image = np.pad(
            image,
            ((0, 0), (pad_d0, pad_d1), (pad_h0, pad_h1), (pad_w0, pad_w1)),
            mode='reflect',
        )

    Dp, Hp, Wp = image.shape[1:]

    # Regular grid patch positions
    d_positions = list(range(0, Dp - pd + 1, step_d))
    h_positions = list(range(0, Hp - ph + 1, step_h))
    w_positions = list(range(0, Wp - pw + 1, step_w))

    patch_locs = [(d, h, w)
                  for d in d_positions for h in h_positions for w in w_positions]
    n_patches = len(patch_locs)

    # Gaussian importance map
    importance = make_gaussian_importance_map(patch_voxels)  # (pd, ph, pw)

    # TTA flip combos
    if mirror_axes and len(mirror_axes) > 0:
        spatial_dims = [a + 2 for a in mirror_axes]  # +2 for batch+channel dims
        n_combos = 1 << len(spatial_dims)
        flip_combos = []
        for combo in range(n_combos):
            dims = tuple(spatial_dims[i] for i in range(len(spatial_dims))
                         if combo & (1 << i))
            flip_combos.append(dims)
    else:
        flip_combos = [()]
        n_combos = 1

    # Split model once for this call
    graphdef, state = nnx.split(model)

    # Output buffers
    output = None
    weight_sum = np.zeros((1, Dp, Hp, Wp), dtype=np.float32)

    print(f"  Processing {n_patches} patches (patch_voxels={patch_voxels}, "
          f"sw_batch_size={sw_batch_size}, tta={n_combos}x)...")

    for batch_start in range(0, n_patches, sw_batch_size):
        batch_locs = patch_locs[batch_start:batch_start + sw_batch_size]

        # Extract batch of patches -> (B, C, pd, ph, pw)
        patches = np.stack([
            image[:, d:d+pd, h:h+ph, w:w+pw]
            for d, h, w in batch_locs
        ])
        patches_jax = jnp.array(patches, dtype=jnp.bfloat16)

        # TTA: average predictions over flip combos
        pred_sum = None
        for flip_dims in flip_combos:
            inp = jnp.flip(patches_jax, axis=flip_dims) if flip_dims else patches_jax
            pred = _forward_jit(graphdef, state, inp)
            if flip_dims:
                pred = jnp.flip(pred, axis=flip_dims)
            pred_sum = pred if pred_sum is None else pred_sum + pred

        pred_avg = np.array(pred_sum / n_combos, dtype=np.float32)

        if output is None:
            n_classes = pred_avg.shape[1]
            output = np.zeros((n_classes, Dp, Hp, Wp), dtype=np.float32)

        # Scatter-add weighted predictions
        for i, (d, h, w) in enumerate(batch_locs):
            output[:, d:d+pd, h:h+ph, w:w+pw] += pred_avg[i] * importance
            weight_sum[:, d:d+pd, h:h+ph, w:w+pw] += importance

    # Normalize and crop back to original size
    output = output / np.clip(weight_sum, 1e-8, None)
    output = output[:, pad_d0:pad_d0+D, pad_h0:pad_h0+H, pad_w0:pad_w0+W]
    return output  # (C, D, H, W)


# =============================================================================
# NIfTI saving
# =============================================================================

def save_prediction_nifti(pred_cropped, props, output_path):
    """Save a cropped prediction array as a NIfTI in original image space.

    Requires augmented pkl metadata (original_affine, original_nifti_shape).
    """
    original_affine = props['original_affine']
    original_shape = props['original_shape']
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_nii = nib.Nifti1Image(output_data, original_affine)
    out_nii.header.set_data_dtype(np.int16)
    nib.save(out_nii, str(output_path))


# =============================================================================
# Dice score
# =============================================================================

def dice_score(pred, target):
    """Compute foreground dice score."""
    pred_fg = (pred == 1).astype(float)
    tgt_fg = (target == 1).astype(float)
    intersection = (pred_fg * tgt_fg).sum()
    union = pred_fg.sum() + tgt_fg.sum()
    if union == 0:
        return 1.0 if tgt_fg.sum() == 0 else 0.0
    return (2.0 * intersection) / union


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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Full-volume validation inference for JAX E3nnUNet')
    parser.add_argument('experiment_dir', type=str,
                        help='Path to experiment directory')
    parser.add_argument('--checkpoint', type=str, default='model_best.pkl',
                        help='Checkpoint filename (default: model_best.pkl)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--overlap', type=float, default=0.69,
                        help='Overlap fraction for sliding window')
    parser.add_argument('--patch_size_mm', type=float, default=None,
                        help='Patch size in mm (default: from config)')
    parser.add_argument('--sw_batch_size', type=int, default=4,
                        help='Patches per forward pass')
    parser.add_argument('--mirror_tta', action='store_true',
                        help='Enable mirror TTA on all 3 axes')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save NIfTI prediction files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for predictions (default: experiment_dir/predictions)')
    parser.add_argument('--canonical_spacing', action='store_true', default=True,
                        help='Use canonical (grouped) spacing for model kernels (default: True)')
    parser.add_argument('--no_canonical_spacing', dest='canonical_spacing', action='store_false',
                        help='Use real per-case spacing instead of canonical')
    args = parser.parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    experiment_dir = Path(args.experiment_dir)

    # Load model
    model, config, ckpt_data = load_jax_model(experiment_dir, args.checkpoint)

    # Get validation cases
    preprocessed_dir = Path(config['data']['preprocessed_dir'])
    fold = config['data']['fold']
    splits_file = preprocessed_dir / 'splits_final.json'
    with open(splits_file) as f:
        splits = json.load(f)
    val_cases = splits[fold]['val']
    print(f"Fold {fold}: {len(val_cases)} validation cases")

    # Patch size from config or CLI
    patch_mm = args.patch_size_mm
    if patch_mm is None:
        patch_mm = config['training'].get('patch_size_mm', [128.0, 128.0, 128.0])
        if isinstance(patch_mm, list):
            patch_mm = patch_mm[0]  # Use isotropic
    print(f"Patch size: {patch_mm}mm, overlap: {args.overlap}")
    print(f"Canonical spacing: {args.canonical_spacing}")

    mirror_axes = (0, 1, 2) if args.mirror_tta else None

    # Output directory for predictions
    pred_dir = None
    if args.save_predictions:
        pred_dir = Path(args.output_dir) if args.output_dir else experiment_dir / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving predictions to: {pred_dir}")

    # ---- Load existing results for resume ----
    output_name = f"val_jax_{args.checkpoint.replace('.pkl', '')}"
    if args.mirror_tta:
        output_name += "_tta"
    output_path = experiment_dir / f"{output_name}.json"

    results = {}
    plane_stats = defaultdict(list)
    res_stats = defaultdict(list)
    if output_path.exists():
        with open(output_path) as f:
            prev = json.load(f)
        prev_cases = prev.get('per_case', {})
        for case, r in prev_cases.items():
            sp = tuple(r['spacing']) if isinstance(r['spacing'], list) else r['spacing']
            results[case] = {
                'dice': r['dice'],
                'spacing': sp,
                'plane': r['plane'],
                'resolution_group': tuple(r['resolution_group']) if isinstance(r['resolution_group'], list) else r['resolution_group'],
                'shape': r['shape'],
                'time': r['time'],
            }
            plane_stats[r['plane']].append(r['dice'])
            res_stats[results[case]['resolution_group']].append(r['dice'])
        print(f"Resuming: loaded {len(results)} previous results from {output_path}")

    # ---- Group cases by model spacing ----
    print("Grouping cases by spacing...")
    case_spacings = {}  # case -> (real_spacing, model_spacing)
    model_spacing_groups = defaultdict(list)  # model_spacing -> [case_names]

    for case in sorted(val_cases):
        if case in results:
            continue  # already have result
        img_path = preprocessed_dir / f"{case}.npy"
        pkl_path = preprocessed_dir / f"{case}.pkl"
        if not img_path.exists():
            print(f"  {case}: MISSING, skipping")
            continue
        with open(pkl_path, 'rb') as f:
            props = pickle.load(f)
        real_spacing = tuple(float(s) for s in props['spacing'])
        model_sp = round_spacing_to_tolerance(real_spacing) if args.canonical_spacing else real_spacing
        case_spacings[case] = (real_spacing, model_sp)
        model_spacing_groups[model_sp].append(case)

    # Sort groups: largest first (amortize compile cost over more cases)
    sorted_groups = sorted(model_spacing_groups.keys(),
                           key=lambda s: len(model_spacing_groups[s]), reverse=True)
    n_groups = len(sorted_groups)
    n_remaining = len(case_spacings)
    n_total = n_remaining + len(results)
    print(f"Spacing groups: {n_groups} ({n_remaining} remaining, "
          f"{len(results)} already done, {n_total} total)")
    for sp in sorted_groups:
        print(f"  {sp}: {len(model_spacing_groups[sp])} cases")

    # ---- Pre-snapshot graphdefs for background compilation ----
    print("Snapshotting graphdefs for background compilation...")
    snapshots = {}  # model_spacing -> (graphdef, state)
    for sp in sorted_groups:
        model.update_spacing(sp)
        gd, st = nnx.split(model)
        snapshots[sp] = (gd, st)

    def _bg_compile(model_sp):
        """Pre-compile forward pass for next spacing group (background thread)."""
        gd, st = snapshots[model_sp]
        # Use unclamped patch size (large img_shape avoids clamping)
        rep_patch = _mm_to_voxels(patch_mm, model_sp, (99999, 99999, 99999))
        # Skip bg compile for very large patches to avoid OOM during inference
        n_voxels = rep_patch[0] * rep_patch[1] * rep_patch[2]
        if n_voxels > 4_000_000:
            print(f"  [bg compile] {model_sp} skipped (patch {rep_patch} "
                  f"too large for bg compile)", flush=True)
            return
        dummy = jnp.zeros((args.sw_batch_size, 1, *rep_patch), dtype=jnp.bfloat16)
        try:
            _forward_jit.lower(gd, st, dummy).compile()
        except Exception as e:
            print(f"  [bg compile] {model_sp} failed: {e}", flush=True)

    # ---- Process groups ----
    total_t0 = time.time()
    case_idx = len(results)  # start counting from already-done cases
    bg_thread = None

    for gi, model_sp in enumerate(sorted_groups):
        cases_in_group = model_spacing_groups[model_sp]

        # Wait for any pending background compile
        if bg_thread is not None:
            bg_thread.join()
            bg_thread = None

        # Free snapshot for current group (no longer needed)
        snapshots.pop(model_sp, None)

        # Update model to this group's spacing
        model.update_spacing(model_sp)

        # Start background compile for NEXT group
        if gi + 1 < len(sorted_groups):
            next_sp = sorted_groups[gi + 1]
            bg_thread = threading.Thread(
                target=_bg_compile, args=(next_sp,), daemon=True)
            bg_thread.start()

        print(f"\nGroup {gi+1}/{n_groups}: {model_sp} "
              f"({len(cases_in_group)} cases)", flush=True)

        for case in cases_in_group:
            case_idx += 1
            case_t0 = time.time()

            image = np.load(preprocessed_dir / f"{case}.npy")
            label = np.load(preprocessed_dir / f"{case}_seg.npy")
            if label.ndim == 4:
                label = label[0]
            with open(preprocessed_dir / f"{case}.pkl", 'rb') as f:
                props = pickle.load(f)
            real_spacing = case_spacings[case][0]

            # Run sliding window (model already at correct spacing)
            probs = sliding_window_inference_jax(
                model, image, real_spacing,
                patch_size_mm=patch_mm,
                overlap=args.overlap,
                mirror_axes=mirror_axes,
                sw_batch_size=args.sw_batch_size,
            )

            pred = probs.argmax(axis=0).astype(np.int16)
            d = dice_score(pred, label)
            dt = time.time() - case_t0

            plane = classify_plane(real_spacing)
            res_group = quantize_spacing(real_spacing)

            print(f"  [{case_idx}/{n_total}] {case}: dice={d:.4f} "
                  f"sp={tuple(round(s,2) for s in real_spacing)} plane={plane} "
                  f"shape={image.shape[1:]} time={dt:.1f}s")

            results[case] = {
                'dice': d,
                'spacing': real_spacing,
                'plane': plane,
                'resolution_group': res_group,
                'shape': list(image.shape[1:]),
                'time': dt,
            }
            plane_stats[plane].append(d)
            res_stats[res_group].append(d)

            if pred_dir and 'original_affine' in props:
                save_prediction_nifti(pred, props, pred_dir / f"{case}.nii.gz")

        # Incremental save after each group (survives crashes)
        all_dice_so_far = [r['dice'] for r in results.values()]
        _inc_data = {
            'checkpoint': args.checkpoint,
            'epoch': ckpt_data.get('epoch', None),
            'overlap': args.overlap,
            'patch_size_mm': patch_mm,
            'mirror_tta': args.mirror_tta,
            'canonical_spacing': args.canonical_spacing,
            'n_spacing_groups': n_groups,
            'mean_dice': float(np.mean(all_dice_so_far)),
            'std_dice': float(np.std(all_dice_so_far)),
            'median_dice': float(np.median(all_dice_so_far)),
            'n_cases': len(results),
            'complete': False,
            'per_case': results,
        }
        with open(output_path, 'w') as f:
            json.dump(_inc_data, f, indent=2, default=str)

    # Wait for last background compile
    if bg_thread is not None:
        bg_thread.join()

    total_time = time.time() - total_t0

    # Summary
    print(f"\n{'='*60}")
    print(f"Validation complete: {len(results)} cases in {total_time:.0f}s")
    all_dice = [r['dice'] for r in results.values()]
    print(f"Mean dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"Median dice: {np.median(all_dice):.4f}")

    print(f"\nBy imaging plane:")
    for plane in sorted(plane_stats):
        vals = plane_stats[plane]
        print(f"  {plane:4s}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

    print(f"\nBy resolution group (in-plane, slice):")
    for res in sorted(res_stats):
        vals = res_stats[res]
        print(f"  {str(res):20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

    # Save results JSON (output_path computed at top for resume)
    save_data = {
        'checkpoint': args.checkpoint,
        'epoch': ckpt_data.get('epoch', None),
        'overlap': args.overlap,
        'patch_size_mm': patch_mm,
        'mirror_tta': args.mirror_tta,
        'canonical_spacing': args.canonical_spacing,
        'n_spacing_groups': n_groups,
        'mean_dice': float(np.mean(all_dice)),
        'std_dice': float(np.std(all_dice)),
        'median_dice': float(np.median(all_dice)),
        'n_cases': len(results),
        'total_time': total_time,
        'complete': True,
        'per_case': results,
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
