#!/usr/bin/env python
"""
Validate project_to_spacing() against the original e3nn model.

For each validation case:
1. Load preprocessed data (.npy) and metadata (.pkl)
2. Run sliding-window inference with the original e3nn model
3. Run sliding-window inference with the projected (distilled) model
4. Compare Dice scores and measure per-case inference time

Usage:
    python validate_projection.py \
        --experiment_dir ./experiments/plan_80mm_maxip1.1 \
        --gpu 1
"""

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from irrepunet.models import E3nnUNet, project_to_spacing


def compute_dice(pred: np.ndarray, gt: np.ndarray, n_classes: int = 2) -> float:
    """Compute mean foreground Dice score."""
    dice_scores = []
    for c in range(1, n_classes):  # skip background
        pred_c = (pred == c)
        gt_c = (gt == c)
        intersection = (pred_c & gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append(2.0 * intersection / union)
    return float(np.mean(dice_scores))


def sliding_window_inference(
    image: torch.Tensor,
    model: torch.nn.Module,
    patch_size: tuple,
    spacing: tuple = None,
    overlap: float = 0.5,
    device: torch.device = torch.device('cuda'),
    use_amp: bool = True,
) -> torch.Tensor:
    """Sliding window inference.

    If spacing is None, model.forward(patch) is called (projected model).
    If spacing is given, model.forward(patch, spacing=spacing) is called (e3nn model).
    """
    model.eval()

    _, _, D, H, W = image.shape
    pd, ph, pw = patch_size

    step_d = max(1, int(pd * (1 - overlap)))
    step_h = max(1, int(ph * (1 - overlap)))
    step_w = max(1, int(pw * (1 - overlap)))

    output = None
    counts = torch.zeros(1, 1, D, H, W, device=device)

    d_positions = list(range(0, max(1, D - pd + 1), step_d))
    h_positions = list(range(0, max(1, H - ph + 1), step_h))
    w_positions = list(range(0, max(1, W - pw + 1), step_w))

    if d_positions[-1] + pd < D:
        d_positions.append(D - pd)
    if h_positions[-1] + ph < H:
        h_positions.append(H - ph)
    if w_positions[-1] + pw < W:
        w_positions.append(W - pw)

    with torch.no_grad():
        for d in d_positions:
            for h in h_positions:
                for w in w_positions:
                    patch = image[:, :, d:d+pd, h:h+ph, w:w+pw].to(device)

                    actual_shape = patch.shape[2:]
                    if actual_shape != (pd, ph, pw):
                        pad = [
                            0, pw - actual_shape[2],
                            0, ph - actual_shape[1],
                            0, pd - actual_shape[0]
                        ]
                        patch = F.pad(patch, pad)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        if spacing is not None:
                            pred = model(patch, spacing=spacing)
                        else:
                            pred = model(patch)

                    # Handle deep supervision (take finest level)
                    if isinstance(pred, list):
                        pred = pred[-1]

                    pred = F.softmax(pred, dim=1)

                    if actual_shape != (pd, ph, pw):
                        pred = pred[:, :, :actual_shape[0], :actual_shape[1], :actual_shape[2]]

                    if output is None:
                        n_classes = pred.shape[1]
                        output = torch.zeros(1, n_classes, D, H, W, device=device)

                    output[:, :, d:d+actual_shape[0], h:h+actual_shape[1], w:w+actual_shape[2]] += pred
                    counts[:, :, d:d+actual_shape[0], h:h+actual_shape[1], w:w+actual_shape[2]] += 1

    output = output / counts.clamp(min=1)
    return output


def main():
    parser = argparse.ArgumentParser(description='Validate projection consistency')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='model_best.pt')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--patch_size_mm', type=float, default=80.0,
                        help='Patch size in mm (cubic)')
    parser.add_argument('--max_cases', type=int, default=None,
                        help='Limit number of cases (for quick testing)')
    parser.add_argument('--use_2d', action='store_true', default=True,
                        help='Use 2D conv optimization for anisotropic kernels')
    parser.add_argument('--no_use_2d', action='store_true',
                        help='Disable 2D conv optimization')
    args = parser.parse_args()

    use_2d = not args.no_use_2d

    experiment_dir = Path(args.experiment_dir)
    config_path = experiment_dir / 'config.json'

    with open(config_path) as f:
        config = json.load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ========================================================================
    # Load model
    # ========================================================================
    preprocessed_dir = Path(config['preprocessed_dir'])
    fold = config.get('fold', 0)

    with open(preprocessed_dir / 'splits_final.json') as f:
        splits = json.load(f)
    val_cases = splits[fold]['val']

    if args.max_cases:
        val_cases = val_cases[:args.max_cases]

    n_classes = config.get('n_classes', 2)

    # Determine model class and params
    model_kwargs = dict(
        n_classes=n_classes,
        in_channels=config.get('in_channels', 1),
        diameter=config.get('diameter', 5.0),
        num_radial_basis=config.get('num_radial_basis', 5),
        spacing=(1.0, 1.0, 1.0),  # Will be rebuilt per-case
        normalization=config.get('normalization', 'instance'),
        n_base_filters=config.get('n_base_filters', 2),
        n_downsample=config.get('n_downsample', 4),
        equivariance=config.get('equivariance', 'SO3'),
        lmax=config.get('lmax', 2),
        dropout_prob=0.0,
        cutoff=True,
        deep_supervision=config.get('deep_supervision', False),
        max_features=config.get('max_features', 320),
        irrep_ratios=tuple(config.get('irrep_ratios', [4, 2, 1])),
        fill_to_max=config.get('fill_to_max', False),
    )

    model = E3nnUNet(**model_kwargs)

    checkpoint_path = experiment_dir / args.checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint['model_state_dict']
    # Filter spacing-dependent buffers (will be regenerated)
    filtered_state = {k: v for k, v in state_dict.items()
                      if not any(x in k for x in ['lattice', '.emb', '.sh'])}
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    val_dice = checkpoint.get('val_dice', '?')
    print(f"Checkpoint epoch: {epoch}, val_dice: {val_dice}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"use_2d optimization: {use_2d}")
    print()

    # ========================================================================
    # Group cases by spacing for efficient projection
    # ========================================================================
    spacing_groups = defaultdict(list)
    case_metadata = {}

    for case_name in val_cases:
        with open(preprocessed_dir / f"{case_name}.pkl", 'rb') as f:
            props = pickle.load(f)
        spacing = tuple(float(s) for s in props['spacing'])
        spacing_groups[spacing].append(case_name)
        case_metadata[case_name] = props

    print(f"Unique spacings: {len(spacing_groups)}")
    print()

    # ========================================================================
    # Run inference: e3nn vs projected
    # ========================================================================
    results = []
    total_time_e3nn = 0.0
    total_time_proj = 0.0
    total_patches = 0
    case_idx = 0

    for spacing, cases in sorted(spacing_groups.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{'='*70}")
        print(f"Spacing: {spacing} ({len(cases)} cases)")
        print(f"{'='*70}")

        # Project model to this spacing (once per group)
        t0 = time.time()
        projected = project_to_spacing(model, spacing, use_2d=use_2d)
        projected.to(device)
        projected.eval()
        proj_time = time.time() - t0
        print(f"  Projection time: {proj_time:.2f}s")

        # Compute patch size in voxels
        spacing_arr = np.array(spacing)
        patch_voxels = np.round(args.patch_size_mm / spacing_arr).astype(int)

        # Ensure divisible by pooling factor
        pooling_factor = config.get('pooling_factor', 8)
        patch_voxels = tuple(
            max(pooling_factor, int(v) // pooling_factor * pooling_factor)
            for v in patch_voxels
        )
        print(f"  Patch size (voxels): {patch_voxels}")

        for case_name in cases:
            case_idx += 1
            print(f"\n  [{case_idx}/{len(val_cases)}] {case_name}")

            # Load data
            image = np.load(preprocessed_dir / f"{case_name}.npy")  # (C, D, H, W)
            seg = np.load(preprocessed_dir / f"{case_name}_seg.npy")  # (D, H, W)
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, C, D, H, W)

            print(f"    Shape: {image.shape[1:]}, spacing: {spacing}")

            # Clamp patch size to image size
            ps = tuple(min(p, s) for p, s in zip(patch_voxels, image.shape[1:]))

            # --- e3nn model ---
            torch.cuda.synchronize(device)
            t_start = time.time()

            pred_e3nn = sliding_window_inference(
                image_tensor, model, ps, spacing=spacing,
                overlap=args.overlap, device=device
            )

            torch.cuda.synchronize(device)
            t_e3nn = time.time() - t_start

            pred_e3nn_np = pred_e3nn.cpu().numpy()[0].argmax(axis=0).astype(np.int16)
            dice_e3nn = compute_dice(pred_e3nn_np, seg, n_classes)

            del pred_e3nn
            torch.cuda.empty_cache()

            # --- projected model ---
            torch.cuda.synchronize(device)
            t_start = time.time()

            pred_proj = sliding_window_inference(
                image_tensor, projected, ps, spacing=None,
                overlap=args.overlap, device=device
            )

            torch.cuda.synchronize(device)
            t_proj = time.time() - t_start

            pred_proj_np = pred_proj.cpu().numpy()[0].argmax(axis=0).astype(np.int16)
            dice_proj = compute_dice(pred_proj_np, seg, n_classes)

            # Compare outputs
            match = np.array_equal(pred_e3nn_np, pred_proj_np)
            voxel_diff = (pred_e3nn_np != pred_proj_np).sum()
            total_voxels = pred_e3nn_np.size

            del pred_proj
            torch.cuda.empty_cache()

            speedup = t_e3nn / max(t_proj, 1e-6)

            print(f"    e3nn:  Dice={dice_e3nn:.4f}  time={t_e3nn:.2f}s")
            print(f"    proj:  Dice={dice_proj:.4f}  time={t_proj:.2f}s  speedup={speedup:.2f}x")
            print(f"    Match: {match}  diff_voxels={voxel_diff}/{total_voxels} ({100*voxel_diff/total_voxels:.4f}%)")

            total_time_e3nn += t_e3nn
            total_time_proj += t_proj

            results.append({
                'case': case_name,
                'spacing': list(spacing),
                'shape': list(image.shape[1:]),
                'dice_e3nn': dice_e3nn,
                'dice_proj': dice_proj,
                'dice_diff': abs(dice_e3nn - dice_proj),
                'time_e3nn': t_e3nn,
                'time_proj': t_proj,
                'speedup': speedup,
                'match': match,
                'diff_voxels': int(voxel_diff),
                'total_voxels': int(total_voxels),
            })

        # Free projected model for this spacing group
        del projected
        torch.cuda.empty_cache()

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY ({len(results)} cases)")
    print(f"{'='*70}")

    dice_e3nn_arr = np.array([r['dice_e3nn'] for r in results])
    dice_proj_arr = np.array([r['dice_proj'] for r in results])
    dice_diff_arr = np.array([r['dice_diff'] for r in results])
    speedup_arr = np.array([r['speedup'] for r in results])
    match_arr = np.array([r['match'] for r in results])

    print(f"\nDice (e3nn):     mean={dice_e3nn_arr.mean():.4f}  median={np.median(dice_e3nn_arr):.4f}  std={dice_e3nn_arr.std():.4f}")
    print(f"Dice (projected): mean={dice_proj_arr.mean():.4f}  median={np.median(dice_proj_arr):.4f}  std={dice_proj_arr.std():.4f}")
    print(f"Dice diff:        mean={dice_diff_arr.mean():.6f}  max={dice_diff_arr.max():.6f}")
    print(f"\nExact match: {match_arr.sum()}/{len(match_arr)} cases ({100*match_arr.mean():.1f}%)")
    print(f"\nTotal time (e3nn):      {total_time_e3nn:.1f}s")
    print(f"Total time (projected): {total_time_proj:.1f}s")
    print(f"Overall speedup:        {total_time_e3nn/max(total_time_proj, 1e-6):.2f}x")
    print(f"Speedup per case:       mean={speedup_arr.mean():.2f}x  median={np.median(speedup_arr):.2f}x")

    # Save results
    output = {
        'experiment': str(experiment_dir),
        'checkpoint_epoch': epoch,
        'n_cases': len(results),
        'use_2d': use_2d,
        'patch_size_mm': args.patch_size_mm,
        'overlap': args.overlap,
        'summary': {
            'dice_e3nn_mean': float(dice_e3nn_arr.mean()),
            'dice_e3nn_median': float(np.median(dice_e3nn_arr)),
            'dice_proj_mean': float(dice_proj_arr.mean()),
            'dice_proj_median': float(np.median(dice_proj_arr)),
            'dice_diff_mean': float(dice_diff_arr.mean()),
            'dice_diff_max': float(dice_diff_arr.max()),
            'exact_match_pct': float(100 * match_arr.mean()),
            'total_time_e3nn': total_time_e3nn,
            'total_time_proj': total_time_proj,
            'overall_speedup': total_time_e3nn / max(total_time_proj, 1e-6),
            'speedup_mean': float(speedup_arr.mean()),
            'speedup_median': float(np.median(speedup_arr)),
        },
        'per_case': results,
    }

    output_path = experiment_dir / 'val_projection_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
