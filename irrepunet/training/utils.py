"""Shared training utilities for e3nnUNet.

Framework-agnostic functions used by both train.py (PyTorch) and train_jax.py (JAX).
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from irrepunet.data.multi_resolution_loader import verify_receptive_field
from irrepunet.models import compute_kernel_sizes


def load_experiment_config(config_path: Path) -> Dict:
    """Load experiment configuration from config.json."""
    with open(config_path) as f:
        return json.load(f)


def args_from_config(config: Dict, config_path: Path = None, cli_resume: bool = False) -> argparse.Namespace:
    """Convert config dict to argparse.Namespace.

    Args:
        config: Configuration dictionary from config.json
        config_path: Path to config.json (used to derive output_dir if not in config)
        cli_resume: Whether --resume was passed on the CLI

    Returns:
        argparse.Namespace with all training arguments
    """
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
    args_dict['bottleneck_kernel'] = model.get('bottleneck_kernel', 0)

    # Training arguments
    training = config['training']
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
    args_dict['targeted_pooling'] = training.get('targeted_pooling', False)
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
    args_dict['curriculum'] = aug.get('curriculum', None)
    args_dict['curriculum_bs_tiers'] = aug.get('curriculum_bs_tiers', None)
    args_dict['curriculum_phase_len'] = aug.get('curriculum_phase_len', 30)

    # Hardware arguments
    hw = config['hardware']
    args_dict['gpu'] = hw['gpu']
    args_dict['fp16'] = hw.get('fp16', True)
    args_dict['deep_supervision'] = hw.get('deep_supervision', True)
    args_dict['no_background_dice'] = hw.get('no_background_dice', False)
    args_dict['batch_dice'] = hw.get('batch_dice', False)

    # Resume flag
    args_dict['resume'] = cli_resume
    args_dict['plan_only'] = False
    args_dict['init_checkpoint'] = training.get('init_checkpoint', config.get('init_checkpoint', None))

    # W&B arguments
    args_dict['wandb'] = training.get('wandb', False)
    args_dict['wandb_project'] = training.get('wandb_project', 'irrepunet')
    args_dict['wandb_name'] = training.get('wandb_name', None)

    return argparse.Namespace(**args_dict)


def _extract_resolutions(resolutions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract in-plane and slice thickness from resolution entries."""
    in_plane_res = []
    slice_thickness = []

    for entry in resolutions:
        spacing = entry['spacing']
        batch_size = entry['batch_size']
        sorted_spacing = tuple(sorted(spacing))
        ip = sorted_spacing[0]
        st = sorted_spacing[2]

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
    """Plot 2D density of training and validation samples by resolution."""
    if not train_resolutions and not val_resolutions:
        return

    train_ip, train_st = _extract_resolutions(train_resolutions) if train_resolutions else (np.array([]), np.array([]))

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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_train, ax_val = axes[0]
    ax_ip, ax_st = axes[1]

    xlim = (0, 2.5)
    ylim = (0.4, 15)

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

    plt.savefig(output_dir / 'resolution.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_progress(train_losses, val_losses, pseudo_dice, ema_pseudo_dice,
                  lrs, epoch_times, output_dir):
    """Generate nnUNet-style 3-panel progress plot."""
    if not train_losses:
        return

    epochs = list(range(len(train_losses)))

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

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

    ax2 = axes[1]
    ax2.plot(epochs, epoch_times, 'b-', linewidth=1.0)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Time [s]')
    ax2.set_title('Epoch Duration')

    ax3 = axes[2]
    ax3.plot(epochs, lrs, 'b-', linewidth=1.0)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'progress.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def _write_plan_validation_log(args, output_dir: Path, config_hash: str = None):
    """Compare runtime args against planned config.json and log results."""
    config_path = output_dir / 'config.json'
    if not config_path.exists():
        log_lines = [
            f"Plan validation: {datetime.now().isoformat()}",
            "No config.json found (legacy mode) — skipping validation.",
        ]
        (output_dir / 'plan_validation.log').write_text('\n'.join(log_lines))
        return

    with open(config_path) as f:
        config = json.load(f)

    checks = [
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
        ('training', 'epochs', 'epochs', True),
        ('training', 'learning_rate', 'lr', True),
        ('training', 'weight_decay', 'weight_decay', True),
        ('training', 'grad_clip', 'grad_clip', True),
        ('training', 'patch_size_mm', 'patch_size_mm', True),
        ('training', 'patches_per_epoch', 'patches_per_epoch', True),
        ('training', 'dynamic_batch_size', 'dynamic_batch_size', True),
        ('training', 'target_memory_mb', 'target_memory_mb', True),
        ('hardware', 'fp16', 'fp16', True),
        ('hardware', 'deep_supervision', 'deep_supervision', True),
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


def write_loader_config(filepath, args, groups, n_train_cases, n_val_cases, model_scale=2.0,
                        curriculum_phases=None):
    """Write unified loader_config.txt used by both --plan_only and training."""
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT CONFIGURATION")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    lines.append("DATA")
    lines.append("-" * 80)
    lines.append(f"Fold: {args.fold}")
    lines.append(f"Training cases: {n_train_cases}")
    lines.append(f"Validation cases: {n_val_cases}")
    lines.append(f"Preprocessed dir: {args.preprocessed_dir}")
    lines.append("")

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

    if getattr(args, 'subsample_weight', 0) > 0:
        lines.append("AUGMENTATION")
        lines.append("-" * 80)
        lines.append(f"Subsampled data weight: {args.subsample_weight}")
        lines.append("")

    sorted_groups = sorted(groups, key=lambda g: g['n_cases'], reverse=True)

    lines.append("RESOLUTION GROUPS")
    lines.append("-" * 120)

    min_batch = getattr(args, 'min_batch_size', 1)
    lines.append(f"{'Spacing (mm)':<22} {'Patch (voxels)':<18} {'Patch (mm)':<16} {'Batch':<6} {'Split':<6} {'Accum':<6} {'Eff.BS':<7} {'Cases':<7} {'Type':<12} {'Memory':<10} {'RF Error (mm)':<20}")
    lines.append("-" * 156)

    shrunken = []  # (spacing_str, effective_mm, shortfall)
    SHRINK_TOL_MM = 4.0

    for group in sorted_groups:
        raw_spacing = group['spacing']

        if isinstance(raw_spacing, (list, tuple)) and len(raw_spacing) == 3 and raw_spacing[0] == 'superres':
            sub_sp = tuple(float(s) for s in raw_spacing[1])
            orig_sp = tuple(float(s) for s in raw_spacing[2])
            spacing_str = f"SR {sub_sp[0]:.2f},{sub_sp[1]:.2f},{sub_sp[2]:.2f}"
            spacing = sub_sp
        else:
            spacing = tuple(float(s) for s in raw_spacing)
            spacing_str = f"({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f})"

        patch_voxels = tuple(group['patch_size_voxels'])
        patch_str = f"{patch_voxels[0]}x{patch_voxels[1]}x{patch_voxels[2]}"

        n_splits = group.get('n_spatial_splits', 1)
        bs = group['batch_size']
        accum = math.ceil(min_batch / bs) if bs < min_batch else 1
        eff_bs = bs * accum

        rf_info = verify_receptive_field(
            patch_voxels, spacing, patch_mm,
            args.n_downsample, model_scale
        )
        error = rf_info['error']
        rf_err_str = f"({error[0]:.1f}, {error[1]:.1f}, {error[2]:.1f})"

        group_type = group.get('group_type', 'real')

        if 'measured_memory_bs1' in group:
            mem_str = f"{group['measured_memory_bs1']:.0f} MB*"
        else:
            mem_str = f"{group['estimated_memory_mb']:.0f} MB"

        # Effective physical patch size and shrink detection
        effective_mm = tuple(v * s for v, s in zip(patch_voxels, spacing))
        shortfall = tuple(max(0.0, r - e) for r, e in zip(patch_mm, effective_mm))
        is_shrunken = any(sh > SHRINK_TOL_MM for sh in shortfall)
        patch_mm_str = f"{effective_mm[0]:.0f}x{effective_mm[1]:.0f}x{effective_mm[2]:.0f}"
        if is_shrunken:
            patch_mm_str = patch_mm_str + "!"
            shrunken.append((spacing_str, effective_mm, shortfall))

        lines.append(f"{spacing_str:<22} {patch_str:<18} {patch_mm_str:<16} {bs:<6} {n_splits:<6} {accum:<6} {eff_bs:<7} {group['n_cases']:<7} {group_type:<12} {mem_str:<10} {rf_err_str:<20}")

    has_measured = any('measured_memory_bs1' in g for g in groups)
    if has_measured:
        lines.append("")
        lines.append("* Memory values marked with * are directly measured on GPU (bs=1).")
        lines.append("  Batch sizes are computed from measured per-item memory cost.")

    if shrunken:
        lines.append("")
        lines.append(f"! Patch (mm) columns marked with '!' have an effective patch")
        lines.append(f"  smaller than the requested {patch_mm} mm.  This happens")
        lines.append(f"  when the request can't be divided by the cumulative pool")
        lines.append(f"  factor; mm_to_voxels floors down silently.  Affected groups:")
        lines.append(f"    {'spacing':<24} {'effective (mm)':<22} {'shortfall (mm)'}")
        for sp_str, eff_mm, shortfall in shrunken:
            eff_s = f"({eff_mm[0]:.0f}, {eff_mm[1]:.0f}, {eff_mm[2]:.0f})"
            sh_s = f"({shortfall[0]:.0f}, {shortfall[1]:.0f}, {shortfall[2]:.0f})"
            lines.append(f"    {sp_str:<24} {eff_s:<22} {sh_s}")

    lines.append("")

    scale = getattr(args, 'scale', 2.0)
    trim_th = getattr(args, 'kernel_trim_threshold', 1.0)
    kg = getattr(args, 'kernel_growth', 2.0)
    n_down = args.n_downsample
    diam = args.diameter
    scales_list = [scale * (2 ** i) for i in range(n_down)]

    unique_spacings = []
    for group in sorted_groups:
        raw_spacing = group['spacing']
        if isinstance(raw_spacing, (list, tuple)) and len(raw_spacing) == 3 and raw_spacing[0] == 'superres':
            continue
        sp = tuple(float(s) for s in raw_spacing)
        if sp not in unique_spacings:
            unique_spacings.append(sp)

    if unique_spacings:
        lines.append("KERNEL SIZE TABLE")
        lines.append("-" * 120)
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

    lines.append("HARDWARE")
    lines.append("-" * 80)
    lines.append(f"GPU: {args.gpu}")
    lines.append(f"Mixed precision (BF16): {args.fp16}")
    lines.append(f"Deep supervision: {getattr(args, 'deep_supervision', False)}")
    lines.append("")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
