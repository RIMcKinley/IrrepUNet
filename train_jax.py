#!/usr/bin/env python
"""
JAX training script for e3nnUNet.

Feature parity with train.py (PyTorch), using JAX/Flax NNX backend.
Reuses the same data pipeline, multi-resolution loader, transforms, and
experiment configuration utilities.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# JAX imports
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# Framework-agnostic utilities
from irrepunet.training.utils import (
    load_experiment_config,
    args_from_config,
    write_loader_config,
    plot_progress,
    plot_resolution_density,
    _write_plan_validation_log,
)
from irrepunet.data.spacing import group_cases_by_spacing
from irrepunet.data.dataloader import MultiResolutionLoader
from irrepunet.data.multi_resolution_loader import (
    estimate_memory_mb, estimate_batch_size,
    discover_skip_files, verify_receptive_field,
    compute_steps_through_pooling, mm_to_voxels,
)
from irrepunet.data.jax_adapter import get_training_transforms_jax, get_validation_transforms_jax

# JAX model and training
from irrepunet.models_jax import E3nnUNet, create_model
from irrepunet.models_jax.train import (
    configure_memory_optimizations,
    dice_loss, cross_entropy_loss, dice_ce_loss, deep_supervision_loss,
    create_jitted_train_step_dynamic,
    create_jitted_val_step,
    accumulated_train_step,
    BackgroundJITCompiler,
    _get_rss_mb,
)
from irrepunet.models_jax.split_jit import SplitJITManager, create_split_val_step


# =============================================================================
# Model construction
# =============================================================================

def build_model(args, n_classes, ref_spacing):
    """Build JAX E3nnUNet from training arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Training arguments (from CLI or config.json).
    n_classes : int
        Number of segmentation classes.
    ref_spacing : tuple
        Reference spacing for initial model construction.

    Returns
    -------
    E3nnUNet
        Initialized JAX model.
    """
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
    return model


# =============================================================================
# Optimizer
# =============================================================================

def build_optimizer(model, args, steps_per_epoch):
    """Build optimizer with gradient clipping and dynamic LR.

    Uses inject_hyperparams so the learning rate is a traced JAX array
    in the optimizer state, not a static schedule baked into the XLA graph.
    This allows the same compiled program to be reused across different
    schedule parameters (persistent cache survives schedule changes).

    The caller is responsible for updating the LR each step via
    set_optimizer_lr().

    Parameters
    ----------
    model : E3nnUNet
        The model to optimize.
    args : argparse.Namespace
        Training arguments with lr, weight_decay, grad_clip, epochs.
    steps_per_epoch : int
        Estimated optimizer steps per epoch (patches_per_epoch / avg_batch_size).

    Returns
    -------
    tuple[nnx.Optimizer, optax.Schedule]
        Configured optimizer and the LR schedule function (to be called
        externally to compute the LR at each step).
    """
    total_steps = args.epochs * steps_per_epoch
    schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=total_steps,
        alpha=1e-7 / max(args.lr, 1e-10),  # match CosineAnnealingLR eta_min=1e-7
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=args.lr, weight_decay=args.weight_decay,
        ),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return optimizer, schedule


def set_optimizer_lr(optimizer, lr):
    """Update the optimizer's learning rate in-place.

    Works with inject_hyperparams: the LR is stored as a JAX array
    in opt_state[1].hyperparams['learning_rate'].
    """
    optimizer.opt_state[1].hyperparams['learning_rate'] = lr


# =============================================================================
# Checkpointing (pickle-based)
# =============================================================================

def save_checkpoint(path, model, optimizer, epoch, best_val_dice, logging_dict,
                    config_hash):
    """Save checkpoint using pickle.

    Serializes model and optimizer state as NumPy arrays (host memory)
    for portable storage.

    Parameters
    ----------
    path : str or Path
        Output file path.
    model : E3nnUNet
    optimizer : nnx.Optimizer
    epoch : int
    best_val_dice : float
    logging_dict : dict
    config_hash : str
    """
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)
    checkpoint = {
        'epoch': epoch,
        'model_state': jax.tree.map(np.array, model_state),
        'optimizer_state': jax.tree.map(np.array, opt_state),
        'best_val_dice': best_val_dice,
        'logging': logging_dict,
        'config_hash': config_hash,
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path, model, optimizer):
    """Load checkpoint and restore model/optimizer state.

    Parameters
    ----------
    path : str or Path
        Path to checkpoint .pkl file.
    model : E3nnUNet
    optimizer : nnx.Optimizer

    Returns
    -------
    dict
        Checkpoint metadata (epoch, best_val_dice, logging, config_hash).
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    model_state = jax.tree.map(jnp.array, data['model_state'])
    nnx.update(model, model_state)

    opt_state = jax.tree.map(jnp.array, data['optimizer_state'])
    nnx.update(optimizer, opt_state)

    return data


# =============================================================================
# Validation
# =============================================================================

def validate_epoch(model, val_loader, n_classes, val_patches, fp16,
                   deep_supervision, no_background=False, val_step_fn=None,
                   split_val_step_fn=None, pool_kernel_overrides=None):
    """Run validation epoch, compute pseudo-dice via TP/FP/FN accumulation.

    Mimics the nnUNet-style global dice calculation: accumulate TP, FP, FN
    across all batches, then compute dice at the end.

    Parameters
    ----------
    model : E3nnUNet
    val_loader : MultiResolutionLoader
    n_classes : int
    val_patches : int
        Maximum number of validation patches to process.
    fp16 : bool
        If True, cast input to bfloat16.
    deep_supervision : bool
        If True, model returns list of multi-scale outputs.
    no_background : bool
        If True, skip background in dice computation.
    val_step_fn : callable or None
        JIT-compiled validation step from create_jitted_val_step().
        If None, falls back to eager model forward pass.
    split_val_step_fn : callable or None
        Split JIT validation step from create_split_val_step().
        Compiles lazily on first call. Falls back to monolithic on OOM.

    Returns
    -------
    dict
        val_loss, pseudo_dice, dice_per_class, n_patches, val_resolutions,
        val_time.
    """
    val_start = time.time()

    # Per-class TP, FP, FN (excluding background)
    tp_hard = [0] * (n_classes - 1)
    fp_hard = [0] * (n_classes - 1)
    fn_hard = [0] * (n_classes - 1)
    total_loss = 0.0
    n_batches = 0
    n_patches_done = 0
    n_split = 0
    n_mono = 0
    val_resolutions = []

    while n_patches_done < val_patches:
        try:
            batch, spacing = next(val_loader)
        except StopIteration:
            break

        # Use canonical group spacing for model (bounds JIT traces)
        group_sp = batch.get('group_spacing')
        if group_sp is not None:
            spacing = tuple(float(s) for s in group_sp)
        else:
            spacing = tuple(float(s) for s in spacing)
        sp_overrides = pool_kernel_overrides.get(spacing) if pool_kernel_overrides else None
        model.update_spacing(spacing, override_pool_kernels=sp_overrides)

        # Convert NumPy -> JAX in main process (workers keep data as NumPy)
        image = jnp.array(batch['image'])
        label_np = batch['label']  # keep NumPy copy for metrics
        label = jnp.array(label_np)  # (B, D, H, W), int32
        batch_size = image.shape[0]

        if fp16:
            image = image.astype(jnp.bfloat16)

        # Choose validation path: split JIT > monolithic JIT > eager
        # Split programs compile lazily via @jax.jit on first call.
        # If split OOMs (too many cached programs), fall back to monolithic.
        used_split = False
        if split_val_step_fn is not None:
            try:
                loss, pred_jax = split_val_step_fn(image, label)
                total_loss += float(loss)
                n_batches += 1
                pred_np = np.array(pred_jax)  # (B, D, H, W), int32
                used_split = True
                n_split += 1
            except Exception:
                pass  # fall through to monolithic

        if not used_split:
            if val_step_fn is not None:
                # Monolithic JIT-compiled validation path
                loss, pred_jax = val_step_fn(image, label)
                total_loss += float(loss)
                n_batches += 1
                pred_np = np.array(pred_jax)  # (B, D, H, W), int32
                n_mono += 1
            else:
                # Eager fallback
                logits = model(image, deterministic=True, use_running_average=True)

                if deep_supervision and isinstance(logits, list):
                    loss = deep_supervision_loss(logits, label, n_classes=n_classes,
                                                 no_background=no_background)
                    logits_finest = logits[-1]
                else:
                    if isinstance(logits, list):
                        logits = logits[-1]
                    loss = dice_ce_loss(logits, label, n_classes, no_background=no_background)
                    logits_finest = logits

                total_loss += float(loss)
                n_batches += 1
                pred_np = np.array(jnp.argmax(logits_finest, axis=1))  # (B, D, H, W)

        for b in range(batch_size):
            patch_dice_scores = []
            for c in range(1, n_classes):
                pred_c = (pred_np[b] == c)
                label_c = (label_np[b] == c)
                tp_b = int(np.sum(pred_c & label_c))
                fp_b = int(np.sum(pred_c & ~label_c))
                fn_b = int(np.sum(~pred_c & label_c))
                tp_hard[c - 1] += tp_b
                fp_hard[c - 1] += fp_b
                fn_hard[c - 1] += fn_b
                denom = 2 * tp_b + fp_b + fn_b
                patch_dice_scores.append(2 * tp_b / denom if denom > 0 else 0.0)

            val_resolutions.append({
                'spacing': spacing,
                'dice': float(np.mean(patch_dice_scores)),
            })

        n_patches_done += batch_size

    # Compute global dice per class: 2*TP / (2*TP + FP + FN)
    dice_per_class = []
    for c in range(n_classes - 1):
        denom = 2 * tp_hard[c] + fp_hard[c] + fn_hard[c]
        if denom > 0:
            dice_per_class.append(2 * tp_hard[c] / denom)

    # Mean foreground dice (like nnUNet's mean_fg_dice)
    mean_dice = float(np.mean(dice_per_class)) if dice_per_class else 0.0

    val_time = time.time() - val_start
    if n_split or n_mono:
        print(f"  Val path: {n_split} split, {n_mono} mono, {n_batches - n_split - n_mono} eager")

    return {
        'val_loss': total_loss / max(n_batches, 1),
        'pseudo_dice': mean_dice,
        'dice_per_class': dice_per_class,
        'n_patches': n_patches_done,
        'val_resolutions': val_resolutions,
        'val_time': val_time,
    }


# =============================================================================
# Training loop
# =============================================================================

def train(args, config_hash=None, planned_batch_sizes=None,
          planned_val_batch_sizes=None):
    """Main training function.

    Parameters
    ----------
    args : argparse.Namespace
        Training arguments.
    config_hash : str, optional
        SHA256 hash of config.json for integrity checking.
    planned_batch_sizes : dict, optional
        Mapping spacing -> batch_size from experiment planner.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B (optional)
    wandb_run = None
    if getattr(args, 'wandb', False):
        try:
            import wandb
            run_name = getattr(args, 'wandb_name', None) or output_dir.name
            wandb_run = wandb.init(
                project=getattr(args, 'wandb_project', 'irrepunet-jax'),
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
    splits_path = preprocessed_dir / 'splits_final.json'
    if not splits_path.exists():
        # Fallback to legacy pickle format
        splits_path = preprocessed_dir / 'splits.pkl'
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
    else:
        with open(splits_path) as f:
            splits = json.load(f)

    train_cases = splits[args.fold]['train']
    val_cases = splits[args.fold]['val']
    print(f"Fold {args.fold}: {len(train_cases)} train, {len(val_cases)} val")

    # Get patch size in mm
    patch_size_mm = tuple(args.patch_size_mm)

    # Get reference spacing for transforms and initial model construction
    with open(preprocessed_dir / f"{train_cases[0]}.pkl", 'rb') as f:
        ref_props = pickle.load(f)
    ref_spacing = tuple(ref_props['spacing'])

    # Convert mm to voxels for transform configuration (approximate)
    pooling_factor = getattr(args, 'pooling_factor', 8)
    ref_patch_voxels = tuple(
        max(pooling_factor, int(round(mm / sp)) // pooling_factor * pooling_factor)
        for mm, sp in zip(patch_size_mm, ref_spacing)
    )

    # Create transforms using JAX adapter (NumpyToJax instead of NumpyToTensor)
    train_transforms = get_training_transforms_jax(
        ref_patch_voxels,
        disable_mirroring=getattr(args, 'disable_mirroring', False),
        disable_spatial=getattr(args, 'disable_spatial', True),
    )
    val_transforms = get_validation_transforms_jax()

    # Create multi-resolution data loaders
    print("Setting up training loader...", flush=True)
    train_loader = MultiResolutionLoader(
        preprocessed_dir=str(preprocessed_dir),
        case_identifiers=train_cases,
        batch_size=getattr(args, 'batch_size', 2),
        patch_size_mm=patch_size_mm,
        oversample_foreground_percent=getattr(args, 'foreground_oversample', 0.33),
        num_workers=getattr(args, 'num_workers', 1),
        transforms=train_transforms,
        pooling_factor=pooling_factor,
        dynamic_batch_size=getattr(args, 'dynamic_batch_size', False),
        target_memory_mb=getattr(args, 'target_memory_mb', 0),
        n_base_filters=args.n_base_filters,
        n_downsample=args.n_downsample,
        min_batch_size=1,  # Allow batch_size=1; gradient accumulation handles the rest
        max_batch_size=getattr(args, 'max_batch_size', 24),
        fp16=args.fp16,
        subsample_weight=getattr(args, 'subsample_weight', 0.0),
        model_scale=getattr(args, 'scale', 2.0),
        min_spacing=getattr(args, 'min_spacing', 0.0),
        max_inplane_spacing=getattr(args, 'max_inplane_spacing', 0.0),
        min_slice_thickness=getattr(args, 'min_slice_thickness', 0.0),
        max_slice_thickness=getattr(args, 'max_slice_thickness', 0.0),
        min_loader_cases=getattr(args, 'min_loader_cases', 2),
        group_balance=getattr(args, 'group_balance', 0.0),
        planned_batch_sizes=planned_batch_sizes,
    )
    print("Training loader ready.", flush=True)

    print("Setting up validation loader...", flush=True)
    val_loader = MultiResolutionLoader(
        preprocessed_dir=str(preprocessed_dir),
        case_identifiers=val_cases,
        batch_size=getattr(args, 'batch_size', 2),
        patch_size_mm=patch_size_mm,
        oversample_foreground_percent=0.33,
        num_workers=0,  # Synchronous for validation
        transforms=val_transforms,
        model_scale=getattr(args, 'scale', 2.0),
        n_downsample=args.n_downsample,
        min_loader_cases=getattr(args, 'min_loader_cases', 2),
        planned_batch_sizes=planned_batch_sizes,
        planned_val_batch_sizes=planned_val_batch_sizes,
    )
    print("Validation loader ready.", flush=True)

    # Determine number of classes
    sample_seg = np.load(preprocessed_dir / f"{train_cases[0]}_seg.npy")
    n_classes = int(sample_seg.max()) + 1
    print(f"Number of classes: {n_classes}", flush=True)

    # Build model
    print("Building JAX E3nnUNet model...", flush=True)
    model = build_model(args, n_classes, ref_spacing)
    n_params = sum(p.size for p in jax.tree.leaves(nnx.split(model)[1]))
    print(f"Model parameters: {n_params:,}")

    # Compute steps_per_epoch from loader group weights and batch sizes
    weights = train_loader.group_weights  # list of (spacing, weight)
    weight_map_opt = {sp: w for sp, w in weights if w > 0}
    total_weight = sum(weight_map_opt.values())
    avg_batch = sum(
        train_loader.group_batch_sizes[sp] * w
        for sp, w in weight_map_opt.items()
    ) / max(total_weight, 1e-10)
    steps_per_epoch = max(1, int(args.patches_per_epoch / avg_batch))
    print(f"Estimated steps/epoch: {steps_per_epoch} (avg batch: {avg_batch:.1f})")

    # Build optimizer (cosine decay over full training run in optimizer steps)
    optimizer, lr_schedule_fn = build_optimizer(model, args, steps_per_epoch)

    # Plan validation: compare runtime args against planned config.json
    _write_plan_validation_log(args, output_dir, config_hash)

    # Training state
    start_epoch = 0
    best_val_dice = 0.0
    total_opt_steps = 0

    # BF16 mixed precision
    use_fp16 = args.fp16
    if use_fp16:
        print("Using BF16 mixed precision training")

    # Training history for plotting (nnUNet-style)
    train_losses_history = []
    val_losses_history = []
    pseudo_dice_history = []
    ema_pseudo_dice_history = []
    lr_history = []
    epoch_times_history = []

    # Cumulative resolution tracking across all epochs
    all_train_resolutions = []
    last_val_resolutions = []

    # Safety: refuse to overwrite existing checkpoint without --resume
    if not args.resume and (output_dir / 'checkpoint.pkl').exists():
        raise RuntimeError(
            f"checkpoint.pkl already exists in {output_dir}. "
            f"Use --resume to continue training, or remove the checkpoint to start fresh."
        )

    # Resume if requested
    if args.resume and (output_dir / 'checkpoint.pkl').exists():
        print(f"Resuming from {output_dir / 'checkpoint.pkl'}")
        data = load_checkpoint(output_dir / 'checkpoint.pkl', model, optimizer)

        # Verify config hash matches
        saved_hash = data.get('config_hash')
        if saved_hash and config_hash and saved_hash != config_hash:
            raise RuntimeError(
                f"Config hash mismatch on resume! "
                f"Checkpoint was trained with config hash {saved_hash}, "
                f"but current config.json has hash {config_hash}. "
                f"This suggests config.json was modified after training started."
            )

        start_epoch = data['epoch'] + 1
        best_val_dice = data['best_val_dice']
        if 'logging' in data:
            log_data = data['logging']
            train_losses_history = log_data.get('train_losses', [])
            val_losses_history = log_data.get('val_losses', [])
            pseudo_dice_history = log_data.get('pseudo_dice', [])
            ema_pseudo_dice_history = log_data.get('ema_pseudo_dice', [])
            lr_history = log_data.get('lrs', [])
            epoch_times_history = log_data.get('epoch_times', [])

        # Recover total_opt_steps from adamw step counter in optax chain.
        # With inject_hyperparams: opt_state[1].inner_state[0].count
        # Without inject_hyperparams: opt_state[1].count
        try:
            adamw_count = optimizer.opt_state[1].inner_state[0].count
            total_opt_steps = int(adamw_count)
            print(f"Recovered optimizer step count: {total_opt_steps}")
        except (IndexError, AttributeError, TypeError):
            try:
                adamw_count = optimizer.opt_state[1].count
                total_opt_steps = int(adamw_count)
                print(f"Recovered optimizer step count (legacy): {total_opt_steps}")
            except (IndexError, AttributeError, TypeError):
                total_opt_steps = start_epoch * steps_per_epoch
                print(f"Could not read optimizer step count, estimated: {total_opt_steps}")

        print(f"Resumed at epoch {start_epoch}, best dice: {best_val_dice:.4f}, "
              f"opt_steps: {total_opt_steps}, lr: {float(lr_schedule_fn(total_opt_steps)):.2e}")

    # Build JIT-compiled train step (must be after potential resume)
    no_background = getattr(args, 'no_background_dice', False)
    step_fn = create_jitted_train_step_dynamic(
        model, optimizer, n_classes=n_classes, donate=False,
        no_background=no_background,
    )
    val_step_fn = create_jitted_val_step(
        model, n_classes=n_classes,
        deep_supervision=args.deep_supervision,
        no_background=no_background,
    )

    # Structured JSON log: one dict per epoch, append-friendly
    json_log_path = output_dir / 'training_log.json'
    if json_log_path.exists():
        with open(json_log_path) as f:
            json_log = json.load(f)
    else:
        json_log = []

    # Min batch size for gradient accumulation
    min_batch_size = getattr(args, 'min_batch_size', 1)

    # Open text log file
    log_file = open(output_dir / 'train.log', 'a')

    # Load pretrained weights if requested
    init_checkpoint = getattr(args, 'init_checkpoint', None)
    validate_only_first = init_checkpoint and start_epoch == 0
    if init_checkpoint and start_epoch == 0:
        init_path = Path(init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_path}")
        print(f"Loading pretrained weights from: {init_path}", flush=True)
        from irrepunet.models_jax.weight_transfer import load_pytorch_checkpoint
        load_pytorch_checkpoint(model, str(init_path))
        print(f"Loaded pretrained weights", flush=True)
        # Rebuild step_fn and val_step_fn after loading weights
        step_fn = create_jitted_train_step_dynamic(
            model, optimizer, n_classes=n_classes, donate=False,
            no_background=no_background,
        )
        val_step_fn = create_jitted_val_step(
            model, n_classes=n_classes,
            deep_supervision=args.deep_supervision,
            no_background=no_background,
        )

    # Build spacing_groups info dict for SplitJITManager
    # Maps spacing tuple -> {'patch_size': (D,H,W), 'batch_size': int, 'weight': float}
    weight_map = dict(train_loader.group_weights)
    spacing_groups_info = {}
    for spacing in train_loader.group_batch_sizes:
        spacing_groups_info[spacing] = {
            'patch_size': train_loader.group_patch_sizes[spacing],
            'batch_size': train_loader.group_batch_sizes[spacing],
            'weight': weight_map.get(spacing, 0.0),
        }

    # Targeted pooling: compute L1 pool kernel overrides to consolidate L2+ families
    targeted_pooling = getattr(args, 'targeted_pooling', False)
    pool_kernel_overrides = {}
    if targeted_pooling and model.n_downsample >= 3:
        from irrepunet.models_jax.bands import compute_targeted_k1_for_groups
        raw_overrides = compute_targeted_k1_for_groups(
            spacing_groups_info, model.scale, model.n_downsample, model.diameter,
        )
        # Convert k1 override -> full override dict {level_1: k1_tuple}
        for spacing, k1 in raw_overrides.items():
            pool_kernel_overrides[spacing] = {1: k1}

    # SplitJITManager: manages both split (L0|L1|L2+) and monolithic JIT
    # compilations in a background thread. Split pieces compile first (fewer
    # unique programs) enabling fast validation. Monolithic programs compile
    # after for training readiness.
    # Must be after init_checkpoint since step_fn may be rebuilt.
    bg_compiler = SplitJITManager(
        model, optimizer, step_fn, spacing_groups_info,
        use_fp16=args.fp16,
        cache_max=len(spacing_groups_info),  # ~0.5-0.7 GB RSS per program; revisit if spacing groups grow large
        pool_kernel_overrides=pool_kernel_overrides if targeted_pooling else None,
    )
    bg_compiler.warmup(n=15)
    bg_compiler.start()

    # Split validation step: uses 5 sequential JIT calls instead of monolithic
    split_val_step_fn = create_split_val_step(
        model, bg_compiler, n_classes=n_classes,
        deep_supervision=args.deep_supervision,
        no_background=no_background,
    )

    print(f"\nStarting training from epoch {start_epoch} to {args.epochs}", flush=True)
    print(f"LR schedule: {args.lr:.2e} -> 1e-7 over {args.epochs * steps_per_epoch} optimizer steps")
    if getattr(args, 'resolution_jitter_sigma', 0) > 0:
        print(f"Resolution jitter enabled: sigma={args.resolution_jitter_sigma}")
    if getattr(args, 'scale_jitter_std', 0) > 0:
        print(f"Scale jitter enabled: std={args.scale_jitter_std}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Skip training on epoch 0 when using pretrained weights (baseline validation)
        if validate_only_first and epoch == 0:
            train_loss = 0.0
            print(f"Epoch {epoch:4d} | Skipping training (baseline validation of pretrained weights)")
        else:
            # --- Training ---
            train_loss = 0.0
            n_train_batches = 0
            n_train_patches = 0

            n_skipped = 0

            while n_train_patches < args.patches_per_epoch:
                batch, spacing = next(train_loader)

                # Use canonical group spacing for model (bounds JIT traces
                # to exactly the number of spacing groups)
                group_sp = batch.get('group_spacing')
                if group_sp is not None:
                    spacing = tuple(float(s) for s in group_sp)
                else:
                    spacing = tuple(float(s) for s in spacing)
                batch_size = batch['image'].shape[0]

                # Skip if not yet compiled — enqueue for priority compilation
                if not bg_compiler.is_ready(spacing):
                    bg_compiler.prioritize(spacing)
                    n_skipped += 1
                    if n_skipped % 50 == 1:
                        print(f"    Skipped {spacing} (not yet compiled, "
                              f"{bg_compiler.n_ready}/{len(spacing_groups_info)} "
                              f"ready)", flush=True)
                    # Yield CPU to background compiler thread
                    time.sleep(0.01)
                    continue  # don't count toward n_train_patches

                # Gradient accumulation for small-batch groups
                if batch_size < min_batch_size and spacing in train_loader.group_loaders:
                    accum_steps = math.ceil(min_batch_size / batch_size)
                else:
                    accum_steps = 1

                # Apply resolution jitter if enabled
                resolution_jitter_sigma = getattr(args, 'resolution_jitter_sigma', 0.0)
                if resolution_jitter_sigma > 0:
                    jitter = np.random.normal(0, resolution_jitter_sigma, 3)
                    jittered_spacing = tuple(max(0.1, s + j) for s, j in zip(spacing, jitter))
                else:
                    jittered_spacing = spacing

                # Track resolution for this batch (cumulative across epochs)
                all_train_resolutions.append({
                    'spacing': jittered_spacing,
                    'batch_size': batch_size,
                })

                # Update LR from schedule (applied externally, not baked into XLA graph)
                set_optimizer_lr(optimizer, lr_schedule_fn(total_opt_steps))

                # Get pool kernel overrides for this spacing (only for non-jittered)
                sp_overrides = pool_kernel_overrides.get(jittered_spacing)

                if accum_steps > 1:
                    # Gradient accumulation: collect multiple batches
                    micro_batches = []

                    # First batch (convert NumPy -> JAX in main process)
                    model.update_spacing(jittered_spacing, override_pool_kernels=sp_overrides)
                    image = jnp.array(batch['image'])
                    label = jnp.array(batch['label'])
                    if use_fp16:
                        image = image.astype(jnp.bfloat16)
                    micro_batches.append({'image': image, 'label': label})

                    # Additional accumulation batches from same group
                    for _ in range(accum_steps - 1):
                        accum_batch = next(train_loader.group_loaders[spacing])
                        accum_image = jnp.array(accum_batch['image'])
                        accum_label = jnp.array(accum_batch['label'])
                        if use_fp16:
                            accum_image = accum_image.astype(jnp.bfloat16)
                        micro_batches.append({'image': accum_image, 'label': accum_label})

                        all_train_resolutions.append({
                            'spacing': jittered_spacing,
                            'batch_size': accum_image.shape[0],
                        })

                    loss = accumulated_train_step(
                        model, optimizer, micro_batches, n_classes=n_classes,
                        no_background=no_background,
                    )
                    n_train_patches += batch_size * accum_steps
                else:
                    # Normal single-batch step (convert NumPy -> JAX in main process)
                    model.update_spacing(jittered_spacing, override_pool_kernels=sp_overrides)
                    image = jnp.array(batch['image'])
                    label = jnp.array(batch['label'])
                    if use_fp16:
                        image = image.astype(jnp.bfloat16)

                    jax_batch = {'image': image, 'label': label}
                    loss = step_fn(jax_batch)
                    n_train_patches += batch_size

                train_loss += float(loss)
                n_train_batches += 1
                total_opt_steps += 1

            if n_skipped > 0:
                print(f"    Epoch {epoch}: skipped {n_skipped} batches "
                      f"(BG compile: {bg_compiler.n_ready}/"
                      f"{len(spacing_groups_info)} ready)", flush=True)

            train_loss /= max(n_train_batches, 1)

        # --- Validation ---
        val_results = validate_epoch(
            model, val_loader, n_classes=n_classes,
            val_patches=args.val_patches, fp16=use_fp16,
            deep_supervision=args.deep_supervision,
            no_background=getattr(args, 'no_background_dice', False),
            val_step_fn=val_step_fn,
            split_val_step_fn=split_val_step_fn,
            pool_kernel_overrides=pool_kernel_overrides if targeted_pooling else None,
        )
        val_dice = val_results['pseudo_dice']
        val_loss = val_results['val_loss']
        val_time = val_results.get('val_time', 0.0)
        epoch_val_resolutions = val_results['val_resolutions']

        # EMA pseudo-dice (nnUNet-style smoothing)
        if ema_pseudo_dice_history:
            ema_pseudo_dice = ema_pseudo_dice_history[-1] * 0.9 + 0.1 * val_dice
        else:
            ema_pseudo_dice = val_dice

        epoch_time = time.time() - epoch_start

        # Get current LR from schedule (actual optimizer step count)
        is_baseline = validate_only_first and epoch == 0
        if is_baseline:
            lr = float(lr_schedule_fn(0))
        else:
            lr = float(lr_schedule_fn(total_opt_steps))

        # Logging
        log_msg = (f"Epoch {epoch:4d} | loss: {train_loss:.4f} | val_loss: {val_loss:.4f} "
                   f"| pseudo_dice: {val_dice:.4f} | ema_dice: {ema_pseudo_dice:.4f} "
                   f"| lr: {lr:.2e} | time: {epoch_time:.1f}s (val: {val_time:.1f}s)")
        if is_baseline:
            log_msg = (f"Epoch {epoch:4d} | BASELINE (pretrained) | val_loss: {val_loss:.4f} "
                       f"| pseudo_dice: {val_dice:.4f} | ema_dice: {ema_pseudo_dice:.4f} "
                       f"| time: {epoch_time:.1f}s (val: {val_time:.1f}s)")
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
                'ema_pseudo_dice': ema_pseudo_dice,
                'lr': lr,
                'epoch_time': epoch_time,
                'val_time': val_time,
            }
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
            'val_time': val_time,
        }
        json_log.append(epoch_log)
        with open(json_log_path, 'w') as f:
            json.dump(json_log, f, indent=1)

        # Store last epoch's val resolutions for plotting
        last_val_resolutions = epoch_val_resolutions

        # Log and plot per-resolution-group dice (every 10 epochs)
        if epoch % 10 == 0:
            group_dices = defaultdict(list)
            for entry in last_val_resolutions:
                entry_spacing = entry['spacing']
                sorted_sp = tuple(sorted(entry_spacing))
                key = (f'{sorted_sp[0]:.3f}', f'{sorted_sp[1]:.3f}', f'{sorted_sp[2]:.3f}')
                group_dices[key].append(entry['dice'])
            if group_dices:
                print(f'  Val dice by resolution group (epoch {epoch}):')
                log_file.write(f'  Val dice by resolution group (epoch {epoch}):\n')
                wandb_group_metrics = {}
                for key in sorted(group_dices.keys()):
                    dices = group_dices[key]
                    mean_d = np.mean(dices)
                    print(f'    {key}: {mean_d:.4f} (n={len(dices)})')
                    log_file.write(f'    {key}: {mean_d:.4f} (n={len(dices)})\n')
                    wandb_group_metrics[f'group_dice/{key}'] = mean_d
                log_file.flush()
                if wandb_run is not None and wandb_group_metrics:
                    wandb_run.log(wandb_group_metrics, step=epoch)
                # Append group dice to this epoch's JSON log entry
                epoch_log['group_dice'] = {
                    str(k): round(np.mean(v), 4) for k, v in sorted(group_dices.items())
                }
                with open(json_log_path, 'w') as f:
                    json.dump(json_log, f, indent=1)

            plot_resolution_density(all_train_resolutions, last_val_resolutions,
                                    output_dir, epoch)

        # Plot progress every epoch (overwritten each time)
        plot_progress(train_losses_history, val_losses_history,
                      pseudo_dice_history, ema_pseudo_dice_history,
                      lr_history, epoch_times_history, output_dir)

        # Save checkpoint
        logging_dict = {
            'train_losses': train_losses_history,
            'val_losses': val_losses_history,
            'pseudo_dice': pseudo_dice_history,
            'ema_pseudo_dice': ema_pseudo_dice_history,
            'lrs': lr_history,
            'epoch_times': epoch_times_history,
        }
        save_checkpoint(
            output_dir / 'checkpoint.pkl',
            model, optimizer, epoch, best_val_dice, logging_dict, config_hash,
        )

        # Save best model (using EMA pseudo-dice, like nnUNet)
        if ema_pseudo_dice > best_val_dice:
            best_val_dice = ema_pseudo_dice
            save_checkpoint(
                output_dir / 'model_best.pkl',
                model, optimizer, epoch, best_val_dice, logging_dict, config_hash,
            )
            print(f"  New best EMA pseudo dice: {best_val_dice:.4f}")
            if wandb_run is not None:
                wandb_run.log({'best_pseudo_dice': best_val_dice}, step=epoch)

    bg_compiler.shutdown()
    if wandb_run is not None:
        wandb_run.finish()
    log_file.close()
    print(f"\nTraining complete. Best EMA pseudo dice: {best_val_dice:.4f}")


# =============================================================================
# Parallel JIT Warmup
# =============================================================================

def _compile_cost_estimate(group):
    """Estimate relative compile cost from patch volume * batch_size."""
    p = group['patch_size']
    return p[0] * p[1] * p[2] * group['batch_size']


def run_warmup(args, config):
    """Orchestrate parallel warmup: partition spacing groups across workers.

    Launches W subprocess workers that each compile their assigned spacing
    groups via .lower().compile(), populating the XLA persistent cache.

    Parameters
    ----------
    args : argparse.Namespace
        Must have warmup_workers, warmup_top_n, config (path), gpu.
    config : dict
        Loaded config.json contents.
    """
    t_start = time.time()

    # --- Extract spacing groups from config ---
    if 'loader_groups' in config:
        groups = []
        total_cases = sum(len(g.get('case_ids', [])) for g in config['loader_groups'])
        for g in config['loader_groups']:
            spacing = tuple(float(s) for s in g['spacing'])
            patch_size = tuple(int(v) for v in g['patch_size_voxels'])
            batch_size = int(g['batch_size'])
            n_cases = len(g.get('case_ids', []))
            weight = n_cases / max(total_cases, 1)
            groups.append({
                'spacing': list(spacing),
                'patch_size': list(patch_size),
                'batch_size': batch_size,
                'weight': weight,
            })
    else:
        # Fallback: build a temporary MultiResolutionLoader to discover groups
        print("Config has no loader_groups, discovering from data...", flush=True)
        # Config may nest data under 'data' key or at top level
        data_section = config.get('data', config)
        training_section = config.get('training', config)
        model_section = config.get('model', config)
        preprocessed_dir = Path(
            data_section.get('preprocessed_dir')
            or getattr(args, 'preprocessed_dir', None)
            or ''
        )
        if not preprocessed_dir.exists():
            print(f"Error: preprocessed_dir not found: {preprocessed_dir}", file=sys.stderr)
            sys.exit(1)
        splits_path = preprocessed_dir / 'splits_final.json'
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
        else:
            with open(preprocessed_dir / 'splits.pkl', 'rb') as f:
                splits = pickle.load(f)

        fold = data_section.get('fold', getattr(args, 'fold', 0))
        train_cases = splits[fold]['train']
        patch_size_mm = tuple(training_section.get('patch_size_mm', [80, 80, 80]))

        loader = MultiResolutionLoader(
            preprocessed_dir=str(preprocessed_dir),
            case_identifiers=train_cases,
            batch_size=training_section.get('batch_size', 2),
            patch_size_mm=patch_size_mm,
            oversample_foreground_percent=0.33,
            num_workers=0,
            transforms=None,
            pooling_factor=training_section.get('pooling_factor', 8),
            dynamic_batch_size=training_section.get('dynamic_batch_size', False),
            target_memory_mb=training_section.get('target_memory_mb', 0),
            n_base_filters=model_section.get('n_base_filters', 2),
            n_downsample=model_section.get('n_downsample', 4),
            min_batch_size=1,
            max_batch_size=training_section.get('max_batch_size', 24),
            fp16=config.get('hardware', config).get('fp16', False),
            subsample_weight=training_section.get('subsample_weight', 0.0),
            model_scale=model_section.get('scale', 2.0),
            min_spacing=training_section.get('min_spacing', 0.0),
            max_inplane_spacing=training_section.get('max_inplane_spacing', 0.0),
            min_slice_thickness=training_section.get('min_slice_thickness', 0.0),
            max_slice_thickness=training_section.get('max_slice_thickness', 0.0),
            min_loader_cases=training_section.get('min_loader_cases', 2),
        )

        weight_map = dict(loader.group_weights)
        total_weight = sum(w for _, w in loader.group_weights)
        groups = []
        for spacing in loader.group_batch_sizes:
            groups.append({
                'spacing': list(spacing),
                'patch_size': list(loader.group_patch_sizes[spacing]),
                'batch_size': loader.group_batch_sizes[spacing],
                'weight': weight_map.get(spacing, 0.0) / max(total_weight, 1e-10),
            })
        del loader

    if not groups:
        print("No spacing groups found. Nothing to warmup.", file=sys.stderr)
        return

    # --- Select top N by weight ---
    groups.sort(key=lambda g: g['weight'], reverse=True)
    top_n = args.warmup_top_n
    if top_n > 0:
        groups = groups[:top_n]

    print(f"Warming up {len(groups)} spacing groups with {args.warmup_workers} workers")

    # --- Load-balanced partition across workers ---
    # Sort by compile cost descending, then round-robin to balance load
    groups.sort(key=_compile_cost_estimate, reverse=True)
    n_workers = min(args.warmup_workers, len(groups))
    worker_groups = [[] for _ in range(n_workers)]
    for i, g in enumerate(groups):
        worker_groups[i % n_workers].append(g)

    # Print summary
    for w_idx, wg in enumerate(worker_groups):
        spacings_str = ', '.join(f"({g['spacing'][0]:.2f},{g['spacing'][1]:.2f},{g['spacing'][2]:.2f})"
                                 for g in wg)
        print(f"  Worker {w_idx}: {len(wg)} groups [{spacings_str}]")

    # --- Launch worker subprocesses ---
    config_path = str(Path(args.config).resolve())
    script_path = str(Path(__file__).resolve())
    gpu_id = args.gpu

    procs = []
    for w_idx in range(n_workers):
        groups_json = json.dumps(worker_groups[w_idx])
        cmd = [
            sys.executable, '-u', script_path,
            '--_warmup_worker',
            '--config', config_path,
            '--gpu', str(gpu_id),
            '--_warmup_groups_json', groups_json,
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        procs.append((w_idx, proc))
        print(f"  Worker {w_idx} launched (PID {proc.pid})", flush=True)

    # --- Monitor stdout from all workers ---
    import selectors
    sel = selectors.DefaultSelector()
    for w_idx, proc in procs:
        sel.register(proc.stdout, selectors.EVENT_READ, data=w_idx)

    n_active = len(procs)
    while n_active > 0:
        events = sel.select(timeout=5.0)
        for key, _ in events:
            w_idx = key.data
            line = key.fileobj.readline()
            if line:
                print(f"  [W{w_idx}] {line.rstrip()}", flush=True)
            else:
                # EOF: worker finished
                sel.unregister(key.fileobj)
                n_active -= 1

    sel.close()

    # --- Wait for all workers, report status ---
    failures = []
    for w_idx, proc in procs:
        proc.wait()
        if proc.returncode != 0:
            failures.append((w_idx, proc.returncode))

    elapsed = time.time() - t_start
    if failures:
        print(f"\nWARNING: {len(failures)} worker(s) failed:")
        for w_idx, rc in failures:
            print(f"  Worker {w_idx}: exit code {rc}")
        print("Training can still proceed — BackgroundJITCompiler fills gaps.")
    else:
        print(f"\nAll {n_workers} workers completed successfully.")

    print(f"Warmup finished in {elapsed:.1f}s")


def run_warmup_worker(args):
    """Subprocess worker: compile assigned spacing groups and exit.

    Called via ``train_jax.py --_warmup_worker --config ... --_warmup_groups_json ...``

    Parameters
    ----------
    args : argparse.Namespace
        Must have config, _warmup_groups_json, gpu.
    """
    groups = json.loads(args._warmup_groups_json)
    if not groups:
        print("No groups assigned, exiting.", flush=True)
        return

    # Load config and build args (reuse existing config loading)
    config_path = Path(args.config)
    config = load_experiment_config(config_path)
    train_args = args_from_config(config, config_path=config_path, cli_resume=False)

    # Determine n_classes from a seg file
    preprocessed_dir = Path(train_args.preprocessed_dir)
    splits_path = preprocessed_dir / 'splits_final.json'
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
    else:
        with open(preprocessed_dir / 'splits.pkl', 'rb') as f:
            splits = pickle.load(f)

    fold = getattr(train_args, 'fold', 0)
    train_cases = splits[fold]['train']
    sample_seg = np.load(preprocessed_dir / f"{train_cases[0]}_seg.npy")
    n_classes = int(sample_seg.max()) + 1
    del sample_seg

    # Get reference spacing from first case
    with open(preprocessed_dir / f"{train_cases[0]}.pkl", 'rb') as f:
        ref_props = pickle.load(f)
    ref_spacing = tuple(ref_props['spacing'])

    # Build model
    print(f"Building model (n_classes={n_classes}, ref_spacing={ref_spacing})", flush=True)
    model = build_model(train_args, n_classes, ref_spacing)

    # Build optimizer (steps_per_epoch doesn't affect XLA graph with inject_hyperparams)
    optimizer, _ = build_optimizer(model, train_args, steps_per_epoch=100)

    # Create step_fn (donate=False since we don't need buffer donation for warmup)
    no_bg = getattr(train_args, 'no_background_dice', False)
    step_fn = create_jitted_train_step_dynamic(
        model, optimizer, n_classes=n_classes, donate=False,
        no_background=no_bg,
    )
    jitted_fn = step_fn.jitted_fn
    dtype = jnp.bfloat16 if getattr(train_args, 'fp16', False) else jnp.float32

    # Compile each assigned group
    n_groups = len(groups)
    for i, group in enumerate(groups):
        spacing = tuple(float(s) for s in group['spacing'])
        patch_size = tuple(int(v) for v in group['patch_size'])
        batch_size = int(group['batch_size'])

        t0 = time.time()
        try:
            # Update model spacing and snapshot graphdef + state abstracts
            model.update_spacing(spacing)
            graphdef, state = nnx.split((model, optimizer))
            state_abs = jax.tree.map(
                lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state
            )
            del state

            x_abs = jax.ShapeDtypeStruct((batch_size, 1, *patch_size), dtype)
            y_abs = jax.ShapeDtypeStruct((batch_size, *patch_size), jnp.int32)

            # AOT compile — writes to XLA persistent cache
            jitted_fn.lower(graphdef, state_abs, x_abs, y_abs).compile()

            dt = time.time() - t0
            print(f"[WARMUP done={i+1}/{n_groups} spacing=({spacing[0]:.2f},{spacing[1]:.2f},{spacing[2]:.2f}) "
                  f"patch={patch_size} bs={batch_size} time={dt:.1f}s]", flush=True)
        except Exception as e:
            dt = time.time() - t0
            print(f"[WARMUP FAIL={i+1}/{n_groups} spacing=({spacing[0]:.2f},{spacing[1]:.2f},{spacing[2]:.2f}) "
                  f"error={e} time={dt:.1f}s]", flush=True)


# =============================================================================
# CLI / Config
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train e3nnUNet (JAX backend)')

    # Config file argument (optional)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json from --plan_only. '
                             'If provided, other arguments are ignored.')

    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--preprocessed_dir', type=str, required=False, default=None,
                            help='Path to preprocessed irrepunet directory')
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
                             help='Scalar activation function (default: softplus)')
    model_group.add_argument('--dropout', type=float, default=0.0,
                             help='Dropout probability (default: 0.0)')
    model_group.add_argument('--max_features', type=int, default=320,
                             help='Max features per level (default: 320)')
    model_group.add_argument('--irrep_ratios', type=int, nargs='+', default=[4, 2, 1],
                             help='Irrep multipliers for l=0,1,2 (default: 4 2 1)')
    model_group.add_argument('--fill_to_max', action='store_true',
                             help='Top up capped levels with scalar irreps')
    model_group.add_argument('--pool_mode', type=str, default='maxpool3d',
                             choices=['maxpool3d', 'average'],
                             help='Pooling mode (default: maxpool3d)')
    model_group.add_argument('--scale', type=float, default=2.0,
                             help='Base pooling scale in mm (default: 2.0)')

    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--output_dir', type=str, required=False, default=None,
                             help='Output directory for checkpoints and logs')
    train_group.add_argument('--epochs', type=int, default=1000,
                             help='Number of epochs (default: 1000)')
    train_group.add_argument('--batch_size', type=int, default=2,
                             help='Batch size (default: 2)')
    train_group.add_argument('--dynamic_batch_size', action=argparse.BooleanOptionalAction, default=True,
                             help='Auto-scale batch size per resolution group (default: True). '
                                  'Use --no-dynamic_batch_size to disable.')
    train_group.add_argument('--target_memory_mb', type=float, default=8000,
                             help='Target GPU memory in MB for dynamic batch sizing')
    train_group.add_argument('--min_batch_size', type=int, default=1,
                             help='Minimum effective batch size (default: 1)')
    train_group.add_argument('--max_batch_size', type=int, default=24,
                             help='Maximum batch size for dynamic sizing (default: 24)')
    train_group.add_argument('--lr', type=float, default=0.01,
                             help='Learning rate (default: 0.01)')
    train_group.add_argument('--weight_decay', type=float, default=3e-5,
                             help='Weight decay (default: 3e-5)')
    train_group.add_argument('--grad_clip', type=float, default=12.0,
                             help='Gradient clipping (default: 12.0)')
    train_group.add_argument('--patch_size_mm', type=float, nargs=3, default=[80, 80, 80],
                             help='Patch size in mm (default: 80 80 80)')
    train_group.add_argument('--patches_per_epoch', type=int, default=500,
                             help='Training patches per epoch (default: 500)')
    train_group.add_argument('--val_patches', type=int, default=100,
                             help='Validation patches per epoch (default: 100)')
    train_group.add_argument('--foreground_oversample', type=float, default=0.33,
                             help='Foreground oversampling ratio (default: 0.33)')
    train_group.add_argument('--subsample_weight', type=float, default=0.0,
                             help='Weight for subsampled data (default: 0.0)')
    train_group.add_argument('--group_balance', type=float, default=0.0,
                             help='Balance sampling across resolution groups (default: 0.0)')
    train_group.add_argument('--min_spacing', type=float, default=0.0,
                             help='Exclude cases with finest spacing below this value')
    train_group.add_argument('--max_inplane_spacing', type=float, default=0.0,
                             help='Exclude cases with in-plane spacing above this value')
    train_group.add_argument('--min_slice_thickness', type=float, default=0.0,
                             help='Exclude cases with slice thickness below this value')
    train_group.add_argument('--max_slice_thickness', type=float, default=0.0,
                             help='Exclude cases with slice thickness above this value')
    train_group.add_argument('--min_loader_cases', type=int, default=2,
                             help='Minimum cases per loader group')
    train_group.add_argument('--pooling_factor', type=int, default=8,
                             help='Ensure patch sizes divisible by this (default: 8)')
    train_group.add_argument('--resolution_jitter_sigma', type=float, default=0.0,
                             help='Std dev of Gaussian jitter applied to spacing (default: 0.0)')
    train_group.add_argument('--targeted_pooling', action='store_true',
                             help='Use targeted L1 pooling to consolidate L2+ JIT families')
    train_group.add_argument('--scale_jitter_std', type=float, default=0.0,
                             help='Std dev of multiplicative jitter applied to pooling scales')
    train_group.add_argument('--num_workers', type=int, default=1,
                             help='Number of dataloader workers per resolution group (default: 1)')
    train_group.add_argument('--resume', action='store_true',
                             help='Resume from checkpoint')
    train_group.add_argument('--init_checkpoint', type=str, default=None,
                             help='Initialize model weights from a pretrained checkpoint')

    # Augmentation arguments
    aug_group = parser.add_argument_group('Augmentation')
    aug_group.add_argument('--disable_spatial', action=argparse.BooleanOptionalAction, default=True,
                           help='Disable spatial augmentation (default: True). '
                                'Use --no-disable_spatial to enable.')
    aug_group.add_argument('--disable_mirroring', action='store_true',
                           help='Disable mirroring augmentation')
    aug_group.add_argument('--deep_supervision', action=argparse.BooleanOptionalAction, default=True,
                           help='Enable deep supervision (default: True). '
                                'Use --no-deep_supervision to disable.')
    aug_group.add_argument('--no_background_dice', action='store_true',
                           help='Exclude background class from Dice loss')
    aug_group.add_argument('--batch_dice', action='store_true',
                           help='Use batch dice instead of per-sample dice')

    # Hardware arguments
    hw_group = parser.add_argument_group('Hardware')
    hw_group.add_argument('--gpu', type=int, default=0,
                          help='GPU device ID (default: 0)')
    hw_group.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True,
                          help='Use BF16 mixed precision training (default: True). '
                               'Use --no-fp16 for FP32.')

    # JAX-specific arguments
    jax_group = parser.add_argument_group('JAX')
    jax_group.add_argument('--remat', type=str, default='none',
                           choices=['none', 'block', 'dots'],
                           help='Rematerialization strategy (default: none)')
    jax_group.add_argument('--donate', action='store_true', default=True,
                           help='Donate input buffers to XLA for reuse (default: True)')

    # Plan only / warmup
    parser.add_argument('--plan_only', action='store_true',
                        help='Pre-compile XLA programs for spacing groups in parallel')
    parser.add_argument('--warmup_top_n', type=int, default=0,
                        help='Number of top spacing groups to warmup (0 = all)')
    parser.add_argument('--warmup_workers', type=int, default=2,
                        help='Number of parallel worker processes (default: 2)')
    # Hidden internal flags for worker subprocesses
    parser.add_argument('--_warmup_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_warmup_groups_json', type=str, default=None, help=argparse.SUPPRESS)

    # W&B tracking arguments
    wandb_group = parser.add_argument_group('Weights & Biases')
    wandb_group.add_argument('--wandb', action='store_true',
                             help='Enable W&B logging (default: off)')
    wandb_group.add_argument('--wandb_project', type=str, default='irrepunet-jax',
                             help='W&B project name (default: irrepunet-jax)')
    wandb_group.add_argument('--wandb_name', type=str, default=None,
                             help='W&B run name (default: output_dir basename)')

    args = parser.parse_args()

    # --- Warmup worker subprocess: compile assigned groups and exit ---
    if getattr(args, '_warmup_worker', False):
        if not args.config:
            print("Error: --_warmup_worker requires --config", file=sys.stderr)
            sys.exit(1)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        # Enable XLA persistent compilation cache (same as training)
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '.xla_compilation_cache')
        jax.config.update('jax_compilation_cache_dir', cache_dir)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
        print(f"Warmup worker starting (GPU {args.gpu}, "
              f"cache: {cache_dir})", flush=True)
        run_warmup_worker(args)
        sys.exit(0)

    # --- Plan-only mode: parallel JIT warmup ---
    if getattr(args, 'plan_only', False):
        if not args.config:
            print("Error: --plan_only requires --config <path>/config.json",
                  file=sys.stderr)
            sys.exit(1)
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        config = load_experiment_config(config_path)
        run_warmup(args, config)
        sys.exit(0)

    # Load from config file if provided
    config_hash = None
    planned_batch_sizes = None
    planned_val_batch_sizes = None

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
        cli_wandb_project = getattr(args, 'wandb_project', 'irrepunet-jax')
        cli_wandb_name = getattr(args, 'wandb_name', None)
        cli_gpu = args.gpu
        cli_remat = getattr(args, 'remat', 'none')

        args = args_from_config(config, config_path=config_path, cli_resume=cli_resume)

        # CLI flags override config
        if cli_wandb:
            args.wandb = True
            args.wandb_project = cli_wandb_project
            if cli_wandb_name:
                args.wandb_name = cli_wandb_name

        # JAX-specific args not in config
        args.remat = cli_remat
        args.donate = True

        # Allow GPU override from CLI
        if cli_gpu != 0 or 'gpu' not in vars(args):
            args.gpu = cli_gpu

        # Extract planned batch sizes from loader_groups
        planned_val_batch_sizes = {}
        if 'loader_groups' in config:
            planned_batch_sizes = {}
            for group in config['loader_groups']:
                spacing_key = tuple(float(s) for s in group['spacing'])
                planned_batch_sizes[spacing_key] = group['batch_size']
                if 'val_batch_size' in group:
                    planned_val_batch_sizes[spacing_key] = group['val_batch_size']
            print(f"Loaded planned batch sizes for {len(planned_batch_sizes)} spacing groups"
                  + (f" (+ {len(planned_val_batch_sizes)} val-specific)"
                     if planned_val_batch_sizes else ""))

        print(f"Configuration loaded successfully")
    else:
        # Legacy mode: validate required arguments
        if not args.preprocessed_dir:
            print("Error: --preprocessed_dir is required (or provide --config)", file=sys.stderr)
            sys.exit(1)
        if not args.output_dir:
            print("Error: --output_dir is required (or provide --config)", file=sys.stderr)
            sys.exit(1)

    # Set GPU visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Configure JAX memory optimizations
    remat = getattr(args, 'remat', 'none')
    if remat == 'dots':
        configure_memory_optimizations(disable_xla_remat=True)

    # Enable XLA persistent compilation cache so JIT warmup is a one-time cost
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '.xla_compilation_cache')
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)

    # Print JAX device info
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"JAX backend: {jax.default_backend()}")

    train(args, config_hash=config_hash, planned_batch_sizes=planned_batch_sizes,
          planned_val_batch_sizes=planned_val_batch_sizes)


if __name__ == '__main__':
    main()
