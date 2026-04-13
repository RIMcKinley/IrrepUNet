#!/usr/bin/env python
"""
Single-case inference script for JAX E3nnUNet.

Applies the same preprocessing as training (reorient to RAS, crop, normalize),
runs sliding window inference, then reverses preprocessing to match original NIfTI.

Usage:
    python infer_case_jax.py \
        --experiment experiments/jax_targeted_pooling_450ep \
        --input /path/to/image.nii.gz \
        --output /path/to/pred.nii.gz

    # With mirror TTA:
    python infer_case_jax.py \
        --experiment experiments/jax_targeted_pooling_450ep \
        --input /path/to/image.nii.gz \
        --output /path/to/pred.nii.gz \
        --mirror_axes 0 1 2

    # From nnUNet raw directory (auto-find case):
    python infer_case_jax.py \
        --experiment experiments/jax_targeted_pooling_450ep \
        --input case_0001 \
        --nnunet_raw /home/student/nnunet_data/nnUNet_raw/Dataset556_CAIM_MS \
        --output /path/to/pred.nii.gz
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import nibabel as nib
from nibabel import orientations as nib_orient
import numpy as np
from scipy.ndimage import binary_fill_holes

from validate_jax import (
    load_jax_model,
    sliding_window_inference_jax,
)


def find_nifti_file(directory, case_name, suffix=""):
    """Find NIfTI file with various extensions."""
    directory = Path(directory)
    for ext in ['.nii.gz', '.nii']:
        path = directory / f"{case_name}{suffix}{ext}"
        if path.exists():
            return path
    matches = list(directory.glob(f"{case_name}{suffix}*"))
    if matches:
        return matches[0]
    return None


def preprocess_nifti(input_path):
    """Preprocess a raw NIfTI: reorient RAS, crop nonzero, z-score normalize.

    Returns (image, spacing, preprocessing_info) where preprocessing_info
    contains everything needed to reverse the transform.
    """
    nii = nib.load(str(input_path))
    original_affine = nii.affine.copy()
    original_header = nii.header.copy()

    # Reorient to RAS
    original_ornt = nib.io_orientation(nii.affine)
    ras_ornt = nib_orient.axcodes2ornt(('R', 'A', 'S'))
    ornt_transform = nib_orient.ornt_transform(original_ornt, ras_ornt)
    ras_data = nib_orient.apply_orientation(nii.get_fdata(), ornt_transform).astype(np.float32)
    new_affine = nii.affine @ nib_orient.inv_ornt_aff(ornt_transform, nii.shape)
    ras_nii = nib.Nifti1Image(ras_data, new_affine, nii.header)

    spacing = tuple(float(s) for s in ras_nii.header.get_zooms()[:3])
    ras_shape = ras_data.shape
    print(f"  RAS shape: {ras_shape}, spacing: {tuple(round(s,3) for s in spacing)}")

    # Crop to nonzero
    ras_data_4d = ras_data[np.newaxis, ...]  # (1, D, H, W)
    nonzero_mask = ras_data_4d[0] != 0
    nonzero_mask = binary_fill_holes(nonzero_mask)
    coords = np.argwhere(nonzero_mask)
    if len(coords) > 0:
        bbox_lb = coords.min(axis=0)
        bbox_ub = coords.max(axis=0) + 1
        bbox = tuple((int(lb), int(ub)) for lb, ub in zip(bbox_lb, bbox_ub))
    else:
        bbox = tuple((0, s) for s in ras_shape)
    slices = tuple(slice(b[0], b[1]) for b in bbox)
    cropped = ras_data_4d[(slice(None),) + slices]
    print(f"  Cropped shape: {cropped.shape[1:]}")

    # Z-score normalize (foreground only)
    fg_mask = cropped[0] != 0
    fg_values = cropped[0][fg_mask]
    mean_val = float(fg_values.mean())
    std_val = float(fg_values.std())
    normalized = np.zeros_like(cropped)
    normalized[0][fg_mask] = (cropped[0][fg_mask] - mean_val) / max(std_val, 1e-8)

    preproc_info = {
        'original_affine': original_affine,
        'original_header': original_header,
        'ras_shape': ras_shape,
        'bbox': bbox,
        'slices': slices,
        'ornt_transform': ornt_transform,
    }

    return normalized, spacing, preproc_info


def postprocess_and_save(probs, preproc_info, output_path, save_probabilities=False):
    """Reverse preprocessing and save as NIfTI in original space."""
    original_affine = preproc_info['original_affine']
    original_header = preproc_info['original_header']
    ras_shape = preproc_info['ras_shape']
    slices = preproc_info['slices']

    # Reverse RAS reorientation
    original_ornt = nib.io_orientation(original_affine)
    ras_ornt = nib_orient.axcodes2ornt(('R', 'A', 'S'))
    reverse_transform = nib_orient.ornt_transform(ras_ornt, original_ornt)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_probabilities:
        full_probs = np.zeros((probs.shape[0],) + ras_shape, dtype=np.float32)
        full_probs[(slice(None),) + slices] = probs
        output_data = np.stack([
            nib_orient.apply_orientation(full_probs[c], reverse_transform)
            for c in range(full_probs.shape[0])
        ], axis=-1)  # (D, H, W, C)
        out_nii = nib.Nifti1Image(output_data, original_affine, original_header)
    else:
        pred_cropped = probs.argmax(axis=0).astype(np.int16)
        full_pred = np.zeros(ras_shape, dtype=np.int16)
        full_pred[slices] = pred_cropped
        output_data = nib_orient.apply_orientation(full_pred, reverse_transform).astype(np.int16)
        out_nii = nib.Nifti1Image(output_data, original_affine, original_header)
        out_nii.header.set_data_dtype(np.int16)

    nib.save(out_nii, str(output_path))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Single-case inference for JAX E3nnUNet')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--checkpoint', type=str, default='model_best.pkl',
                        help='Checkpoint filename')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input NIfTI (or case name with --nnunet_raw)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output NIfTI file path')
    parser.add_argument('--nnunet_raw', type=str, default=None,
                        help='Path to nnUNet_raw dataset directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--overlap', type=float, default=0.69,
                        help='Overlap fraction for sliding window')
    parser.add_argument('--patch_size_mm', type=float, default=None,
                        help='Patch size in mm (default: from config)')
    parser.add_argument('--sw_batch_size', type=int, default=4,
                        help='Patches per forward pass')
    parser.add_argument('--mirror_axes', type=int, nargs='*', default=None,
                        help='Axes for test-time augmentation (e.g. 0 1 2)')
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Save softmax probabilities instead of argmax')
    args = parser.parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    experiment_dir = Path(args.experiment)

    # Load model
    model, config, _ = load_jax_model(experiment_dir, args.checkpoint)

    # Determine input path
    if args.nnunet_raw:
        input_path = find_nifti_file(
            Path(args.nnunet_raw) / 'imagesTr', args.input, '_0000')
        if input_path is None:
            raise FileNotFoundError(
                f"Could not find {args.input} in {args.nnunet_raw}/imagesTr")
    else:
        input_path = args.input

    # Patch size
    patch_mm = args.patch_size_mm
    if patch_mm is None:
        patch_mm = config['training'].get('patch_size_mm', [128.0, 128.0, 128.0])
        if isinstance(patch_mm, list):
            patch_mm = patch_mm[0]

    mirror_axes = tuple(args.mirror_axes) if args.mirror_axes else None

    # Preprocess
    print(f"Loading image from {input_path}")
    t0 = time.time()
    image, spacing, preproc_info = preprocess_nifti(input_path)

    # Set model spacing
    model.update_spacing(spacing)

    # Run inference
    probs = sliding_window_inference_jax(
        model, image, spacing,
        patch_size_mm=patch_mm,
        overlap=args.overlap,
        mirror_axes=mirror_axes,
        sw_batch_size=args.sw_batch_size,
    )

    # Save
    postprocess_and_save(probs, preproc_info, args.output,
                         save_probabilities=args.save_probabilities)

    dt = time.time() - t0
    print(f"Total time: {dt:.1f}s")


if __name__ == '__main__':
    main()
