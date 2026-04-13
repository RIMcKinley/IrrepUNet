#!/usr/bin/env python
"""
Single-case inference script for e3nnUNet.

Applies the same preprocessing as training (reorient to RAS, crop, normalize),
runs sliding window inference, then reverses preprocessing to match original NIfTI.
"""

import argparse
from pathlib import Path

import torch
from irrepunet.inference import predict_nifti


def find_nifti_file(directory: Path, case_name: str, suffix: str = "") -> Path:
    """Find NIfTI file with various extensions."""
    for ext in ['.nii.gz', '.nii']:
        path = directory / f"{case_name}{suffix}{ext}"
        if path.exists():
            return path
    # Try glob
    matches = list(directory.glob(f"{case_name}{suffix}*"))
    if matches:
        return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on a raw NIfTI image using trained e3nnUNet model'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input NIfTI image (or case name if --nnunet_raw is provided)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output NIfTI file path')
    parser.add_argument('--nnunet_raw', type=str, default=None,
                        help='Path to nnUNet_raw dataset directory (if input is a case name)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--overlap', type=float, default=0.69,
                        help='Overlap fraction for sliding window')
    parser.add_argument('--patch_size_mm', type=float, default=80.0,
                        help='Patch size in mm')
    parser.add_argument('--mirror_axes', type=int, nargs='*', default=None,
                        help='Axes for test-time augmentation (e.g. 0 1 2)')
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Save softmax probabilities instead of argmax')
    parser.add_argument('--sw_batch_size', type=int, default=4,
                        help='Patches per forward pass (default: 4)')
    parser.add_argument('--jitter_scale', type=float, default=None,
                        help='Scale-jitter TTA range (e.g. 0.2 = ±20%%). Saves mask + probability.')
    parser.add_argument('--num_jitter', type=int, default=9,
                        help='Number of jitter scales (default: 9)')
    parser.add_argument('--spacing_jitter_scale', type=float, default=None,
                        help='Spacing-jitter TTA range (e.g. 0.2 = ±20%%). Re-projects model at each spacing.')
    parser.add_argument('--hierarchical', action='store_true',
                        help='Use hierarchical multi-resolution inference (coarse-to-fine)')
    parser.add_argument('--resolution_levels', type=float, nargs='+', default=None,
                        help='Isotropic resolution levels in mm, coarse to fine '
                             '(e.g. 5.0 2.0 for 5mm then 2mm). Default: 4.0 2.0')
    parser.add_argument('--coarse_threshold', type=float, default=0.3,
                        help='Softmax threshold for coarse detection (default: 0.3)')
    parser.add_argument('--padding_mm', type=float, default=10.0,
                        help='Padding in mm around detected regions (default: 10.0)')
    parser.add_argument('--native_e3nn', action='store_true',
                        help='Native e3nn inference: skip projection, pass spacing at forward time')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    # Determine input path
    if args.nnunet_raw:
        nnunet_raw = Path(args.nnunet_raw)
        case_name = args.input
        input_path = find_nifti_file(nnunet_raw / 'imagesTr', case_name, '_0000')
        if input_path is None:
            raise FileNotFoundError(f"Could not find {case_name} in {nnunet_raw}/imagesTr")
    else:
        input_path = str(args.input)

    mirror_axes = tuple(args.mirror_axes) if args.mirror_axes else None

    # Parse resolution levels: flat list of floats -> list of isotropic tuples
    resolution_levels = None
    if args.resolution_levels:
        resolution_levels = [(r, r, r) for r in args.resolution_levels]

    predict_nifti(
        checkpoint_path=args.checkpoint,
        input_path=str(input_path),
        output_path=args.output,
        device=args.device,
        patch_size_mm=args.patch_size_mm,
        overlap=args.overlap,
        mirror_axes=mirror_axes,
        save_probabilities=args.save_probabilities,
        sw_batch_size=args.sw_batch_size,
        jitter_scale=args.jitter_scale,
        num_jitter=args.num_jitter,
        spacing_jitter_scale=args.spacing_jitter_scale,
        hierarchical=args.hierarchical,
        resolution_levels=resolution_levels,
        coarse_threshold=args.coarse_threshold,
        padding_mm=args.padding_mm,
        native_e3nn=args.native_e3nn,
    )


if __name__ == '__main__':
    main()
