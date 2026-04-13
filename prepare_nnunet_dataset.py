#!/usr/bin/env python
"""
Prepare a dataset for nnUNet format.

Converts datasets from various formats to the nnUNet raw data structure:
    nnUNet_raw/
        DatasetXXX_Name/
            imagesTr/
                case_0000.nii.gz  (channel 0)
                case_0001.nii.gz  (channel 1, if multi-channel)
            labelsTr/
                case.nii.gz
            dataset.json

Usage:
    python prepare_nnunet_dataset.py \
        --source /path/to/source/data \
        --dataset_id 556 \
        --dataset_name FlorinLesions \
        --output /home/student/nnunet_data/nnUNet_raw
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import gzip

import nibabel as nib
import numpy as np
from tqdm import tqdm


def convert_nii_to_nii_gz(src: Path, dst: Path):
    """Convert .nii to .nii.gz or copy .nii.gz directly."""
    if src.suffix == '.gz':
        shutil.copy2(src, dst)
    else:
        # Load and save as gzipped
        img = nib.load(str(src))
        nib.save(img, str(dst))


def find_florin_annotated_cases(source_dir: Path) -> List[Tuple[str, Path, Path]]:
    """
    Find all annotated cases in the Florin_Annotated dataset format.

    Structure:
        source_dir/
            {patient_id}/
                {scan_id}.nii                    <- image
                {scan_id}/
                    lesions2-corrected.nii.gz    <- segmentation

    Returns:
        List of (case_name, image_path, label_path) tuples
    """
    cases = []

    for patient_dir in sorted(source_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        # Find all scan subdirectories with segmentations
        for scan_subdir in sorted(patient_dir.iterdir()):
            if not scan_subdir.is_dir():
                continue

            # Check for segmentation file
            seg_file = scan_subdir / 'lesions2-corrected.nii.gz'
            if not seg_file.exists():
                # Try alternative name
                seg_file = scan_subdir / 'lesions2.nii.gz'
                if not seg_file.exists():
                    continue

            # Find corresponding image
            scan_id = scan_subdir.name
            img_file = patient_dir / f'{scan_id}.nii'
            if not img_file.exists():
                img_file = patient_dir / f'{scan_id}.nii.gz'
                if not img_file.exists():
                    print(f"Warning: No image found for {patient_id}/{scan_id}")
                    continue

            # Create case name (combine patient and scan, shortened)
            # Use first part of scan ID (date_accession)
            scan_short = '_'.join(scan_id.split('_')[:2])
            case_name = f'{patient_id}_{scan_short}'

            cases.append((case_name, img_file, seg_file))

    return cases


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    dataset_id: int,
    dataset_name: str,
    channel_names: Dict[str, str] = None,
    labels: Dict[str, int] = None,
    source_format: str = 'florin_annotated'
):
    """
    Prepare a dataset in nnUNet raw format.

    Args:
        source_dir: Source data directory
        output_dir: nnUNet_raw directory
        dataset_id: Dataset ID (e.g., 556)
        dataset_name: Dataset name (e.g., FlorinLesions)
        channel_names: Dict mapping channel index to name
        labels: Dict mapping label name to value
        source_format: Format of source data ('florin_annotated', 'generic')
    """
    # Create output directory
    dataset_dir = output_dir / f'Dataset{dataset_id:03d}_{dataset_name}'
    images_dir = dataset_dir / 'imagesTr'
    labels_dir = dataset_dir / 'labelsTr'

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {dataset_dir}")

    # Find cases based on source format
    if source_format == 'florin_annotated':
        cases = find_florin_annotated_cases(source_dir)
    else:
        raise ValueError(f"Unknown source format: {source_format}")

    print(f"Found {len(cases)} cases")

    # Default channel names and labels
    if channel_names is None:
        channel_names = {'0': 'MRI'}
    if labels is None:
        labels = {'background': 0, 'lesion': 1}

    # Determine file ending by checking first case
    file_ending = '.nii.gz'

    # Process cases
    label_values_seen = set()

    for case_name, img_path, seg_path in tqdm(cases, desc="Processing cases"):
        # Copy image (with channel suffix)
        img_out = images_dir / f'{case_name}_0000.nii.gz'
        convert_nii_to_nii_gz(img_path, img_out)

        # Copy segmentation
        seg_out = labels_dir / f'{case_name}.nii.gz'
        convert_nii_to_nii_gz(seg_path, seg_out)

        # Check label values
        seg = nib.load(str(seg_path))
        seg_data = seg.get_fdata()
        label_values_seen.update(np.unique(seg_data).astype(int).tolist())

    print(f"Label values found: {sorted(label_values_seen)}")

    # Update labels dict if we found more values
    if max(label_values_seen) > max(labels.values()):
        print(f"Warning: Found label values {label_values_seen} but labels dict only has {labels}")
        # Add missing labels
        for v in sorted(label_values_seen):
            if v not in labels.values():
                labels[f'class_{v}'] = v

    # Create dataset.json
    dataset_json = {
        'channel_names': channel_names,
        'labels': labels,
        'numTraining': len(cases),
        'file_ending': file_ending
    }

    with open(dataset_dir / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset prepared successfully!")
    print(f"  Location: {dataset_dir}")
    print(f"  Cases: {len(cases)}")
    print(f"  Channels: {channel_names}")
    print(f"  Labels: {labels}")

    return dataset_dir


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for nnUNet')
    parser.add_argument('--source', type=str, required=True,
                        help='Source data directory')
    parser.add_argument('--output', type=str,
                        default='/home/student/nnunet_data/nnUNet_raw',
                        help='nnUNet_raw directory')
    parser.add_argument('--dataset_id', type=int, required=True,
                        help='Dataset ID (e.g., 556)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name (e.g., FlorinLesions)')
    parser.add_argument('--format', type=str, default='florin_annotated',
                        choices=['florin_annotated'],
                        help='Source data format')
    parser.add_argument('--channel_name', type=str, default='MRI',
                        help='Name for channel 0')
    args = parser.parse_args()

    channel_names = {'0': args.channel_name}
    labels = {'background': 0, 'lesion': 1}

    prepare_dataset(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        channel_names=channel_names,
        labels=labels,
        source_format=args.format
    )


if __name__ == '__main__':
    main()
