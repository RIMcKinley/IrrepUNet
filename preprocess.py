#!/usr/bin/env python
"""
Preprocessing script for e3nnUNet.

Preprocesses NIfTI files from nnUNet_raw and saves to nnUNet_preprocessed/irrepunet/.

Preprocessing steps (matching nnUNet minus resampling):
    1. Reorient to RAS orientation
    2. Crop to non-zero region
    3. Normalization matching nnUNet plans (ZScoreNormalization or CTNormalization)

Usage:
    python preprocess.py \
        --nnunet_raw /path/to/nnUNet_raw \
        --nnunet_preprocessed /path/to/nnUNet_preprocessed \
        --dataset Dataset555_CSF01
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm


# =============================================================================
# Normalization classes matching nnUNet
# =============================================================================

class ZScoreNormalization:
    """Z-score normalization matching nnUNet's implementation.

    When use_mask_for_norm=True:
        - Computes mean/std only on masked region (where seg >= 0)
        - Only normalizes the masked region, leaves outside unchanged

    When use_mask_for_norm=False:
        - Computes mean/std on entire image
        - Normalizes entire image
    """

    def __init__(self, use_mask_for_norm: bool = False, intensity_properties: dict = None):
        self.use_mask_for_norm = use_mask_for_norm
        self.intensity_properties = intensity_properties

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(np.float32, copy=True)

        if self.use_mask_for_norm and seg is not None:
            # Normalize only within mask (seg >= 0)
            # In nnUNet, seg == -1 marks "outside" region
            mask = seg >= 0
            if mask.sum() > 0:
                mean = image[mask].mean()
                std = image[mask].std()
                image[mask] = (image[mask] - mean) / max(std, 1e-8)
        else:
            # Global normalization (entire image)
            mean = image.mean()
            std = image.std()
            image = (image - mean) / max(std, 1e-8)

        return image


class CTNormalization:
    """CT normalization matching nnUNet's implementation.

    Clips to percentile range and normalizes using dataset-wide statistics.
    """

    def __init__(self, use_mask_for_norm: bool = False, intensity_properties: dict = None):
        self.use_mask_for_norm = use_mask_for_norm
        self.intensity_properties = intensity_properties

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensity_properties is not None, "CTNormalization requires intensity_properties"

        mean_intensity = self.intensity_properties['mean']
        std_intensity = self.intensity_properties['std']
        lower_bound = self.intensity_properties['percentile_00_5']
        upper_bound = self.intensity_properties['percentile_99_5']

        image = image.astype(np.float32, copy=True)
        np.clip(image, lower_bound, upper_bound, out=image)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)

        return image


def get_normalizer(scheme: str, use_mask_for_norm: bool, intensity_properties: dict = None):
    """Get normalizer instance based on scheme name."""
    if scheme == 'ZScoreNormalization':
        return ZScoreNormalization(use_mask_for_norm, intensity_properties)
    elif scheme == 'CTNormalization':
        return CTNormalization(use_mask_for_norm, intensity_properties)
    elif scheme == 'NoNormalization':
        return None
    else:
        print(f"Warning: Unknown normalization scheme '{scheme}', using ZScoreNormalization")
        return ZScoreNormalization(use_mask_for_norm, intensity_properties)


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient a NIfTI image to RAS orientation.

    Parameters
    ----------
    img : nib.Nifti1Image
        Input NIfTI image

    Returns
    -------
    nib.Nifti1Image
        Reoriented image in RAS orientation
    """
    orig_ornt = nib.io_orientation(img.affine)
    ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)

    reoriented_data = nib.orientations.apply_orientation(img.get_fdata(), transform)
    new_affine = img.affine @ nib.orientations.inv_ornt_aff(transform, img.shape)

    return nib.Nifti1Image(reoriented_data, new_affine, img.header)


def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    """Create a mask of the non-zero region (matching nnUNet's cropping.py).

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (C, D, H, W)

    Returns
    -------
    np.ndarray
        Boolean mask of shape (D, H, W), True where data is non-zero
    """
    assert data.ndim == 4, "data must have shape (C, D, H, W)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def compute_nonzero_bbox(data: np.ndarray) -> Tuple[Tuple[Tuple[int, int], ...], np.ndarray]:
    """Compute bounding box of non-zero region and return the mask.

    Parameters
    ----------
    data : np.ndarray
        Input array (can be multi-channel: C, D, H, W or single: D, H, W)

    Returns
    -------
    bbox : tuple
        Bounding box as ((d0, d1), (h0, h1), (w0, w1))
    nonzero_mask : np.ndarray
        Boolean mask of the non-zero region
    """
    # Handle multi-channel case
    if data.ndim == 4:
        nonzero_mask = create_nonzero_mask(data)
    else:
        nonzero_mask = binary_fill_holes(data != 0)

    # Find non-zero indices
    nonzero = np.where(nonzero_mask)

    if len(nonzero[0]) == 0:
        # No non-zero voxels, return full shape
        return tuple((0, s) for s in nonzero_mask.shape), nonzero_mask

    bbox = tuple((int(np.min(coords)), int(np.max(coords)) + 1) for coords in nonzero)
    return bbox, nonzero_mask


def crop_to_bbox(
    data: np.ndarray,
    bbox: Tuple[Tuple[int, int], ...]
) -> np.ndarray:
    """Crop array to bounding box.

    Parameters
    ----------
    data : np.ndarray
        Input array (C, D, H, W or D, H, W)
    bbox : tuple
        Bounding box as ((d0, d1), (h0, h1), (w0, w1))

    Returns
    -------
    np.ndarray
        Cropped array
    """
    if data.ndim == 4:
        (d0, d1), (h0, h1), (w0, w1) = bbox
        return data[:, d0:d1, h0:h1, w0:w1]
    else:
        (d0, d1), (h0, h1), (w0, w1) = bbox
        return data[d0:d1, h0:h1, w0:w1]


def compute_class_locations(
    label: np.ndarray,
    max_samples_per_class: int = 10000
) -> Dict[int, np.ndarray]:
    """Compute foreground voxel locations for each class.

    Parameters
    ----------
    label : np.ndarray
        Segmentation label of shape (D, H, W)
    max_samples_per_class : int
        Maximum number of voxels to store per class

    Returns
    -------
    dict
        Dictionary mapping class id to array of voxel coordinates
    """
    class_locations = {}
    unique_classes = np.unique(label)

    for c in unique_classes:
        if c == 0:  # Skip background
            continue

        coords = np.array(np.where(label == c)).T  # Shape: (N, 3)

        if len(coords) > max_samples_per_class:
            # Randomly sample if too many voxels
            indices = np.random.choice(len(coords), max_samples_per_class, replace=False)
            coords = coords[indices]

        class_locations[int(c)] = coords

    return class_locations


def preprocess_case(
    case_name: str,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    n_channels: int,
    file_ending: str,
    normalization_schemes: List[str] = None,
    use_mask_for_norm: List[bool] = None,
    intensity_properties: Dict[str, dict] = None
) -> Dict[str, Any]:
    """Preprocess a single case.

    Parameters
    ----------
    case_name : str
        Case identifier
    images_dir : Path
        Directory containing images
    labels_dir : Path
        Directory containing labels
    output_dir : Path
        Output directory
    n_channels : int
        Number of input channels
    file_ending : str
        File extension (e.g., '.nii.gz')
    normalization_schemes : list of str
        Normalization scheme per channel (from nnUNet plans)
    use_mask_for_norm : list of bool
        Whether to use mask for normalization per channel (from nnUNet plans)
    intensity_properties : dict
        Intensity properties per channel for CT normalization

    Returns
    -------
    dict
        Case properties
    """
    # Default normalization settings if not provided
    if normalization_schemes is None:
        normalization_schemes = ['ZScoreNormalization'] * n_channels
    if use_mask_for_norm is None:
        use_mask_for_norm = [False] * n_channels

    # Load all channels
    images = []
    spacing = None

    original_affine = None
    ras_affine = None
    original_nifti_shape = None

    for ch in range(n_channels):
        img_path = images_dir / f"{case_name}_{ch:04d}{file_ending}"
        nii = nib.load(str(img_path))

        # Capture original NIfTI metadata before reorientation (first channel only)
        if ch == 0:
            original_affine = nii.affine.copy()
            original_nifti_shape = nii.shape[:3]

        # Reorient to RAS
        nii = reorient_to_ras(nii)

        img_data = nii.get_fdata().astype(np.float32)
        images.append(img_data)

        # Get spacing and RAS affine from first channel
        if ch == 0:
            spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
            ras_affine = nii.affine.copy()

    # Stack channels: (C, D, H, W)
    image = np.stack(images, axis=0)
    original_shape = image.shape[1:]  # (D, H, W)

    # Load and reorient label
    label_path = labels_dir / f"{case_name}{file_ending}"
    label_nii = nib.load(str(label_path))
    label_nii = reorient_to_ras(label_nii)
    label = label_nii.get_fdata().astype(np.int64)

    # Compute bounding box and nonzero mask (like nnUNet's crop_to_nonzero)
    combined = np.concatenate([image, label[np.newaxis]], axis=0)
    bbox, nonzero_mask = compute_nonzero_bbox(combined)

    # Crop to non-zero region
    image = crop_to_bbox(image, bbox)
    label = crop_to_bbox(label, bbox)
    nonzero_mask = crop_to_bbox(nonzero_mask[np.newaxis], bbox)[0]  # Crop mask too
    cropped_shape = image.shape[1:]  # (D, H, W)

    # Mark outside region in segmentation (like nnUNet)
    # seg == -1 indicates "outside" region for use_mask_for_norm
    seg_for_norm = label.copy()
    seg_for_norm[(label == 0) & (~nonzero_mask)] = -1

    # Normalize each channel according to nnUNet plans
    for c in range(image.shape[0]):
        scheme = normalization_schemes[c] if c < len(normalization_schemes) else 'ZScoreNormalization'
        mask_for_norm = use_mask_for_norm[c] if c < len(use_mask_for_norm) else False
        channel_intensity_props = intensity_properties.get(str(c)) if intensity_properties else None

        normalizer = get_normalizer(scheme, mask_for_norm, channel_intensity_props)
        if normalizer is not None:
            image[c] = normalizer.run(image[c], seg_for_norm)

    # Compute class locations for foreground sampling
    class_locations = compute_class_locations(label)

    # Save preprocessed data
    np.save(output_dir / f"{case_name}.npy", image)
    np.save(output_dir / f"{case_name}_seg.npy", label)

    # Save properties
    properties = {
        'spacing': spacing,
        'shape': cropped_shape,
        'original_shape': original_shape,
        'bbox': bbox,
        'class_locations': class_locations,
        'use_mask_for_norm': use_mask_for_norm,
        'normalization_schemes': normalization_schemes,
        'original_affine': original_affine,
        'ras_affine': ras_affine,
        'original_nifti_shape': original_nifti_shape,
    }

    with open(output_dir / f"{case_name}.pkl", 'wb') as f:
        pickle.dump(properties, f)

    return properties


def load_nnunet_plans(preprocessed_dir: Path, configuration: str = '3d_fullres') -> Tuple[
    List[str], List[bool], Optional[Dict[str, dict]]
]:
    """Load normalization settings from nnUNet plans.

    Parameters
    ----------
    preprocessed_dir : Path
        Path to nnUNet_preprocessed/{dataset_name} directory
    configuration : str
        nnUNet configuration to use (e.g., '3d_fullres', '3d_lowres', '2d')

    Returns
    -------
    normalization_schemes : list of str
        Normalization scheme per channel
    use_mask_for_norm : list of bool
        Whether to use mask for normalization per channel
    intensity_properties : dict or None
        Intensity properties per channel (for CT normalization)
    """
    plans_file = preprocessed_dir / 'nnUNetPlans.json'

    if not plans_file.exists():
        print(f"Warning: nnUNetPlans.json not found at {plans_file}")
        print("Using default ZScoreNormalization with use_mask_for_norm=False")
        return ['ZScoreNormalization'], [False], None

    with open(plans_file) as f:
        plans = json.load(f)

    # Get configuration settings
    configurations = plans.get('configurations', {})
    if configuration not in configurations:
        available = list(configurations.keys())
        print(f"Warning: Configuration '{configuration}' not found. Available: {available}")
        if '3d_fullres' in configurations:
            configuration = '3d_fullres'
        elif available:
            configuration = available[0]
        else:
            return ['ZScoreNormalization'], [False], None

    config = configurations[configuration]

    normalization_schemes = config.get('normalization_schemes', ['ZScoreNormalization'])
    use_mask_for_norm = config.get('use_mask_for_norm', [False])

    # Load intensity properties if available (for CT normalization)
    intensity_properties = None
    dataset_fingerprint_file = preprocessed_dir / 'dataset_fingerprint.json'
    if dataset_fingerprint_file.exists():
        with open(dataset_fingerprint_file) as f:
            fingerprint = json.load(f)
        intensity_properties = fingerprint.get('foreground_intensity_properties_per_channel')

    print(f"Loaded nnUNet plans (configuration: {configuration}):")
    print(f"  normalization_schemes: {normalization_schemes}")
    print(f"  use_mask_for_norm: {use_mask_for_norm}")

    return normalization_schemes, use_mask_for_norm, intensity_properties


def preprocess_dataset(
    nnunet_raw: Path,
    nnunet_preprocessed: Path,
    dataset_name: str,
    num_workers: int = 4,
    verify: bool = True,
    configuration: str = '3d_fullres'
) -> None:
    """Preprocess entire dataset.

    Parameters
    ----------
    nnunet_raw : Path
        Path to nnUNet_raw directory
    nnunet_preprocessed : Path
        Path to nnUNet_preprocessed directory
    dataset_name : str
        Dataset name
    num_workers : int
        Number of parallel workers
    verify : bool
        Whether to verify preprocessed files
    configuration : str
        nnUNet configuration to use for normalization settings
    """
    raw_dir = nnunet_raw / dataset_name
    preprocessed_dir = nnunet_preprocessed / dataset_name
    output_dir = preprocessed_dir / 'irrepunet'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset info
    with open(raw_dir / 'dataset.json') as f:
        dataset_info = json.load(f)

    n_channels = len(dataset_info.get('channel_names', {'0': 'CT'}))
    file_ending = dataset_info.get('file_ending', '.nii.gz')

    # Load normalization settings from nnUNet plans
    normalization_schemes, use_mask_for_norm, intensity_properties = load_nnunet_plans(
        preprocessed_dir, configuration
    )

    # Extend to match number of channels if needed
    while len(normalization_schemes) < n_channels:
        normalization_schemes.append(normalization_schemes[-1])
    while len(use_mask_for_norm) < n_channels:
        use_mask_for_norm.append(use_mask_for_norm[-1])

    # Get case names from images directory
    images_dir = raw_dir / 'imagesTr'
    labels_dir = raw_dir / 'labelsTr'

    # Find all cases (extract unique case names from image files)
    case_names = set()
    for img_file in images_dir.glob(f'*{file_ending}'):
        # Remove channel suffix: CaseName_XXXX.nii.gz -> CaseName
        name = img_file.name.replace(file_ending, '')
        case_name = '_'.join(name.split('_')[:-1])  # Remove _XXXX channel suffix
        case_names.add(case_name)

    case_names = sorted(case_names)
    print(f"Found {len(case_names)} cases to preprocess")

    # Process cases
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    preprocess_case,
                    case_name,
                    images_dir,
                    labels_dir,
                    output_dir,
                    n_channels,
                    file_ending,
                    normalization_schemes,
                    use_mask_for_norm,
                    intensity_properties
                ): case_name
                for case_name in case_names
            }

            for future in tqdm(as_completed(futures), total=len(case_names), desc="Preprocessing"):
                case_name = futures[future]
                try:
                    props = future.result()
                    if verify:
                        # Quick verification
                        assert (output_dir / f"{case_name}.npy").exists()
                        assert (output_dir / f"{case_name}_seg.npy").exists()
                        assert (output_dir / f"{case_name}.pkl").exists()
                except Exception as e:
                    print(f"Error processing {case_name}: {e}")
                    raise
    else:
        for case_name in tqdm(case_names, desc="Preprocessing"):
            props = preprocess_case(
                case_name,
                images_dir,
                labels_dir,
                output_dir,
                n_channels,
                file_ending,
                normalization_schemes,
                use_mask_for_norm,
                intensity_properties
            )

    # Copy splits_final.json to irrepunet directory
    splits_src = preprocessed_dir / 'splits_final.json'
    splits_dst = output_dir / 'splits_final.json'

    if splits_src.exists():
        shutil.copy(splits_src, splits_dst)
        print(f"Copied splits_final.json to {splits_dst}")
    else:
        print(f"Warning: splits_final.json not found at {splits_src}")
        print("You will need to create or copy splits_final.json manually")

    # Save preprocessing info
    preprocess_info = {
        'dataset_name': dataset_name,
        'n_channels': n_channels,
        'n_cases': len(case_names),
        'cases': list(case_names),
        'configuration': configuration,
        'normalization_schemes': normalization_schemes,
        'use_mask_for_norm': use_mask_for_norm,
        'preprocessing_steps': [
            'Reorient to RAS',
            'Crop to non-zero region (with binary_fill_holes)',
            f'Normalization: {normalization_schemes} (use_mask_for_norm={use_mask_for_norm})'
        ]
    }

    with open(output_dir / 'preprocessing_info.json', 'w') as f:
        json.dump(preprocess_info, f, indent=2)

    print(f"\nPreprocessing complete!")
    print(f"Output directory: {output_dir}")
    print(f"Cases processed: {len(case_names)}")


def subsample_array(arr: np.ndarray, axis: int, level: int, has_channel: bool = True) -> np.ndarray:
    """Subsample array by skipping slices along an axis.

    Parameters
    ----------
    arr : np.ndarray
        Input array. Shape (C, D, H, W) if has_channel else (D, H, W)
    axis : int
        Spatial axis to subsample (0, 1, or 2 for D, H, W)
    level : int
        Subsampling factor (2 = skip every other slice, 4 = keep every 4th, etc.)
    has_channel : bool
        Whether array has channel dimension

    Returns
    -------
    np.ndarray
        Subsampled array
    """
    if has_channel:
        axis = axis + 1  # Account for channel dimension

    # Create slice indices
    n_slices = arr.shape[axis]
    indices = list(range(0, n_slices, level))

    # Apply subsampling
    return np.take(arr, indices, axis=axis)


def find_slice_axis(spacing: Tuple[float, float, float]) -> int:
    """Find the slice (z) axis - the "odd one out" axis.

    The slice axis is identified as the axis whose spacing differs most
    from the other two axes. This works for both:
    - Thick-slice data: (0.5, 0.5, 5.0) -> axis 2 is odd one out
    - Thin-slice data: (0.56, 1.0, 1.0) -> axis 0 is odd one out

    Parameters
    ----------
    spacing : tuple of float
        Voxel spacing (D, H, W)

    Returns
    -------
    int
        Index of the slice axis (0, 1, or 2)
    """
    s = np.array(spacing)
    # For each axis, compute how different it is from the other two
    # The "odd one out" has the largest difference
    differences = []
    for i in range(3):
        others = [s[j] for j in range(3) if j != i]
        # Difference from mean of others, normalized
        diff = abs(s[i] - np.mean(others)) / max(np.mean(others), 1e-6)
        differences.append(diff)
    return int(np.argmax(differences))


def find_inplane_axes(spacing: Tuple[float, float, float], tolerance: float = 0.1) -> Optional[Tuple[int, int]]:
    """Find the two in-plane axes with similar spacing.

    Parameters
    ----------
    spacing : tuple of float
        Voxel spacing (D, H, W)
    tolerance : float
        Relative tolerance for considering two spacings as "similar"
        Default 0.1 means 10% difference is acceptable

    Returns
    -------
    tuple of int or None
        Indices of the two in-plane axes, or None if no pair is similar enough
    """
    s = np.array(spacing)

    # Check each pair of axes
    pairs = [(0, 1), (0, 2), (1, 2)]
    best_pair = None
    best_diff = float('inf')

    for i, j in pairs:
        # Relative difference between the two spacings
        mean_spacing = (s[i] + s[j]) / 2
        rel_diff = abs(s[i] - s[j]) / mean_spacing if mean_spacing > 0 else float('inf')

        if rel_diff <= tolerance and rel_diff < best_diff:
            best_diff = rel_diff
            best_pair = (i, j)

    return best_pair


def subsample_inplane(arr: np.ndarray, axes: Tuple[int, int], level: int, has_channel: bool = True) -> np.ndarray:
    """Subsample array by skipping voxels along two in-plane axes.

    Parameters
    ----------
    arr : np.ndarray
        Input array. Shape (C, D, H, W) if has_channel else (D, H, W)
    axes : tuple of int
        Two spatial axes to subsample (e.g., (0, 1) for D and H)
    level : int
        Subsampling factor (2 = skip every other, 4 = keep every 4th, etc.)
    has_channel : bool
        Whether array has channel dimension

    Returns
    -------
    np.ndarray
        Subsampled array
    """
    result = arr
    for axis in axes:
        ax = axis + 1 if has_channel else axis
        n_voxels = result.shape[ax]
        indices = list(range(0, n_voxels, level))
        result = np.take(result, indices, axis=ax)
    return result


def create_subsampled_case(
    case_name: str,
    output_dir: Path,
    max_thickness: float,
    max_inplane: float = None,
    inplane_tolerance: float = 0.1,
    inplane_ratio_limit: float = 1.1
) -> List[Dict[str, Any]]:
    """Create subsampled versions of a preprocessed case.

    Creates subsampled versions in two stages:
    1. Slice-axis downsampling: increases slice thickness
    2. In-plane downsampling: applied to BOTH original and slice-augmented data

    In-plane downsampling is only applied when the resulting in-plane spacing
    is no more than inplane_ratio_limit times the slice thickness.

    Parameters
    ----------
    case_name : str
        Case identifier (must already be preprocessed)
    output_dir : Path
        Directory containing preprocessed data
    max_thickness : float
        Maximum allowed slice thickness in mm. Slice-axis subsampled versions
        are created for skip levels 2, 3, 4, ... until the resulting thickness
        exceeds max_thickness + 1mm.
    max_inplane : float, optional
        Maximum allowed in-plane spacing in mm. If None, uses max_thickness.
    inplane_tolerance : float
        Relative tolerance for determining if two axes have similar spacing
        and should be treated as "in-plane". Default 0.1 (10%).
    inplane_ratio_limit : float
        Maximum ratio of in-plane spacing to slice thickness. In-plane
        downsampling is only applied when new_inplane <= slice_thickness * ratio.
        Default 1.1 (in-plane can be at most 10% coarser than slice thickness).

    Returns
    -------
    list of dict
        Properties for each subsampled version created
    """
    if max_inplane is None:
        max_inplane = min(max_thickness, 1.5)

    # Load original case
    image = np.load(output_dir / f"{case_name}.npy")
    label = np.load(output_dir / f"{case_name}_seg.npy")

    with open(output_dir / f"{case_name}.pkl", 'rb') as f:
        props = pickle.load(f)

    spacing = props['spacing']
    created = []

    # Determine axes: isotropic cases get subsampled along all 3 axes
    is_isotropic = max(spacing) / (min(spacing) + 1e-9) < 1.5
    if is_isotropic:
        skip_axes = [0, 1, 2]
    else:
        skip_axes = [find_slice_axis(spacing)]

    # Track whether the original case has had in-plane processing applied
    original_inplane_done = False

    for skip_axis in skip_axes:
        inplane_axes = tuple(i for i in range(3) if i != skip_axis)

        # === Stage 1: Slice-axis downsampling ===
        # Collect all versions: original + slice-augmented for this skip axis
        # Each entry: (suffix, image, label, spacing)
        versions = [("", image, label, spacing)]

        current_thickness = spacing[skip_axis]
        level = 2

        while True:
            new_thickness = current_thickness * level
            if new_thickness > max_thickness + 1.0:
                break

            n_slices = image.shape[skip_axis + 1]
            if n_slices < level * 2:
                break

            sub_image = subsample_array(image, skip_axis, level, has_channel=True)
            sub_label = subsample_array(label, skip_axis, level, has_channel=False)

            new_spacing = list(spacing)
            new_spacing[skip_axis] = new_thickness
            new_spacing = tuple(new_spacing)

            class_locations = compute_class_locations(sub_label)
            suffix = f"_skip{skip_axis}_{level}x"

            np.save(output_dir / f"{case_name}{suffix}.npy", sub_image)
            np.save(output_dir / f"{case_name}{suffix}_seg.npy", sub_label)

            sub_props = {
                'spacing': new_spacing,
                'shape': sub_image.shape[1:],
                'original_shape': props.get('original_shape'),
                'bbox': props.get('bbox'),
                'class_locations': class_locations,
                'is_subsampled': True,
                'subsample_type': 'slice',
                'skip_axis': skip_axis,
                'skip_level': level,
                'parent': case_name,
            }

            with open(output_dir / f"{case_name}{suffix}.pkl", 'wb') as f:
                pickle.dump(sub_props, f)

            created.append(sub_props)

            # Add to versions list for in-plane processing
            versions.append((suffix, sub_image, sub_label, new_spacing))
            level += 1

        # === Stage 2: In-plane downsampling ===
        # Apply to original and all slice-augmented versions for this skip axis
        ax0, ax1 = inplane_axes

        for base_suffix, base_image, base_label, base_spacing in versions:
            # Skip in-plane processing of the original for all but the first skip axis
            if base_suffix == "" and original_inplane_done:
                continue

            # Get current in-plane and slice spacing for this version
            current_inplane = (base_spacing[ax0] + base_spacing[ax1]) / 2
            slice_thickness = base_spacing[skip_axis]

            level = 2
            while True:
                new_inplane = current_inplane * level

                # Check max_inplane limit
                if new_inplane > max_inplane:
                    break

                # Check ratio limit: in-plane should not exceed slice_thickness * ratio
                if new_inplane > slice_thickness * inplane_ratio_limit:
                    break

                # Check if we have enough voxels in both in-plane dimensions
                n_voxels_0 = base_image.shape[ax0 + 1]
                n_voxels_1 = base_image.shape[ax1 + 1]
                if n_voxels_0 < level * 2 or n_voxels_1 < level * 2:
                    break

                sub_image = subsample_inplane(base_image, inplane_axes, level, has_channel=True)
                sub_label = subsample_inplane(base_label, inplane_axes, level, has_channel=False)

                # Update spacing for both in-plane axes
                new_spacing = list(base_spacing)
                new_spacing[ax0] = base_spacing[ax0] * level
                new_spacing[ax1] = base_spacing[ax1] * level
                new_spacing = tuple(new_spacing)

                class_locations = compute_class_locations(sub_label)

                # Create suffix: combine base suffix with in-plane suffix
                inplane_suffix = f"_skipxy_{level}x"
                full_suffix = base_suffix + inplane_suffix

                np.save(output_dir / f"{case_name}{full_suffix}.npy", sub_image)
                np.save(output_dir / f"{case_name}{full_suffix}_seg.npy", sub_label)

                sub_props = {
                    'spacing': new_spacing,
                    'shape': sub_image.shape[1:],
                    'original_shape': props.get('original_shape'),
                    'bbox': props.get('bbox'),
                    'class_locations': class_locations,
                    'is_subsampled': True,
                    'subsample_type': 'inplane' if base_suffix == "" else 'combined',
                    'inplane_axes': inplane_axes,
                    'skip_level': level,
                    'parent': case_name if base_suffix == "" else f"{case_name}{base_suffix}",
                }

                with open(output_dir / f"{case_name}{full_suffix}.pkl", 'wb') as f:
                    pickle.dump(sub_props, f)

                created.append(sub_props)

                level += 1

            if base_suffix == "":
                original_inplane_done = True

    return created


def create_subsampled_dataset(
    output_dir: Path,
    num_workers: int = 4,
    max_inplane: float = None,
    inplane_tolerance: float = 0.1,
    inplane_ratio_limit: float = 1.1
) -> None:
    """Create subsampled versions of all preprocessed cases.

    Parameters
    ----------
    output_dir : Path
        Directory containing preprocessed irrepunet data
    num_workers : int
        Number of parallel workers
    max_inplane : float, optional
        Maximum in-plane spacing for in-plane downsampled versions.
        If None, uses the same limit as slice thickness.
    inplane_tolerance : float
        Relative tolerance for determining in-plane axes (default 0.1 = 10%)
    inplane_ratio_limit : float
        Maximum ratio of in-plane to slice thickness (default 1.1).
        In-plane downsampling only applied when new_inplane <= slice * ratio.

    Notes
    -----
    Creates subsampled versions in two stages:
    1. Slice-axis: Increases slice thickness up to max_thickness + 1mm
    2. In-plane: Applied to original AND slice-augmented data, constrained by ratio
    """
    # Find all original cases (exclude already subsampled ones)
    case_names = []
    max_thickness = 0.0
    max_existing_inplane = 0.0

    print("Scanning dataset to find spacing statistics...")
    for pkl_file in output_dir.glob('*.pkl'):
        case_name = pkl_file.stem
        # Skip if already a subsampled version
        if '_skip' in case_name:
            continue
        case_names.append(case_name)

        # Load properties to check spacing
        with open(pkl_file, 'rb') as f:
            props = pickle.load(f)
        spacing = props['spacing']

        # Slice thickness is the largest spacing value
        thickness = max(spacing)
        if thickness > max_thickness:
            max_thickness = thickness

        # In-plane spacing (smallest two values)
        sorted_spacing = sorted(spacing)
        inplane = sorted_spacing[0]  # Finest spacing
        if inplane > max_existing_inplane:
            max_existing_inplane = inplane

    case_names = sorted(case_names)

    # Default max_inplane to max_thickness if not specified
    if max_inplane is None:
        max_inplane = min(max_thickness, 1.5)

    print(f"Found {len(case_names)} cases")
    print(f"Maximum slice thickness in dataset: {max_thickness:.2f} mm")
    print(f"Maximum in-plane spacing in dataset: {max_existing_inplane:.2f} mm")
    print(f"Slice thickness limit: {max_thickness + 1.0:.2f} mm")
    print(f"In-plane spacing limit: {max_inplane:.2f} mm")

    total_slice = 0
    total_inplane = 0

    total_combined = 0

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    create_subsampled_case,
                    case_name,
                    output_dir,
                    max_thickness,
                    max_inplane,
                    inplane_tolerance,
                    inplane_ratio_limit
                ): case_name
                for case_name in case_names
            }

            for future in tqdm(as_completed(futures), total=len(case_names), desc="Subsampling"):
                case_name = futures[future]
                try:
                    created = future.result()
                    for c in created:
                        stype = c.get('subsample_type', 'slice')
                        if stype == 'inplane':
                            total_inplane += 1
                        elif stype == 'combined':
                            total_combined += 1
                        else:
                            total_slice += 1
                except Exception as e:
                    print(f"Error processing {case_name}: {e}")
                    raise
    else:
        for case_name in tqdm(case_names, desc="Subsampling"):
            created = create_subsampled_case(
                case_name, output_dir, max_thickness, max_inplane,
                inplane_tolerance, inplane_ratio_limit
            )
            for c in created:
                stype = c.get('subsample_type', 'slice')
                if stype == 'inplane':
                    total_inplane += 1
                elif stype == 'combined':
                    total_combined += 1
                else:
                    total_slice += 1

    total = total_slice + total_inplane + total_combined
    print(f"\nSubsampling complete!")
    print(f"Created {total} subsampled versions:")
    print(f"  - Slice-axis only: {total_slice}")
    print(f"  - In-plane only: {total_inplane}")
    print(f"  - Combined (slice + in-plane): {total_combined}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess data for e3nnUNet (matching nnUNet normalization)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Preprocess command (default)
    preprocess_parser = subparsers.add_parser(
        'preprocess', help='Preprocess raw NIfTI data'
    )
    preprocess_parser.add_argument(
        '--nnunet_raw', type=str, required=True,
        help='Path to nnUNet_raw directory'
    )
    preprocess_parser.add_argument(
        '--nnunet_preprocessed', type=str, required=True,
        help='Path to nnUNet_preprocessed directory'
    )
    preprocess_parser.add_argument(
        '--dataset', type=str, required=True,
        help='Dataset name (e.g., Dataset555_CSF01)'
    )
    preprocess_parser.add_argument(
        '--configuration', type=str, default='3d_fullres',
        help='nnUNet configuration to use for normalization settings'
    )
    preprocess_parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of parallel workers'
    )
    preprocess_parser.add_argument(
        '--no_verify', action='store_true',
        help='Skip verification of preprocessed files'
    )

    # Subsample command
    subsample_parser = subparsers.add_parser(
        'subsample', help='Create subsampled versions of preprocessed data'
    )
    subsample_parser.add_argument(
        '--preprocessed_dir', type=str, required=True,
        help='Path to irrepunet preprocessed directory'
    )
    subsample_parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of parallel workers'
    )
    subsample_parser.add_argument(
        '--max_inplane', type=float, default=None,
        help='Maximum in-plane spacing (mm) for in-plane downsampling. '
             'Default: same as max slice thickness in dataset.'
    )
    subsample_parser.add_argument(
        '--inplane_tolerance', type=float, default=0.1,
        help='Relative tolerance for identifying in-plane axes (default: 0.1 = 10%%)'
    )
    subsample_parser.add_argument(
        '--inplane_ratio_limit', type=float, default=1.1,
        help='Max ratio of in-plane to slice thickness (default: 1.1). '
             'In-plane augmentation only applied when new_inplane <= slice * ratio.'
    )

    # For backwards compatibility, also support old-style arguments
    parser.add_argument('--nnunet_raw', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--nnunet_preprocessed', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--dataset', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--configuration', type=str, default='3d_fullres', help=argparse.SUPPRESS)
    parser.add_argument('--num_workers', type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument('--no_verify', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Handle backwards compatibility (no subcommand specified)
    if args.command is None and args.nnunet_raw and args.dataset:
        preprocess_dataset(
            nnunet_raw=Path(args.nnunet_raw),
            nnunet_preprocessed=Path(args.nnunet_preprocessed),
            dataset_name=args.dataset,
            num_workers=args.num_workers,
            verify=not args.no_verify,
            configuration=args.configuration
        )
    elif args.command == 'preprocess':
        preprocess_dataset(
            nnunet_raw=Path(args.nnunet_raw),
            nnunet_preprocessed=Path(args.nnunet_preprocessed),
            dataset_name=args.dataset,
            num_workers=args.num_workers,
            verify=not args.no_verify,
            configuration=args.configuration
        )
    elif args.command == 'subsample':
        create_subsampled_dataset(
            output_dir=Path(args.preprocessed_dir),
            num_workers=args.num_workers,
            max_inplane=args.max_inplane,
            inplane_tolerance=args.inplane_tolerance,
            inplane_ratio_limit=args.inplane_ratio_limit
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
