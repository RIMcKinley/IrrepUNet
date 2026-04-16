"""Dataset and DataLoader classes compatible with batchgenerators.

Provides E3nnDataset (case loader) and E3nnDataLoader (foreground-oversampling loader).
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader as BGDataLoader

from .spacing import get_canonical_permutation, apply_axis_permutation
from .decimation import decimate_array, random_offsets, compute_class_locations


class E3nnDataset:
    """Dataset class compatible with batchgenerators DataLoader.

    Loads preprocessed .npy files and provides class_locations for foreground sampling.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        case_identifiers: List[str],
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.identifiers = case_identifiers

        # Load properties for all cases
        self.properties: Dict[str, dict] = {}
        for case_name in case_identifiers:
            pkl_path = self.preprocessed_dir / f"{case_name}.pkl"
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    self.properties[case_name] = pickle.load(f)

    def load_case(self, case_identifier: str):
        """Load a single case.

        Synthetic decimated cases (``decimation_strides`` in properties) load
        the base source's .npy and apply ``np.take`` on the fly.

        Returns:
            data: np.ndarray of shape (C, D, H, W)
            seg: np.ndarray of shape (1, D, H, W)
            seg_prev: None (no cascade)
            properties: dict with spacing, class_locations, etc.
        """
        properties = self.properties.get(case_identifier, {})
        strides = properties.get('decimation_strides')
        if strides and any(s > 1 for s in strides):
            base_id = properties['decimation_base_id']
            data = np.load(self.preprocessed_dir / f"{base_id}.npy")
            seg = np.load(self.preprocessed_dir / f"{base_id}_seg.npy")
            offsets = random_offsets(strides)
            data = decimate_array(data, strides, has_channel=True, offsets=offsets)
            seg = decimate_array(seg, strides, has_channel=False, offsets=offsets)
            # Per-sample random offset: class_locations must be recomputed from
            # the freshly decimated seg (cheap: decimated shapes are small).
            properties = {
                **properties,
                'class_locations': compute_class_locations(seg),
            }
        else:
            data = np.load(self.preprocessed_dir / f"{case_identifier}.npy")
            seg = np.load(self.preprocessed_dir / f"{case_identifier}_seg.npy")

        # Add channel dimension to seg: (D, H, W) -> (1, D, H, W)
        seg = seg[np.newaxis, ...]

        return data, seg, None, properties

    def __len__(self):
        return len(self.identifiers)


class E3nnDataLoader(BGDataLoader):
    """DataLoader with foreground oversampling for e3nnUNet.

    Based on nnUNet's nnUNetDataLoader but simplified.

    Parameters
    ----------
    data : E3nnDataset
        Dataset to load from
    batch_size : int
        Batch size
    patch_size : tuple
        Output patch size in voxels
    oversample_foreground_percent : float
        Fraction of samples guaranteed to contain foreground
    probabilistic_oversampling : bool
        If True, use probabilistic oversampling; else deterministic
    num_threads_in_multithreaded : int
        Number of threads for multi-threaded loading
    """

    def __init__(
        self,
        data: E3nnDataset,
        batch_size: int,
        patch_size: Tuple[int, ...],
        oversample_foreground_percent: float = 0.33,
        probabilistic_oversampling: bool = True,
        num_threads_in_multithreaded: int = 1,
    ):
        super().__init__(
            data, batch_size, num_threads_in_multithreaded,
            seed_for_shuffle=None, return_incomplete=False,
            shuffle=True, infinite=True
        )

        self.indices = data.identifiers
        self.patch_size = np.array(patch_size)  # Output patch size
        self.oversample_foreground_percent = oversample_foreground_percent
        self.probabilistic_oversampling = probabilistic_oversampling

        self.extract_patch_size = np.array(patch_size).copy()

        # Determine data shapes
        sample_data, sample_seg, _, _ = data.load_case(data.identifiers[0])
        self.num_channels = sample_data.shape[0]
        self.num_seg_channels = 1

        self.data_shape = (batch_size, self.num_channels, *patch_size)
        self.seg_shape = (batch_size, self.num_seg_channels, *patch_size)

    def get_do_oversample(self, sample_idx: int) -> bool:
        """Determine whether this sample should have guaranteed foreground."""
        if self.probabilistic_oversampling:
            return np.random.uniform() < self.oversample_foreground_percent
        else:
            # Deterministic: last X% of batch gets foreground
            return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: Optional[Dict] = None
    ) -> Tuple[List[int], List[int]]:
        """Get bounding box for patch extraction.

        If force_fg=True, centers patch on a foreground voxel.
        Uses extract_patch_size (which may be larger than output patch_size
        if slice skipping is enabled).
        """
        dim = len(data_shape)

        # Compute valid range for bbox lower bounds (use extraction size)
        lbs = [0] * dim
        ubs = [max(0, data_shape[i] - self.extract_patch_size[i]) for i in range(dim)]

        if force_fg and class_locations is not None:
            # Find a foreground voxel to center on
            eligible_classes = [k for k in class_locations.keys() if len(class_locations[k]) > 0]

            if eligible_classes:
                selected_class = np.random.choice(eligible_classes)
                voxels = class_locations[selected_class]
                selected_voxel = voxels[np.random.choice(len(voxels))]

                # Center patch on selected voxel (use extraction size)
                bbox_lbs = []
                for i in range(dim):
                    lb = max(lbs[i], selected_voxel[i] - self.extract_patch_size[i] // 2)
                    lb = min(lb, ubs[i])  # Ensure within bounds
                    bbox_lbs.append(int(lb))
            else:
                # No foreground found, random crop
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) if ubs[i] > lbs[i] else lbs[i] for i in range(dim)]
        else:
            # Random crop
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) if ubs[i] > lbs[i] else lbs[i] for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.extract_patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        """Generate a training batch with foreground oversampling."""
        selected_keys = self.get_indices()

        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        spacings_all = []  # Per-sample spacings (canonical sorted form)

        for j, case_id in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)

            data, seg, _, properties = self._data.load_case(case_id)
            original_spacing = properties.get('spacing', (1.0, 1.0, 1.0))
            class_locations = properties.get('class_locations', {})

            # Apply axis permutation to canonical (sorted) form
            perm = get_canonical_permutation(original_spacing)
            # Only permute if not identity
            if perm != (0, 1, 2):
                data = apply_axis_permutation(data, perm, has_channel=True)
                seg = apply_axis_permutation(seg, perm, has_channel=True)
                # Permute class_locations coordinates
                class_locations = self._permute_class_locations(class_locations, perm)
            # Store canonical (sorted) spacing
            canonical_spacing = tuple(original_spacing[i] for i in perm)
            spacings_all.append(canonical_spacing)

            shape = data.shape[1:]  # (D, H, W) - now in canonical orientation
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, class_locations)

            # Extract patch
            slices = tuple(slice(lb, ub) for lb, ub in zip(bbox_lbs, bbox_ubs))
            data_patch = data[(slice(None),) + slices]
            seg_patch = seg[(slice(None),) + slices]

            # Pad if needed (when image is smaller than patch)
            if data_patch.shape[1:] != tuple(self.patch_size):
                data_patch = self._pad_to_size(data_patch, (self.num_channels,) + tuple(self.patch_size))
                seg_patch = self._pad_to_size(seg_patch, (1,) + tuple(self.patch_size))

            data_all[j] = data_patch
            seg_all[j] = seg_patch

        return {'data': data_all, 'seg': seg_all, 'keys': selected_keys, 'spacings': spacings_all}

    def _pad_to_size(self, arr: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Pad array to target shape."""
        pad_width = []
        for current, target in zip(arr.shape, target_shape):
            diff = target - current
            pad_width.append((0, max(0, diff)))
        return np.pad(arr, pad_width, mode='constant', constant_values=0)

    def _permute_class_locations(
        self,
        class_locations: Dict[int, np.ndarray],
        perm: tuple
    ) -> Dict[int, np.ndarray]:
        """Permute foreground coordinates to match axis permutation.

        Parameters
        ----------
        class_locations : dict
            Mapping from class index to array of coordinates (N, 3)
        perm : tuple
            Permutation indices for spatial dimensions

        Returns
        -------
        dict
            Class locations with permuted coordinates
        """
        permuted = {}
        for cls, coords in class_locations.items():
            if len(coords) > 0:
                # coords is (N, 3) array, permute the columns
                permuted[cls] = coords[:, perm]
            else:
                permuted[cls] = coords
        return permuted

