"""Training and validation transforms using batchgenerators (nnUNet-style).

These are the transforms used by the batchgenerators-based multi-resolution
data pipeline.
"""

from typing import Tuple

import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor


class BiasFieldTransform(AbstractTransform):
    """Random MRI bias field augmentation via torchio.

    Applies a smooth, low-frequency multiplicative field to simulate
    B0 inhomogeneity. Only modifies 'data', not 'seg'.

    Parameters
    ----------
    coefficients : float
        Maximum magnitude of polynomial coefficients (default 0.5).
    order : int
        Order of the polynomial basis (default 3).
    p_per_sample : float
        Probability of applying to each sample in the batch.
    """

    def __init__(self, coefficients: float = 0.5, order: int = 3,
                 p_per_sample: float = 0.3):
        self.coefficients = coefficients
        self.order = order
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        import torchio as tio
        data = data_dict['data']  # (B, C, D, H, W)
        for b in range(data.shape[0]):
            if np.random.random() < self.p_per_sample:
                img = tio.ScalarImage(tensor=data[b])
                subj = tio.Subject(image=img)
                tf = tio.RandomBiasField(
                    coefficients=self.coefficients, order=self.order)
                data[b] = tf(subj)['image'].data.numpy()
        data_dict['data'] = data
        return data_dict


def get_training_transforms(
    patch_size: Tuple[int, ...],
    disable_mirroring: bool = False,
    disable_spatial: bool = False,
    disable_low_res_sim: bool = True,
    bias_field: bool = True,
) -> Compose:
    """Get nnUNet-style training transforms."""
    transforms = []

    # Spatial transforms
    if not disable_spatial:
        transforms.append(
            SpatialTransform(
                patch_size,
                patch_center_dist_from_border=[i // 2 for i in patch_size],
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=(-30 / 180 * np.pi, 30 / 180 * np.pi),
                angle_y=(-30 / 180 * np.pi, 30 / 180 * np.pi),
                angle_z=(-30 / 180 * np.pi, 30 / 180 * np.pi),
                p_rot_per_axis=1,
                do_scale=True,
                scale=(0.7, 1.4),
                border_mode_data='constant',
                border_mode_seg='constant',
                order_seg=0,
                order_data=3,
                random_crop=True,  # Allow variable input sizes from different resolution groups
                p_el_per_sample=0,
                p_rot_per_sample=0.2,
                p_scale_per_sample=0.2,
            )
        )

    # Gaussian noise
    transforms.append(
        GaussianNoiseTransform(
            noise_variance=(0, 0.1),
            p_per_sample=0.1,
            p_per_channel=1.0,
        )
    )

    # Gaussian blur
    transforms.append(
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.0),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5,
        )
    )

    # Brightness
    transforms.append(
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.75, 1.25),
            p_per_sample=0.15,
        )
    )

    # Contrast
    transforms.append(
        ContrastAugmentationTransform(
            contrast_range=(0.75, 1.25),
            p_per_sample=0.15,
        )
    )

    # Random MRI bias field (smooth multiplicative inhomogeneity)
    if bias_field:
        transforms.append(
            BiasFieldTransform(
                coefficients=0.5,
                order=3,
                p_per_sample=0.3,
            )
        )

    # Simulate low resolution (disabled by default — resampling the image
    # without updating the spacing breaks the physical-unit kernel model)
    if not disable_low_res_sim:
        transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.5, 1.0),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.25,
            )
        )

    # Gamma transforms
    transforms.append(
        GammaTransform(
            gamma_range=(0.7, 1.5),
            invert_image=True,
            per_channel=True,
            retain_stats=True,
            p_per_sample=0.1,
        )
    )
    transforms.append(
        GammaTransform(
            gamma_range=(0.7, 1.5),
            invert_image=False,
            per_channel=True,
            retain_stats=True,
            p_per_sample=0.3,
        )
    )

    # Mirroring
    if not disable_mirroring:
        transforms.append(
            MirrorTransform(axes=(0, 1, 2))
        )

    # Convert to tensor
    transforms.append(NumpyToTensor(keys=['data', 'seg'], cast_to='float'))

    return Compose(transforms)


def get_validation_transforms() -> Compose:
    """Get validation transforms (minimal, just convert to tensor)."""
    return Compose([
        NumpyToTensor(keys=['data', 'seg'], cast_to='float')
    ])
