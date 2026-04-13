"""JAX adapter for the batchgenerators data pipeline."""

import numpy as np


class NumpyToJax:
    """Replace NumpyToTensor as the final transform.

    Remaps keys: 'data' -> 'image', 'seg' -> 'label' (squeezed, int32).

    Data stays as NumPy arrays to avoid JAX CUDA conflicts in forked
    DataLoader workers. JAX will convert them lazily when they enter
    JIT-compiled functions.
    """

    def __call__(self, **data_dict):
        data_dict['image'] = data_dict.pop('data')
        if 'seg' in data_dict:
            seg = data_dict.pop('seg')
            # (B, 1, D, H, W) -> (B, D, H, W) int32
            data_dict['label'] = np.squeeze(seg, axis=1).astype(np.int32)
        return data_dict


def get_training_transforms_jax(patch_size, disable_mirroring=False, disable_spatial=False,
                                disable_low_res_sim=True, bias_field=True):
    """Identical to get_training_transforms() but with NumpyToJax instead of NumpyToTensor."""
    from irrepunet.data.batchgen_transforms import get_training_transforms

    compose = get_training_transforms(patch_size, disable_mirroring, disable_spatial,
                                      disable_low_res_sim=disable_low_res_sim,
                                      bias_field=bias_field)
    # Replace last transform (NumpyToTensor) with NumpyToJax
    compose.transforms[-1] = NumpyToJax()
    return compose


def get_validation_transforms_jax():
    """Validation transforms with NumpyToJax output."""
    from batchgenerators.transforms.abstract_transforms import Compose
    return Compose([NumpyToJax()])
