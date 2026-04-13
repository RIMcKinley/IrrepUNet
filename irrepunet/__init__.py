"""
e3nnUNet: Equivariant 3D UNet for Medical Image Segmentation

A modular framework combining e3nn equivariant neural networks with
UNet architecture for 3D medical image segmentation.
"""

__version__ = "0.1.0"

# PyTorch-based imports are optional (not available in JAX-only environments)
try:
    from .models import E3nnUNet
    from .inference import sliding_window_inference, predict_nifti
except ImportError:
    pass

__all__ = ["E3nnUNet", "sliding_window_inference", "predict_nifti"]
