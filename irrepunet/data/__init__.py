"""Data loading utilities for e3nnUNet."""

from .spacing import (
    SPACING_GRID, group_cases_by_spacing, get_canonical_permutation,
    apply_axis_permutation, round_to_grid, round_spacing_to_tolerance,
)
from .multi_resolution_loader import (
    estimate_memory_mb, estimate_batch_size,
    discover_skip_files,
    compute_steps_through_pooling, verify_receptive_field,
    adjust_for_divisibility_per_dim, mm_to_voxels,
)

# PyTorch-dependent imports are optional (not available in JAX-only environments)
try:
    from .batchgen_dataset import E3nnDataset, E3nnDataLoader
    from .batchgen_transforms import (
        get_training_transforms, get_validation_transforms,
    )
    from .dataloader import MultiResolutionLoader
except ImportError:
    pass

__all__ = [
    # Spacing utilities
    "SPACING_GRID",
    "group_cases_by_spacing",
    "get_canonical_permutation",
    "apply_axis_permutation",
    "round_to_grid",
    "round_spacing_to_tolerance",
    # Batchgenerators dataset/loaders
    "E3nnDataset",
    "E3nnDataLoader",
    # Batchgenerators transforms
    "get_training_transforms",
    "get_validation_transforms",
    # Multi-resolution loader
    "MultiResolutionLoader",
    "estimate_memory_mb",
    "estimate_batch_size",
    "discover_skip_files",
    "compute_steps_through_pooling",
    "verify_receptive_field",
    "adjust_for_divisibility_per_dim",
    "mm_to_voxels",
]
