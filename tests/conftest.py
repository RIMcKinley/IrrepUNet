"""Shared fixtures and configuration for irrepunet tests."""

import sys
import pytest
import torch

# Ensure the project root is on the Python path
sys.path.insert(0, "/data/disk2/projects/diaz-testing")


@pytest.fixture
def device():
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def small_model_kwargs():
    """Minimal model kwargs for fast tests."""
    return dict(
        n_classes=2,
        in_channels=1,
        n_base_filters=1,
        n_downsample=2,
        lmax=1,
        diameter=5.0,
        num_radial_basis=3,
        spacing=(1.0, 1.0, 1.0),
        equivariance="SO3",
        normalization="instance",
        dropout_prob=0.0,
        cutoff=True,
        deep_supervision=False,
        max_features=320,
        irrep_ratios=(4, 2, 1),
    )


def skip_no_cuda():
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
