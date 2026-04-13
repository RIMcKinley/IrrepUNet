"""Test fixtures for JAX model tests."""

import sys
import os
import pytest

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Remove vendored e3nn from path (same as tests/conftest.py)
e3nn_dir = os.path.join(project_root, 'e3nn')
if e3nn_dir in sys.path:
    sys.path.remove(e3nn_dir)


# Common irreps configurations for parametrized tests
SMALL_CONFIGS = [
    pytest.param("4x0e", "4x0e", 0, id="scalar-only-l0"),
    pytest.param("4x0e + 4x1o", "4x0e + 4x1o", 2, id="mixed-l2"),
]

UNET_CONFIGS = [
    pytest.param(2, 2, 2, (4, 2, 1), id="base2-down2-l2"),
    pytest.param(2, 1, 1, (4,), id="base2-down1-l0-scalar"),
]
