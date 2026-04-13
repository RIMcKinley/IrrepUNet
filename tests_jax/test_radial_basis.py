"""Cross-validate JAX radial basis against e3nn."""

import pytest
import numpy as np
import jax.numpy as jnp
import torch

from e3nn.math import soft_one_hot_linspace as e3nn_soft_one_hot_linspace
from e3nn.math import soft_unit_step as e3nn_soft_unit_step

from irrepunet.models_jax.radial_basis import soft_one_hot_linspace, soft_unit_step


class TestSoftUnitStep:
    """Test soft_unit_step matches e3nn."""

    def test_positive_values(self):
        x_np = np.linspace(0.01, 5.0, 100)
        jax_out = np.array(soft_unit_step(jnp.array(x_np)))
        torch_out = e3nn_soft_unit_step(torch.tensor(x_np)).numpy()
        np.testing.assert_allclose(jax_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_negative_values(self):
        x_np = np.linspace(-5.0, -0.01, 100)
        jax_out = np.array(soft_unit_step(jnp.array(x_np)))
        torch_out = e3nn_soft_unit_step(torch.tensor(x_np)).numpy()
        np.testing.assert_allclose(jax_out, torch_out, atol=1e-7)

    def test_zero(self):
        jax_out = float(soft_unit_step(jnp.array(0.0)))
        torch_out = float(e3nn_soft_unit_step(torch.tensor(0.0)))
        assert jax_out == pytest.approx(torch_out, abs=1e-7)

    def test_mixed(self):
        x_np = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0])
        jax_out = np.array(soft_unit_step(jnp.array(x_np)))
        torch_out = e3nn_soft_unit_step(torch.tensor(x_np)).numpy()
        np.testing.assert_allclose(jax_out, torch_out, rtol=1e-5, atol=1e-7)


class TestSoftOneHotLinspace:
    """Test soft_one_hot_linspace matches e3nn."""

    @pytest.mark.parametrize("number", [3, 4, 8])
    @pytest.mark.parametrize("cutoff", [True, False])
    def test_smooth_finite(self, number, cutoff):
        x_np = np.linspace(0, 5.0, 50)
        start, end = 0.0, 5.0

        jax_out = np.array(soft_one_hot_linspace(
            jnp.array(x_np), start, end, number, 'smooth_finite', cutoff
        ))
        torch_out = e3nn_soft_one_hot_linspace(
            torch.tensor(x_np), start, end, number, 'smooth_finite', cutoff
        ).numpy()

        np.testing.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-5)

    def test_output_shape(self):
        x = jnp.linspace(0, 5.0, 20)
        out = soft_one_hot_linspace(x, 0.0, 5.0, 6, 'smooth_finite', True)
        assert out.shape == (20, 6)

    def test_lattice_norm_input(self):
        """Test with typical lattice norm values (as used in VoxelConvolution)."""
        # Simulate lattice norms from a 5x5x5 grid at 1.0 spacing
        x = jnp.arange(-2, 3.0)
        gx, gy, gz = jnp.meshgrid(x, x, x, indexing='ij')
        lattice = jnp.stack([gx, gy, gz], axis=-1)
        norms = jnp.sqrt(jnp.sum(lattice ** 2, axis=-1)).reshape(-1)

        jax_out = np.array(soft_one_hot_linspace(
            norms, 0.0, 2.5, 4, 'smooth_finite', True
        ))

        # Compare with e3nn
        torch_norms = torch.tensor(np.array(norms))
        torch_out = e3nn_soft_one_hot_linspace(
            torch_norms, 0.0, 2.5, 4, 'smooth_finite', True
        ).numpy()

        np.testing.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-5)
