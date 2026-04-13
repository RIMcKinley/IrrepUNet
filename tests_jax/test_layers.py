"""Layer-by-layer cross-validation against PyTorch."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import torch
import flax.nnx as nnx

from e3nn import o3
from e3nn.o3 import Irreps

import cuequivariance as cue
from irrepunet.models.layers import (
    VoxelConvolution as TorchVoxelConvolution,
    EquivariantPool3d as TorchPool3d,
    EquivariantLayerNorm as TorchLayerNorm,
    _precompute_sc_layout as torch_sc_layout,
    _build_sc_weight_matrix as torch_sc_weight_matrix,
)

from irrepunet.models_jax.layers import (
    VoxelConvolution as JaxVoxelConvolution,
    EquivariantPool3d as JaxPool3d,
    EquivariantLayerNorm as JaxLayerNorm,
    EquivariantGate as JaxGate,
    _precompute_sc_layout as jax_sc_layout,
    _build_sc_weight_matrix as jax_sc_weight_matrix,
    _parse_irreps,
    _spherical_harmonics_numpy,
    _soft_one_hot_linspace_numpy,
    _compute_lattice_buffers_numpy,
)


class TestVoxelConvolutionSpacing:
    """Test VoxelConvolution.update_spacing()."""

    def test_update_spacing_changes_lattice(self):
        """Lattice shape changes when spacing changes."""
        conv = JaxVoxelConvolution(
            "4x0e", "4x0e", "1x0e + 1x1o + 1x2e",
            diameter=5.0, num_radial_basis=4,
            steps=(1.0, 1.0, 1.0), rngs=nnx.Rngs(0),
        )
        orig_shape = conv.lattice[...].shape

        conv.update_spacing((2.0, 2.0, 2.0))
        new_shape = conv.lattice[...].shape

        # Larger spacing -> fewer lattice points
        assert new_shape != orig_shape
        assert all(n <= o for n, o in zip(new_shape[:3], orig_shape[:3]))

    def test_update_spacing_preserves_params(self):
        """Learnable parameters are unchanged after update_spacing."""
        conv = JaxVoxelConvolution(
            "4x0e", "4x0e", "1x0e + 1x1o + 1x2e",
            diameter=5.0, num_radial_basis=4,
            steps=(1.0, 1.0, 1.0), rngs=nnx.Rngs(0),
        )
        weight_before = np.array(conv.weight[...])
        sc_weight_before = np.array(conv.sc_weight[...])

        conv.update_spacing((2.0, 2.0, 2.0))

        np.testing.assert_array_equal(np.array(conv.weight[...]), weight_before)
        np.testing.assert_array_equal(np.array(conv.sc_weight[...]), sc_weight_before)

    def test_kernel_matches_fresh_construction(self):
        """Kernel after update_spacing matches a fresh model built at that spacing."""
        irreps_in = "4x0e"
        irreps_out = "4x0e"
        irreps_sh = "1x0e + 1x1o + 1x2e"
        diameter = 5.0
        num_radial_basis = 4
        new_steps = (2.0, 2.0, 2.0)

        # Build at (1,1,1), then update to (2,2,2)
        conv_updated = JaxVoxelConvolution(
            irreps_in, irreps_out, irreps_sh,
            diameter, num_radial_basis,
            steps=(1.0, 1.0, 1.0), rngs=nnx.Rngs(0),
        )
        conv_updated.update_spacing(new_steps)

        # Build fresh at (2,2,2) with same weights
        conv_fresh = JaxVoxelConvolution(
            irreps_in, irreps_out, irreps_sh,
            diameter, num_radial_basis,
            steps=new_steps, rngs=nnx.Rngs(0),
        )
        # Copy weights from updated to fresh
        conv_fresh.weight.value = conv_updated.weight[...]
        conv_fresh.sc_weight.value = conv_updated.sc_weight[...]

        k_updated = np.array(conv_updated.kernel())
        k_fresh = np.array(conv_fresh.kernel())

        np.testing.assert_allclose(k_updated, k_fresh, rtol=1e-5, atol=1e-6)


CONV_CONFIGS = [
    pytest.param("4x0e", "4x0e", 2, 5.0, 4, id="scalar-l2"),
    pytest.param("4x0e + 4x1o", "4x0e + 4x1o", 2, 5.0, 4, id="mixed-l2"),
    pytest.param("2x0e + 2x1o + 2x2e", "2x0e + 2x1o + 2x2e", 2, 5.0, 4, id="full-l2"),
]


class TestSCLayout:
    """Test self-connection layout matches between PyTorch and JAX."""

    @pytest.mark.parametrize("irreps_in,irreps_out", [
        ("4x0e", "4x0e"),
        ("4x0e + 4x1o", "4x0e + 4x1o"),
        ("2x0e + 2x1o", "4x0e + 4x1o + 2x2e"),
    ])
    def test_layout_matches(self, irreps_in, irreps_out):
        e3nn_in = Irreps(irreps_in)
        e3nn_out = Irreps(irreps_out)
        cue_in = _parse_irreps(irreps_in)
        cue_out = _parse_irreps(irreps_out)

        torch_layout = torch_sc_layout(e3nn_in, e3nn_out)
        jax_layout_result = jax_sc_layout(cue_in, cue_out)

        assert len(torch_layout) == len(jax_layout_result)
        for t, j in zip(torch_layout, jax_layout_result):
            assert t == pytest.approx(j, abs=1e-7)

    @pytest.mark.parametrize("irreps_in,irreps_out", [
        ("4x0e", "4x0e"),
        ("4x0e + 4x1o", "4x0e + 4x1o"),
    ])
    def test_weight_matrix_matches(self, irreps_in, irreps_out):
        e3nn_in = Irreps(irreps_in)
        e3nn_out = Irreps(irreps_out)
        cue_in = _parse_irreps(irreps_in)
        cue_out = _parse_irreps(irreps_out)

        layout = torch_sc_layout(e3nn_in, e3nn_out)
        jax_layout = jax_sc_layout(cue_in, cue_out)

        # Random weights
        np.random.seed(42)
        total_w = layout[-1][0] + layout[-1][3] * layout[-1][4] if layout else 0
        w_np = np.random.randn(total_w).astype(np.float32)

        torch_mat = torch_sc_weight_matrix(
            torch.tensor(w_np), layout, e3nn_out.dim, e3nn_in.dim
        ).numpy()
        jax_mat = np.array(jax_sc_weight_matrix(
            jnp.array(w_np), jax_layout,
            sum(m * ir.dim for m, ir in cue_out),
            sum(m * ir.dim for m, ir in cue_in),
        ))

        np.testing.assert_allclose(jax_mat, torch_mat, rtol=1e-5, atol=1e-6)


class TestVoxelConvolutionKernel:
    """Test JAX VoxelConvolution kernel matches PyTorch."""

    @pytest.mark.parametrize("irreps_in,irreps_out,lmax,diameter,num_radial_basis", CONV_CONFIGS)
    def test_kernel_matches(self, irreps_in, irreps_out, lmax, diameter, num_radial_basis):
        """Kernel output matches when weights are identical."""
        e3nn_sh = o3.Irreps.spherical_harmonics(lmax, 1)
        steps = (1.0, 1.0, 1.0)

        # Build PyTorch conv
        torch.manual_seed(42)
        torch_conv = TorchVoxelConvolution(
            irreps_in, irreps_out, e3nn_sh,
            diameter, num_radial_basis, steps,
        )

        # Build JAX conv
        jax_conv = JaxVoxelConvolution(
            irreps_in, irreps_out,
            str(e3nn_sh),
            diameter, num_radial_basis, steps,
            rngs=nnx.Rngs(0),
        )

        # Transfer weights from PyTorch to JAX
        jax_conv.weight.value = jnp.array(torch_conv.weight.detach().numpy())
        jax_conv.sc_weight.value = jnp.array(torch_conv.sc.weight.detach().numpy().reshape(-1))

        # Compare kernels
        with torch.no_grad():
            torch_kernel = torch_conv.kernel().numpy()
        jax_kernel = np.array(jax_conv.kernel())

        np.testing.assert_allclose(
            jax_kernel, torch_kernel, rtol=1e-3, atol=1e-4,
            err_msg=f"Kernel mismatch for {irreps_in} -> {irreps_out}"
        )


class TestVoxelConvolutionForward:
    """Test JAX VoxelConvolution forward pass matches PyTorch."""

    @pytest.mark.parametrize("irreps_in,irreps_out,lmax,diameter,num_radial_basis", CONV_CONFIGS)
    def test_forward_matches(self, irreps_in, irreps_out, lmax, diameter, num_radial_basis):
        """Forward pass matches when weights are identical."""
        e3nn_sh = o3.Irreps.spherical_harmonics(lmax, 1)
        e3nn_in = Irreps(irreps_in)
        steps = (1.0, 1.0, 1.0)

        # Build PyTorch conv
        torch.manual_seed(42)
        torch_conv = TorchVoxelConvolution(
            irreps_in, irreps_out, e3nn_sh,
            diameter, num_radial_basis, steps,
        )
        torch_conv.eval()

        # Build JAX conv
        jax_conv = JaxVoxelConvolution(
            irreps_in, irreps_out,
            str(e3nn_sh),
            diameter, num_radial_basis, steps,
            rngs=nnx.Rngs(0),
        )

        # Transfer weights
        jax_conv.weight.value = jnp.array(torch_conv.weight.detach().numpy())
        jax_conv.sc_weight.value = jnp.array(torch_conv.sc.weight.detach().numpy().reshape(-1))

        # Random input
        np.random.seed(123)
        x_np = np.random.randn(1, e3nn_in.dim, 8, 8, 8).astype(np.float32)

        with torch.no_grad():
            torch_out = torch_conv(torch.tensor(x_np)).numpy()
        jax_out = np.array(jax_conv(jnp.array(x_np)))

        # Relaxed tolerance: conv_general_dilated in XLA and F.conv3d in PyTorch
        # use different FP accumulation orders, causing ~1e-3 absolute diffs
        # that are flaky across runs.
        np.testing.assert_allclose(
            jax_out, torch_out, rtol=5e-3, atol=2e-3,
            err_msg=f"Forward mismatch for {irreps_in} -> {irreps_out}"
        )


class TestEquivariantGate:
    """Test EquivariantGate output shape and basic behavior."""

    def test_scalar_only(self):
        gate = JaxGate(
            "4x0e", jax.nn.relu,
            "0x0e", None, "0x0e",
        )
        x = jnp.ones((2, 8, 8, 8, 4))
        y = gate(x)
        assert y.shape == (2, 8, 8, 8, 4)

    def test_with_gated(self):
        gate = JaxGate(
            "4x0e", jax.nn.relu,
            "4x0e", jax.nn.sigmoid,
            "4x1o",
        )
        x = jnp.ones((2, 8, 8, 8, gate.in_dim))
        y = gate(x)
        assert y.shape == (2, 8, 8, 8, gate.out_dim)

    def test_relu_applied(self):
        gate = JaxGate(
            "4x0e", jax.nn.relu,
            "0x0e", None, "0x0e",
        )
        x = jnp.array([-1.0, -0.5, 0.0, 0.5])
        y = gate(x)
        # Gate applies normalize2mom: relu(x) * factor where factor ≈ sqrt(2)
        expected = jax.nn.relu(x) * gate._scalar_act_factor
        np.testing.assert_allclose(np.array(y), np.array(expected), atol=1e-7)


class TestEquivariantLayerNorm:
    """Test EquivariantLayerNorm matches PyTorch."""

    @pytest.mark.parametrize("irreps", ["4x0e", "4x0e + 4x1o"])
    def test_matches_pytorch(self, irreps):
        e3nn_irreps = Irreps(irreps)

        # Build PyTorch
        torch.manual_seed(42)
        torch_ln = TorchLayerNorm(e3nn_irreps)

        # Build JAX
        jax_ln = JaxLayerNorm(irreps)

        # Transfer weights
        jax_ln.weight.value = jnp.array(torch_ln.weight.detach().numpy())
        jax_ln.bias.value = jnp.array(torch_ln.bias.detach().numpy())

        # Test input
        np.random.seed(123)
        x_np = np.random.randn(2, 4, 4, 4, e3nn_irreps.dim).astype(np.float32)

        torch_out = torch_ln(torch.tensor(x_np)).detach().numpy()
        jax_out = np.array(jax_ln(jnp.array(x_np)))

        np.testing.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-5)


class TestEquivariantPool3d:
    """Test EquivariantPool3d output shapes."""

    def test_scalar_pool_shape(self):
        pool = JaxPool3d(2.0, (1.0, 1.0, 1.0), 'maxpool3d', "4x0e")
        x = jnp.ones((1, 4, 8, 8, 8))
        y = pool(x)
        assert y.shape == (1, 4, 4, 4, 4)

    def test_mixed_pool_shape(self):
        pool = JaxPool3d(2.0, (1.0, 1.0, 1.0), 'maxpool3d', "2x0e + 2x1o")
        x = jnp.ones((1, 8, 8, 8, 8))
        y = pool(x)
        assert y.shape == (1, 8, 4, 4, 4)

    def test_avg_pool_shape(self):
        pool = JaxPool3d(2.0, (1.0, 1.0, 1.0), 'average', "4x0e")
        x = jnp.ones((1, 4, 8, 8, 8))
        y = pool(x)
        assert y.shape == (1, 4, 4, 4, 4)


class TestSphericalHarmonicsNumPy:
    """Test pure NumPy SH matches cuequivariance for l=0,1,2."""

    def _cuex_sh(self, ls, vecs_np):
        """Reference SH via cuequivariance_jax."""
        import cuequivariance as cue
        import cuequivariance_jax as cuex
        with cue.assume(cue.mul_ir):
            vec_rep = cuex.RepArray(
                cue.Irreps("O3", "1x1o"), jnp.array(vecs_np)
            )
            sh_rep = cuex.spherical_harmonics(ls, vec_rep, normalize=True)
        return np.array(sh_rep.array)

    @pytest.mark.parametrize("ls", [
        [0],
        [1],
        [2],
        [0, 1],
        [0, 1, 2],
        [1, 2],
    ])
    def test_matches_cuequivariance_axis_vectors(self, ls):
        """Axis-aligned unit vectors."""
        vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float32)

        ref = self._cuex_sh(ls, vecs)
        got = _spherical_harmonics_numpy(ls, vecs)

        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("ls", [
        [0],
        [1],
        [2],
        [0, 1, 2],
    ])
    def test_matches_cuequivariance_random(self, ls):
        """Random vectors (various magnitudes)."""
        rng = np.random.RandomState(42)
        vecs = rng.randn(50, 3).astype(np.float32)

        ref = self._cuex_sh(ls, vecs)
        got = _spherical_harmonics_numpy(ls, vecs)

        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_matches_cuequivariance_zero_vector(self):
        """Origin vector returns 1 for l=0, 0 for l>=1."""
        vecs = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        ref = self._cuex_sh([0, 1, 2], vecs)
        got = _spherical_harmonics_numpy([0, 1, 2], vecs)

        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_matches_cuequivariance_non_unit(self):
        """Non-unit vectors (normalize=True normalizes internally)."""
        vecs = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.5, 0.5, 0.5],
            [10.0, -5.0, 3.0],
        ], dtype=np.float32)

        ref = self._cuex_sh([0, 1, 2], vecs)
        got = _spherical_harmonics_numpy([0, 1, 2], vecs)

        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


class TestLatticBuffersNumPy:
    """Test _compute_lattice_buffers_numpy matches the cuex-based path."""

    def test_matches_cuex_construction(self):
        """NumPy lattice buffers match a VoxelConvolution built via cuex."""
        ls = [0, 1, 2]
        diameter = 5.0
        steps = (1.0, 1.0, 1.0)
        num_radial_basis = 4
        cutoff = True

        np_lat, np_emb, np_sh, np_pad = _compute_lattice_buffers_numpy(
            diameter, steps, ls, num_radial_basis, cutoff,
        )

        # Build via cuex for reference
        conv = JaxVoxelConvolution(
            "4x0e", "4x0e", "1x0e + 1x1o + 1x2e",
            diameter, num_radial_basis, steps, cutoff,
            rngs=nnx.Rngs(0),
        )
        # The conv already used NumPy path, so force cuex path for comparison
        conv._build_lattice_buffers_cuex(steps, ls)
        ref_lat = np.array(conv.lattice[...])
        ref_emb = np.array(conv.emb[...])
        ref_sh = np.array(conv.sh[...])

        np.testing.assert_allclose(np_lat, ref_lat, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np_emb, ref_emb, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np_sh, ref_sh, rtol=1e-5, atol=1e-6)
        assert np_pad == tuple(s // 2 for s in ref_lat.shape[:3])

    def test_matches_cuex_different_spacing(self):
        """NumPy path matches cuex at non-unit spacing."""
        ls = [0, 1, 2]
        diameter = 10.0
        steps = (1.5, 2.0, 1.0)
        num_radial_basis = 5
        cutoff = True

        np_lat, np_emb, np_sh, np_pad = _compute_lattice_buffers_numpy(
            diameter, steps, ls, num_radial_basis, cutoff,
        )

        conv = JaxVoxelConvolution(
            "4x0e", "4x0e", "1x0e + 1x1o + 1x2e",
            diameter, num_radial_basis, steps, cutoff,
            rngs=nnx.Rngs(0),
        )
        ref_lat, ref_emb, ref_sh, ref_pad = conv._build_lattice_buffers_cuex(steps, ls)

        np.testing.assert_allclose(np_lat, np.array(ref_lat), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np_emb, np.array(ref_emb), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np_sh, np.array(ref_sh), rtol=1e-5, atol=1e-6)


class TestUpdateSpacingCache:
    """Test that the within-call cache deduplicates computation."""

    def test_cache_dedup(self):
        """Two convolutions with same (diameter, steps) share cached buffers."""
        steps = (1.0, 1.0, 1.0)
        conv1 = JaxVoxelConvolution(
            "4x0e", "4x0e", "1x0e + 1x1o + 1x2e",
            diameter=5.0, num_radial_basis=4, steps=steps, rngs=nnx.Rngs(0),
        )
        conv2 = JaxVoxelConvolution(
            "4x0e + 4x1o", "4x0e + 4x1o", "1x0e + 1x1o + 1x2e",
            diameter=5.0, num_radial_basis=4, steps=steps, rngs=nnx.Rngs(1),
        )

        new_steps = (2.0, 2.0, 2.0)
        cache = {}
        conv1.update_spacing(new_steps, _cache=cache)
        conv2.update_spacing(new_steps, _cache=cache)

        # Cache should have exactly 1 entry (same diameter + steps)
        assert len(cache) == 1

        # Both convolutions should have identical lattice buffers
        np.testing.assert_array_equal(
            np.array(conv1.lattice[...]), np.array(conv2.lattice[...])
        )
        np.testing.assert_array_equal(
            np.array(conv1.emb[...]), np.array(conv2.emb[...])
        )
        np.testing.assert_array_equal(
            np.array(conv1.sh[...]), np.array(conv2.sh[...])
        )
