"""Tests for irrepunet.models.layers module.

Tests VoxelConvolution shapes, dynamic spacing rebuild, EquivariantPool3d,
and rotation equivariance.
"""

import math
import pytest
import torch
from e3nn import o3
from e3nn.o3 import Irreps

from irrepunet.models.layers import (
    VoxelConvolution,
    EquivariantPool3d,
    ConvolutionBlock,
    NormSoftClamp,
    _build_irreps,
)


# ---------------------------------------------------------------------------
# VoxelConvolution
# ---------------------------------------------------------------------------

class TestVoxelConvolution:
    @pytest.fixture
    def simple_conv(self):
        """A small VoxelConvolution for testing."""
        irreps_in = Irreps("2x0e + 1x1e")
        irreps_out = Irreps("4x0e + 2x1e")
        irreps_sh = o3.Irreps.spherical_harmonics(1, 1)
        return VoxelConvolution(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            irreps_sh=irreps_sh,
            diameter=5.0,
            num_radial_basis=3,
            steps=(1.0, 1.0, 1.0),
        )

    def test_output_shape(self, simple_conv):
        x = torch.randn(1, simple_conv.irreps_in.dim, 8, 8, 8)
        out = simple_conv(x)
        assert out.shape == (1, simple_conv.irreps_out.dim, 8, 8, 8)

    def test_output_batch(self, simple_conv):
        x = torch.randn(2, simple_conv.irreps_in.dim, 8, 8, 8)
        out = simple_conv(x)
        assert out.shape[0] == 2

    def test_lattice_shape_depends_on_spacing(self):
        """Different spacings should produce different lattice sizes."""
        irreps_in = Irreps("2x0e")
        irreps_out = Irreps("2x0e")
        irreps_sh = o3.Irreps.spherical_harmonics(1, 1)

        conv_fine = VoxelConvolution(
            irreps_in, irreps_out, irreps_sh,
            diameter=5.0, num_radial_basis=3, steps=(0.5, 0.5, 0.5),
        )
        conv_coarse = VoxelConvolution(
            irreps_in, irreps_out, irreps_sh,
            diameter=5.0, num_radial_basis=3, steps=(2.0, 2.0, 2.0),
        )
        # Fine spacing should have more lattice points
        assert conv_fine.lattice.shape[0] > conv_coarse.lattice.shape[0]

    def test_update_spacing(self, simple_conv):
        """update_spacing should change lattice without new parameters."""
        old_weight_data = simple_conv.weight.data.clone()
        old_lattice_shape = simple_conv.lattice.shape

        simple_conv.update_spacing((2.0, 2.0, 2.0))

        # Learned weights unchanged
        assert torch.equal(simple_conv.weight.data, old_weight_data)
        # Lattice shape changed (coarser = fewer points)
        assert simple_conv.lattice.shape[0] < old_lattice_shape[0]

    def test_kernel_shape(self, simple_conv):
        kernel = simple_conv.kernel()
        expected_out = simple_conv.irreps_out.dim
        expected_in = simple_conv.irreps_in.dim
        assert kernel.shape[0] == expected_out
        assert kernel.shape[1] == expected_in
        assert kernel.shape[2:] == simple_conv.lattice.shape[:3]

    def test_gradient_flows(self, simple_conv):
        x = torch.randn(1, simple_conv.irreps_in.dim, 6, 6, 6)
        out = simple_conv(x)
        out.sum().backward()
        assert simple_conv.weight.grad is not None


# ---------------------------------------------------------------------------
# EquivariantPool3d
# ---------------------------------------------------------------------------

class TestEquivariantPool3d:
    def test_kernel_size_computation(self):
        """Kernel size depends on scale and spacing."""
        pool = EquivariantPool3d(
            scale=2.0, steps=(1.0, 1.0, 1.0),
            mode="maxpool3d", irreps=Irreps("4x0e"),
        )
        assert pool.kernel_size == (2, 2, 2)

    def test_kernel_size_anisotropic(self):
        """Coarse axis should have kernel=1 when step >= scale."""
        pool = EquivariantPool3d(
            scale=2.0, steps=(1.0, 1.0, 5.0),
            mode="maxpool3d", irreps=Irreps("4x0e"),
        )
        assert pool.kernel_size == (2, 2, 1)

    def test_update_spacing(self):
        pool = EquivariantPool3d(
            scale=2.0, steps=(1.0, 1.0, 1.0),
            mode="maxpool3d", irreps=Irreps("4x0e"),
        )
        pool.update_spacing((0.5, 0.5, 0.5))
        assert pool.kernel_size == (4, 4, 4)

    def test_scalar_pool_output_shape(self):
        irreps = Irreps("4x0e")
        pool = EquivariantPool3d(
            scale=2.0, steps=(1.0, 1.0, 1.0),
            mode="maxpool3d", irreps=irreps,
        )
        x = torch.randn(1, irreps.dim, 8, 8, 8)
        out = pool(x)
        assert out.shape == (1, irreps.dim, 4, 4, 4)

    def test_mixed_irrep_pool(self):
        """Pool with both l=0 and l=1 irreps."""
        irreps = Irreps("2x0e + 1x1e")
        pool = EquivariantPool3d(
            scale=2.0, steps=(1.0, 1.0, 1.0),
            mode="maxpool3d", irreps=irreps,
        )
        x = torch.randn(1, irreps.dim, 8, 8, 8)
        out = pool(x)
        assert out.shape == (1, irreps.dim, 4, 4, 4)

    def test_average_pool(self):
        irreps = Irreps("4x0e")
        pool = EquivariantPool3d(
            scale=2.0, steps=(1.0, 1.0, 1.0),
            mode="average", irreps=irreps,
        )
        x = torch.randn(1, irreps.dim, 8, 8, 8)
        out = pool(x)
        assert out.shape == (1, irreps.dim, 4, 4, 4)


# ---------------------------------------------------------------------------
# NormSoftClamp
# ---------------------------------------------------------------------------

class TestNormSoftClamp:
    def test_scalars_pass_through(self):
        """l=0 features should be unchanged."""
        irreps = Irreps("4x0e")
        clamp = NormSoftClamp(irreps)
        x = torch.randn(2, 10, 4)
        out = clamp(x)
        torch.testing.assert_close(out, x)

    def test_l1_features_clamped(self):
        """Large l=1 features should be compressed."""
        irreps = Irreps("1x1e")
        clamp = NormSoftClamp(irreps, steepness=1.0)
        # Large input
        x = torch.randn(2, 10, 3) * 100.0
        out = clamp(x)
        # Output norms should be smaller than input norms
        in_norm = x.norm(dim=-1).mean()
        out_norm = out.norm(dim=-1).mean()
        assert out_norm < in_norm

    def test_output_shape_preserved(self):
        irreps = Irreps("2x0e + 1x1e + 1x2e")
        clamp = NormSoftClamp(irreps)
        x = torch.randn(2, 10, irreps.dim)
        out = clamp(x)
        assert out.shape == x.shape

    def test_learnable_target(self):
        irreps = Irreps("1x1e")
        clamp = NormSoftClamp(irreps)
        assert clamp.log_target is not None
        assert clamp.log_target.requires_grad


# ---------------------------------------------------------------------------
# _build_irreps helper
# ---------------------------------------------------------------------------

class TestBuildIrreps:
    def test_basic_so3(self):
        irreps = _build_irreps(ne=2, no=0, ratios=(4, 2, 1))
        # Should have 8x0e + 4x1e + 2x2e
        assert irreps.dim == 8 * 1 + 4 * 3 + 2 * 5  # 8 + 12 + 10 = 30

    def test_no_features_returns_empty(self):
        irreps = _build_irreps(ne=0, no=0, ratios=(4, 2, 1))
        assert irreps.dim == 0

    def test_fill_to(self):
        irreps = _build_irreps(ne=1, no=0, ratios=(4, 2, 1), fill_to=50)
        assert irreps.dim == 50


# ---------------------------------------------------------------------------
# Equivariance test (rotation)
# ---------------------------------------------------------------------------

class TestEquivariance:
    @pytest.mark.gpu
    def test_voxelconv_equivariance_scalar(self):
        """For scalar-only irreps, rotation of input should give
        rotation of output (both are just spatial rotations of the volume)."""
        torch.manual_seed(42)
        irreps_in = Irreps("2x0e")
        irreps_out = Irreps("2x0e")
        irreps_sh = o3.Irreps.spherical_harmonics(1, 1)

        conv = VoxelConvolution(
            irreps_in, irreps_out, irreps_sh,
            diameter=5.0, num_radial_basis=3,
            steps=(1.0, 1.0, 1.0),
        )
        conv.eval()

        # Create asymmetric input
        x = torch.randn(1, 2, 8, 8, 8)

        # Forward on original
        y = conv(x)

        # 90-degree rotation around z-axis: (x,y,z) -> (-y,x,z)
        # For the volume, this is flip dim=2 then transpose dims 2,3
        x_rot = x.flip(3).transpose(2, 3)
        y_rot_expected = y.flip(3).transpose(2, 3)

        # Rebuild conv for rotated coordinate system
        y_rot_actual = conv(x_rot)

        # They should match (up to boundary effects from kernel)
        # Compare the interior to avoid boundary artifacts
        c = 2  # crop boundary
        diff = (y_rot_actual[:, :, c:-c, c:-c, c:-c] -
                y_rot_expected[:, :, c:-c, c:-c, c:-c]).abs()
        assert diff.max().item() < 0.1, f"Max diff: {diff.max().item()}"
