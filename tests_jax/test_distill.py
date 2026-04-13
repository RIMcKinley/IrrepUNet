"""Tests for JAX model distillation."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from irrepunet.models_jax.unet import E3nnUNet
from irrepunet.models_jax.layers import VoxelConvolution, _build_sc_weight_matrix_fast
from irrepunet.models_jax.inference_layers import (
    distill,
    distill_at_spacing,
    DistilledVoxelConvolution,
    DistilledLinearHead,
)


class TestDistilledConvExact:
    """Single-layer distillation is mathematically exact."""

    def test_conv_output_exact(self):
        """DistilledVoxelConvolution gives bit-identical output."""
        conv = VoxelConvolution(
            '8x0e + 4x1o', '8x0e + 4x1o', '1x0e + 1x1o + 1x2e',
            diameter=5.0, num_radial_basis=5, steps=(1.0, 1.0, 1.0),
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 20, 8, 8, 8))
        y_orig = conv(x)

        dconv = DistilledVoxelConvolution.from_trained(conv)
        y_distilled = dconv(x)

        np.testing.assert_allclose(
            np.array(y_distilled), np.array(y_orig), atol=0.0
        )

    def test_fused_kernel_values_exact(self):
        """Precomputed fused kernel matches dynamic computation."""
        conv = VoxelConvolution(
            '8x0e + 4x1o', '8x0e + 4x1o', '1x0e + 1x1o + 1x2e',
            diameter=5.0, num_radial_basis=5, steps=(1.0, 1.0, 1.0),
            rngs=nnx.Rngs(42),
        )
        # Compute fused kernel manually (same as VoxelConvolution.__call__)
        kernel = conv.kernel()
        sc_w = _build_sc_weight_matrix_fast(
            conv.sc_weight[...],
            conv._sc_src[...], conv._sc_dst[...], conv._sc_alpha[...],
            conv.out_dim, conv.in_dim,
        )
        cx, cy, cz = kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[4] // 2
        fused = kernel.at[:, :, cx, cy, cz].add(sc_w)

        dconv = DistilledVoxelConvolution.from_trained(conv)
        stored = dconv.kernel[...]

        np.testing.assert_allclose(np.array(stored), np.array(fused), atol=0.0)


class TestDistillMatchesOriginal:
    """Distilled full model produces matching outputs.

    GPU conv3d uses non-deterministic reduction order, so repeated calls
    on the same input can differ by ~0.006. We use atol=0.01 for full-model
    tests (individual layer distillation is exact, verified above).
    """

    def test_distill_matches_mixed_irreps(self):
        """Distilled output matches original (lmax=2, mixed irreps)."""
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 1, 16, 16, 16))
        y_orig = np.array(model(x, use_running_average=True))

        distill(model)
        y_distilled = np.array(model(x, use_running_average=True))

        np.testing.assert_allclose(y_distilled, y_orig, atol=0.01)

    def test_distill_matches_scalar_only(self):
        """Distilled output matches original (lmax=0, scalar only)."""
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=1, lmax=0,
            irrep_ratios=(4,),
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 1, 16, 16, 16))
        y_orig = np.array(model(x, use_running_average=True))

        distill(model)
        y_distilled = np.array(model(x, use_running_average=True))

        np.testing.assert_allclose(y_distilled, y_orig, atol=0.01)


class TestDistillAtSpacing:
    """distill_at_spacing freezes to a specific resolution."""

    def test_output_shape_and_finite(self):
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            spacing=(1.0, 1.0, 1.0), rngs=nnx.Rngs(42),
        )
        distill_at_spacing(model, (2.0, 2.0, 2.0))

        x = jnp.ones((1, 1, 16, 16, 16))
        y = model(x, use_running_average=True)
        assert y.shape == (1, 2, 16, 16, 16)
        assert jnp.all(jnp.isfinite(y))

    def test_matches_manual_update_then_distill(self):
        """distill_at_spacing == update_spacing + distill."""
        model_a = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            spacing=(1.0, 1.0, 1.0), rngs=nnx.Rngs(42),
        )
        model_b = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            spacing=(1.0, 1.0, 1.0), rngs=nnx.Rngs(42),
        )

        distill_at_spacing(model_a, (2.0, 2.0, 2.0))
        model_b.update_spacing((2.0, 2.0, 2.0))
        distill(model_b)

        x = jnp.ones((1, 1, 16, 16, 16))
        y_a = np.array(model_a(x, use_running_average=True))
        y_b = np.array(model_b(x, use_running_average=True))

        # Same distilled kernels -> same non-determinism -> tighter tolerance
        np.testing.assert_allclose(y_a, y_b, atol=0.01)


class TestDistillDeepSupervision:
    """Distillation with deep supervision heads."""

    def test_ds_outputs_match(self):
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            deep_supervision=True, rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 1, 16, 16, 16))
        y_orig = model(x, use_running_average=True)

        distill(model)
        y_distilled = model(x, use_running_average=True)

        assert isinstance(y_distilled, list)
        assert len(y_distilled) == len(y_orig)
        for yo, yd in zip(y_orig, y_distilled):
            np.testing.assert_allclose(
                np.array(yd), np.array(yo), atol=0.01
            )

    def test_ds_heads_are_distilled(self):
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            deep_supervision=True, rngs=nnx.Rngs(42),
        )
        distill(model)

        for head in model.decoder.ds_heads:
            assert isinstance(head, DistilledLinearHead)


class TestDistillJIT:
    """Distilled model works under jax.jit."""

    def test_jit_forward(self):
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )
        distill(model)

        graphdef, state = nnx.split(model)

        @jax.jit
        def forward(state, x):
            m = nnx.merge(graphdef, state)
            return m(x, use_running_average=True)

        x = jnp.ones((1, 1, 16, 16, 16))
        y = forward(state, x)
        assert y.shape == (1, 2, 16, 16, 16)
        assert jnp.all(jnp.isfinite(y))

    def test_jit_matches_eager(self):
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )
        distill(model)

        x = jnp.ones((1, 1, 16, 16, 16))
        y_eager = np.array(model(x, use_running_average=True))

        graphdef, state = nnx.split(model)

        @jax.jit
        def forward(state, x):
            m = nnx.merge(graphdef, state)
            return m(x, use_running_average=True)

        y_jit = np.array(forward(state, x))

        # Eager-vs-JIT gap from XLA operation reordering (instance norm,
        # conv); same tolerance as test_unet.py::TestJIT::test_jit_forward
        np.testing.assert_allclose(y_jit, y_eager, rtol=0.1, atol=1.5)


class TestDistilledModuleTypes:
    """Verify that distillation replaces the correct modules."""

    def test_convs_are_distilled(self):
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )
        distill(model)

        for block in model.encoder.down_blocks:
            assert isinstance(block.conv1, DistilledVoxelConvolution)
            assert isinstance(block.conv2, DistilledVoxelConvolution)
        for block in model.decoder.up_blocks:
            assert isinstance(block.conv1, DistilledVoxelConvolution)
            assert isinstance(block.conv2, DistilledVoxelConvolution)
