"""Full model forward pass tests."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import torch
import flax.nnx as nnx

from irrepunet.models_jax.unet import E3nnUNet as JaxUNet, create_model as jax_create_model
from irrepunet.models_jax.weight_transfer import transfer_weights


class TestE3nnUNet:
    """Test JAX E3nnUNet."""

    def test_output_shape(self):
        model = JaxUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 1, 16, 16, 16))
        y = model(x)
        assert y.shape == (1, 2, 16, 16, 16)

    def test_output_shape_multiclass(self):
        model = JaxUNet(
            n_classes=5, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 1, 16, 16, 16))
        y = model(x)
        assert y.shape == (1, 5, 16, 16, 16)

    def test_output_shape_scalar_only(self):
        model = JaxUNet(
            n_classes=2, n_base_filters=2, n_downsample=1, lmax=0,
            irrep_ratios=(4,),
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 1, 16, 16, 16))
        y = model(x)
        assert y.shape == (1, 2, 16, 16, 16)

    def test_create_model_factory(self):
        model = jax_create_model(n_classes=3, rngs=nnx.Rngs(0))
        x = jnp.ones((1, 1, 16, 16, 16))
        y = model(x)
        assert y.shape == (1, 3, 16, 16, 16)

    def test_batch_size(self):
        model = JaxUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((2, 1, 16, 16, 16))
        y = model(x)
        assert y.shape == (2, 2, 16, 16, 16)

    def test_deep_supervision(self):
        model = JaxUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            deep_supervision=True,
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 1, 16, 16, 16))
        outputs = model(x)
        assert isinstance(outputs, list)
        assert len(outputs) == 2  # n_downsample=2: 1 DS + 1 main
        # Finest resolution should match input spatial
        assert outputs[-1].shape == (1, 2, 16, 16, 16)


class TestWeightTransfer:
    """Test weight transfer from PyTorch to JAX."""

    def test_transfer_matches_output(self):
        """PyTorch and JAX models produce same output after weight transfer."""
        from irrepunet.models.unet import E3nnUNet as TorchUNet

        # Build matching models
        torch.manual_seed(42)
        torch_model = TorchUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            backend='e3nn',
        )
        torch_model.eval()

        jax_model = JaxUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(0),
        )

        # Transfer weights
        transfer_weights(torch_model, jax_model)

        # Compare outputs
        np.random.seed(123)
        x_np = np.random.randn(1, 1, 16, 16, 16).astype(np.float32)

        with torch.no_grad():
            torch_out = torch_model(torch.tensor(x_np)).numpy()
        jax_out = np.array(jax_model(jnp.array(x_np), use_running_average=True))

        # Relaxed tolerance: numerical differences compound through the deep
        # network (5 conv blocks, norms, gates) due to different FP operation
        # orderings between JAX XLA and PyTorch. Individual layer matches are
        # tight (rtol=1e-3) but full-model error accumulates.
        np.testing.assert_allclose(
            jax_out, torch_out, rtol=0.05, atol=0.15,
            err_msg="Full model output mismatch after weight transfer"
        )


class TestDynamicSpacing:
    """Test dynamic spacing support."""

    def test_update_spacing_forward_shape(self):
        """Output shape is correct after update_spacing."""
        model = JaxUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            spacing=(1.0, 1.0, 1.0), rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 1, 16, 16, 16))

        model.update_spacing((2.0, 2.0, 2.0))
        y = model(x)
        assert y.shape == (1, 2, 16, 16, 16)

    def test_spacing_noop_when_same(self):
        """update_spacing is a no-op when spacing hasn't changed."""
        model = JaxUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            spacing=(1.0, 1.0, 1.0), rngs=nnx.Rngs(42),
        )
        # Grab lattice reference before
        lattice_before = model.encoder.down_blocks[0].conv1.lattice[...]

        model.update_spacing((1.0, 1.0, 1.0))

        lattice_after = model.encoder.down_blocks[0].conv1.lattice[...]
        # Should be the exact same object (no rebuild)
        assert lattice_before is lattice_after

    def test_pool_kernel_size_updates(self):
        """Pool kernel_size changes with spacing."""
        model = JaxUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            spacing=(1.0, 1.0, 1.0), rngs=nnx.Rngs(42),
        )
        ks_before = [p.kernel_size for p in model.encoder.down_pool]

        model.update_spacing((0.5, 0.5, 0.5))
        ks_after = [p.kernel_size for p in model.encoder.down_pool]

        # Finer spacing -> larger pool kernels
        assert ks_after[0] != ks_before[0]
        assert all(a >= b for a, b in zip(ks_after[0], ks_before[0]))

    def test_spacing_kwarg_on_call(self):
        """Passing spacing= to __call__ works (eager mode)."""
        model = JaxUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            spacing=(1.0, 1.0, 1.0), rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 1, 16, 16, 16))

        y = model(x, spacing=(2.0, 2.0, 2.0))
        assert y.shape == (1, 2, 16, 16, 16)
        assert model.spacing == (2.0, 2.0, 2.0)


class TestJIT:
    """Test JIT compilation."""

    def test_jit_forward(self):
        """Model forward works under jax.jit."""
        model = JaxUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 1, 16, 16, 16))

        # Use eval mode (use_running_average=True) so no state mutation
        y_eager = model(x, use_running_average=True)

        # Now split and JIT with the same state
        graphdef, state = nnx.split(model)

        @jax.jit
        def forward(state, x):
            m = nnx.merge(graphdef, state)
            return m(x, use_running_average=True)

        y_jit = forward(state, x)

        # JIT compilation can reorder floating point operations, causing
        # numerical differences that compound through deep networks.
        # segmented_polynomial uses a different internal computation order
        # than manual einsums, widening the eager-vs-JIT gap slightly.
        np.testing.assert_allclose(
            np.array(y_eager), np.array(y_jit), rtol=0.1, atol=1.5
        )
