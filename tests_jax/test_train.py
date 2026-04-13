"""Training infrastructure tests."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from irrepunet.models_jax.unet import E3nnUNet
from irrepunet.models_jax.train import (
    dice_loss, cross_entropy_loss, dice_ce_loss,
    create_train_state, train_step, eval_step,
    create_jitted_train_step,
    create_jitted_train_step_dynamic,
)


class TestLossFunctions:
    """Test loss function implementations."""

    def test_dice_loss_perfect(self):
        """Dice loss is 0 for perfect prediction."""
        logits = jnp.array([[[[100.0, -100.0], [-100.0, 100.0]]]])  # (1, 2, 1, 2)
        # Reshape for 3D: (1, 2, 1, 1, 2)
        logits = logits.reshape(1, 2, 1, 1, 2)
        labels = jnp.array([[[[0, 1]]]])  # (1, 1, 1, 2)
        labels = labels.reshape(1, 1, 1, 2)

        loss = dice_loss(logits, labels, n_classes=2)
        assert float(loss) < 0.01

    def test_dice_loss_range(self):
        """Dice loss is in [0, 1]."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4, 4, 4))
        labels = jax.random.randint(jax.random.PRNGKey(1), (2, 4, 4, 4), 0, 3)
        loss = dice_loss(logits, labels, n_classes=3)
        assert 0 <= float(loss) <= 1

    def test_ce_loss_positive(self):
        """CE loss is positive."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4, 4, 4))
        labels = jax.random.randint(jax.random.PRNGKey(1), (2, 4, 4, 4), 0, 3)
        loss = cross_entropy_loss(logits, labels)
        assert float(loss) > 0

    def test_dice_ce_combined(self):
        """Combined loss is sum of components."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4, 4, 4))
        labels = jax.random.randint(jax.random.PRNGKey(1), (2, 4, 4, 4), 0, 3)

        combined = dice_ce_loss(logits, labels, n_classes=3, dice_weight=1.0, ce_weight=1.0)
        d = dice_loss(logits, labels, n_classes=3)
        ce = cross_entropy_loss(logits, labels)

        np.testing.assert_allclose(float(combined), float(d + ce), rtol=1e-5)


class TestTraining:
    """Test training loop."""

    def test_loss_decreases(self):
        """Loss decreases over training steps."""
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )
        optimizer = create_train_state(model, learning_rate=1e-3)

        batch = {
            'image': jax.random.normal(jax.random.PRNGKey(0), (1, 1, 16, 16, 16)),
            'label': jnp.zeros((1, 16, 16, 16), dtype=jnp.int32),
        }

        losses = []
        for _ in range(5):
            loss = train_step(model, optimizer, batch, n_classes=2)
            losses.append(float(loss))

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_overfit_single_batch(self):
        """Model can overfit a single batch (loss gets very low)."""
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=1, lmax=1,
            irrep_ratios=(4, 2),
            rngs=nnx.Rngs(42),
        )
        optimizer = create_train_state(model, learning_rate=1e-2)

        # Simple batch: all zeros label (background class)
        batch = {
            'image': jax.random.normal(jax.random.PRNGKey(0), (1, 1, 8, 8, 8)),
            'label': jnp.zeros((1, 8, 8, 8), dtype=jnp.int32),
        }

        for _ in range(30):
            loss = train_step(model, optimizer, batch, n_classes=2)

        assert float(loss) < 1.0, f"Could not overfit: final loss = {float(loss):.4f}"

    def test_jit_train_step(self):
        """JIT-compiled training step runs and produces finite loss."""
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )
        optimizer = create_train_state(model, learning_rate=1e-3)

        batch = {
            'image': jax.random.normal(jax.random.PRNGKey(0), (1, 1, 16, 16, 16)),
            'label': jnp.zeros((1, 16, 16, 16), dtype=jnp.int32),
        }

        jit_step = create_jitted_train_step(model, optimizer, n_classes=2, donate=False)

        # First call triggers JIT compilation
        loss1 = jit_step(batch)
        assert jnp.isfinite(loss1), f"First JIT step produced non-finite loss: {loss1}"

        # Second call uses cached compilation
        loss2 = jit_step(batch)
        assert jnp.isfinite(loss2), f"Second JIT step produced non-finite loss: {loss2}"

        # Loss should decrease (optimizer is updating)
        assert float(loss2) < float(loss1), (
            f"Loss did not decrease over JIT steps: {float(loss1):.4f} -> {float(loss2):.4f}"
        )

    def test_jit_train_step_deep(self):
        """JIT-compiled training step on a deeper UNet (4 downsamples)."""
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=4, lmax=2,
            rngs=nnx.Rngs(42),
        )
        optimizer = create_train_state(model, learning_rate=1e-3)

        # Spatial dims must be divisible by 2^4 = 16
        batch = {
            'image': jax.random.normal(jax.random.PRNGKey(0), (1, 1, 16, 16, 16)),
            'label': jnp.zeros((1, 16, 16, 16), dtype=jnp.int32),
        }

        jit_step = create_jitted_train_step(model, optimizer, n_classes=2, donate=False)

        loss1 = jit_step(batch)
        assert jnp.isfinite(loss1), f"First JIT step produced non-finite loss: {loss1}"

        loss2 = jit_step(batch)
        assert jnp.isfinite(loss2), f"Second JIT step produced non-finite loss: {loss2}"

        assert float(loss2) < float(loss1), (
            f"Loss did not decrease over JIT steps: {float(loss1):.4f} -> {float(loss2):.4f}"
        )

    def test_jit_train_step_very_deep(self):
        """JIT-compiled training step on a very deep UNet (6 downsamples)."""
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=6, lmax=2,
            rngs=nnx.Rngs(42),
        )
        optimizer = create_train_state(model, learning_rate=1e-3)

        # Spatial dims must be divisible by 2^6 = 64
        batch = {
            'image': jax.random.normal(jax.random.PRNGKey(0), (1, 1, 128, 128, 128)),
            'label': jnp.zeros((1, 128, 128, 128), dtype=jnp.int32),
        }

        jit_step = create_jitted_train_step(model, optimizer, n_classes=2, donate=False)

        loss1 = jit_step(batch)
        assert jnp.isfinite(loss1), f"First JIT step produced non-finite loss: {loss1}"

        loss2 = jit_step(batch)
        assert jnp.isfinite(loss2), f"Second JIT step produced non-finite loss: {loss2}"

        assert float(loss2) < float(loss1), (
            f"Loss did not decrease over JIT steps: {float(loss1):.4f} -> {float(loss2):.4f}"
        )

    def test_jit_train_step_respace(self):
        """JIT training with dynamic spacing change between steps."""
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=6, lmax=2,
            spacing=(1.0, 1.0, 1.0),
            rngs=nnx.Rngs(42),
        )
        optimizer = create_train_state(model, learning_rate=1e-3)
        jit_step = create_jitted_train_step_dynamic(model, optimizer, n_classes=2, donate=False)

        # Step 1: isotropic spacing (1,1,1), 128^3
        batch1 = {
            'image': jax.random.normal(jax.random.PRNGKey(0), (1, 1, 128, 128, 128)),
            'label': jnp.zeros((1, 128, 128, 128), dtype=jnp.int32),
        }
        model.update_spacing((1.0, 1.0, 1.0))
        loss1 = jit_step(batch1)
        assert jnp.isfinite(loss1), f"Spacing (1,1,1) step produced non-finite loss: {loss1}"

        # Step 2: anisotropic spacing (0.5,0.5,6), batch=2, patch 256x256x24
        # 256 and 24 are both divisible by 2^6=64? 256 yes, 24 no — padding handles it
        batch2 = {
            'image': jax.random.normal(jax.random.PRNGKey(1), (2, 1, 256, 256, 24)),
            'label': jnp.zeros((2, 256, 256, 24), dtype=jnp.int32),
        }
        model.update_spacing((0.5, 0.5, 6.0))
        loss2 = jit_step(batch2)
        assert jnp.isfinite(loss2), f"Spacing (0.5,0.5,6) step produced non-finite loss: {loss2}"

        # Step 3: back to (1,1,1) — should reuse the cached JIT trace
        model.update_spacing((1.0, 1.0, 1.0))
        loss3 = jit_step(batch1)
        assert jnp.isfinite(loss3), f"Return to (1,1,1) step produced non-finite loss: {loss3}"

    def test_eval_step(self):
        """Eval step runs without error."""
        model = E3nnUNet(
            n_classes=2, n_base_filters=2, n_downsample=2, lmax=2,
            rngs=nnx.Rngs(42),
        )

        batch = {
            'image': jax.random.normal(jax.random.PRNGKey(0), (1, 1, 16, 16, 16)),
            'label': jnp.zeros((1, 16, 16, 16), dtype=jnp.int32),
        }

        loss = eval_step(model, batch, n_classes=2)
        assert jnp.isfinite(loss)
