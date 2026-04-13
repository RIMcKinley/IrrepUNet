"""Tests for E3nnUNet forward pass and model construction."""

import pytest
import torch

from irrepunet.models.unet import E3nnUNet


@pytest.mark.gpu
class TestE3nnUNetForward:
    def test_basic_forward(self, small_model_kwargs, device):
        """Basic forward pass should produce correct output shape."""
        model = E3nnUNet(**small_model_kwargs).to(device)
        model.eval()
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        with torch.no_grad():
            y = model(x, spacing=(1.0, 1.0, 1.0))
        assert y.shape == (1, 2, 16, 16, 16)

    def test_batch_size_2(self, small_model_kwargs, device):
        model = E3nnUNet(**small_model_kwargs).to(device)
        model.eval()
        x = torch.randn(2, 1, 16, 16, 16, device=device)
        with torch.no_grad():
            y = model(x, spacing=(1.0, 1.0, 1.0))
        assert y.shape[0] == 2

    def test_spacing_switch(self, small_model_kwargs, device):
        """Model should handle switching between spacings."""
        model = E3nnUNet(**small_model_kwargs).to(device)
        model.eval()
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        with torch.no_grad():
            y1 = model(x, spacing=(1.0, 1.0, 1.0))
            y2 = model(x, spacing=(2.0, 2.0, 2.0))
        # Both should have valid shapes
        assert y1.shape == y2.shape
        # But outputs should differ (different receptive fields)
        assert not torch.allclose(y1, y2)

    def test_deep_supervision(self, small_model_kwargs, device):
        """Deep supervision should return a list of outputs."""
        kwargs = {**small_model_kwargs, "deep_supervision": True}
        model = E3nnUNet(**kwargs).to(device)
        model.eval()
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        with torch.no_grad():
            outputs = model(x, spacing=(1.0, 1.0, 1.0))
        assert isinstance(outputs, list)
        assert len(outputs) >= 2
        # Finest output should match input spatial dims
        assert outputs[-1].shape[-3:] == (16, 16, 16)

    def test_get_model_config(self, small_model_kwargs):
        """get_model_config should return reconstructable config."""
        model = E3nnUNet(**small_model_kwargs)
        config = model.get_model_config()
        assert "model_kwargs" in config
        # Should be able to reconstruct
        model2 = E3nnUNet(**config["model_kwargs"])
        assert model2.n_classes == model.n_classes
        assert model2.n_downsample == model.n_downsample

    def test_gradient_backward(self, small_model_kwargs, device):
        """Gradients should flow through the model."""
        model = E3nnUNet(**small_model_kwargs).to(device)
        model.train()
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        y = model(x, spacing=(1.0, 1.0, 1.0))
        y.sum().backward()
        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad
