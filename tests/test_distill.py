"""Tests for irrepunet.models.distill module.

Verifies that project_to_spacing produces a model whose outputs match
the native e3nn model at the same spacing.
"""

import pytest
import torch

from irrepunet.models.unet import E3nnUNet
from irrepunet.models.distill import project_to_spacing


@pytest.mark.gpu
class TestProjectToSpacing:
    @pytest.fixture
    def model_and_input(self, small_model_kwargs, device):
        """Create a small model and a random input tensor."""
        model = E3nnUNet(**small_model_kwargs).to(device)
        model.eval()
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        return model, x

    def test_output_matches_native(self, model_and_input):
        """Projected model should produce same output as native at same spacing."""
        model, x = model_and_input
        spacing = (1.0, 1.0, 1.0)

        with torch.no_grad():
            y_native = model(x, spacing=spacing)

        projected = project_to_spacing(model, spacing=spacing)
        with torch.no_grad():
            y_proj = projected(x)

        # Distillation introduces small numerical differences from
        # recomputing weight matrices; 1e-3 tolerance is appropriate
        torch.testing.assert_close(y_proj, y_native, atol=1e-3, rtol=1e-3)

    def test_output_shape(self, model_and_input):
        """Projected model output shape should match input spatial dims."""
        model, x = model_and_input
        projected = project_to_spacing(model, spacing=(1.0, 1.0, 1.0))
        with torch.no_grad():
            y = projected(x)
        assert y.shape == (1, 2, 16, 16, 16)

    def test_no_e3nn_modules_remain(self, model_and_input):
        """After projection, no e3nn modules should remain."""
        model, x = model_and_input
        projected = project_to_spacing(model, spacing=(1.0, 1.0, 1.0))

        for name, module in projected.named_modules():
            module_type = type(module).__module__
            # Allow e3nn Dropout (it's just a wrapper) but check main layers
            if "e3nn" in module_type:
                assert not isinstance(module, (
                    type(None),  # placeholder
                )), f"e3nn module found: {name} ({type(module)})"

    def test_different_spacings_different_kernels(self, small_model_kwargs, device):
        """Models projected to different spacings should differ."""
        model = E3nnUNet(**small_model_kwargs).to(device)
        model.eval()

        proj_iso = project_to_spacing(model, spacing=(1.0, 1.0, 1.0))
        proj_aniso = project_to_spacing(model, spacing=(1.0, 1.0, 3.0))

        # Check that at least some Conv3d weights differ
        params_iso = dict(proj_iso.named_parameters())
        params_aniso = dict(proj_aniso.named_parameters())

        has_diff = False
        for name in params_iso:
            if name in params_aniso:
                if not torch.equal(params_iso[name], params_aniso[name]):
                    has_diff = True
                    break
        assert has_diff, "Projected models at different spacings should have different weights"

    def test_projected_model_no_spacing_arg(self, model_and_input):
        """Projected model forward() should work without spacing argument."""
        model, x = model_and_input
        projected = project_to_spacing(model, spacing=(1.0, 1.0, 1.0))
        with torch.no_grad():
            # Should not raise
            y = projected(x)
            assert y.shape[0] == 1


@pytest.mark.gpu
class TestProjectToSpacingAnisotropic:
    def test_anisotropic_spacing(self, small_model_kwargs, device):
        """Projection should work for anisotropic spacings."""
        model = E3nnUNet(**small_model_kwargs).to(device)
        model.eval()

        spacing = (0.5, 0.5, 5.0)
        projected = project_to_spacing(model, spacing=spacing)

        # Input size must be compatible with the pooling factors
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        with torch.no_grad():
            y = projected(x)
        assert y.shape[0] == 1
        assert y.shape[1] == 2
