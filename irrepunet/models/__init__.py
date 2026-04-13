"""Model architectures for e3nnUNet."""

# distill.py only depends on PyTorch (not e3nn), safe to import first
from .distill import project_to_spacing, architecture_spacing_range, update_projected_weights, optimal_scales_for_spacing, export_hierarchical_bundle, compute_architecture_key, compute_kernel_sizes, optimize_pooling_scales, optimize_bottleneck_kernels, _assemble_state_dict

# e3nn-dependent imports (optional in JAX-only environments)
try:
    from .unet import E3nnUNet
except ImportError:
    pass

# Buffer names that depend on voxel spacing and are recomputed by update_spacing()
SPACING_DEPENDENT_BUFFERS = {'lattice', 'emb', 'sh', 'proj'}


def spacing_independent_state_dict(model):
    """Return model state_dict with spacing-dependent buffers removed."""
    return {k: v for k, v in model.state_dict().items()
            if k.rsplit('.', 1)[-1] not in SPACING_DEPENDENT_BUFFERS}


def load_spacing_independent_state_dict(model, state_dict):
    """Load a state_dict, skipping spacing-dependent buffers with shape mismatches.

    Missing or shape-mismatched spacing buffers are left at their current
    values (set by the model constructor or a subsequent update_spacing() call).
    """
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape != v.shape:
            skipped.append(k)
        else:
            filtered[k] = v
    if skipped:
        print(f"  Skipped {len(skipped)} shape-mismatched keys (spacing-dependent buffers)")
    model.load_state_dict(filtered, strict=False)


def get_model_config(model):
    """Module-level wrapper: return model constructor args as a dict."""
    return model.get_model_config()


__all__ = [
    "E3nnUNet",
    "get_model_config",
    "project_to_spacing",
    "architecture_spacing_range",
    "update_projected_weights",
    "optimal_scales_for_spacing",
    "SPACING_DEPENDENT_BUFFERS",
    "spacing_independent_state_dict",
    "load_spacing_independent_state_dict",
    "export_hierarchical_bundle",
    "compute_architecture_key",
    "compute_kernel_sizes",
    "optimize_pooling_scales",
    "optimize_bottleneck_kernels",
]
