"""Weight transfer from PyTorch to JAX models.

Loads PyTorch state dicts and maps parameters into JAX E3nnUNet.
"""

import jax.numpy as jnp
import numpy as np


def _torch_to_jax(tensor):
    """Convert a PyTorch tensor to a JAX array."""
    return jnp.array(tensor.detach().cpu().numpy())


def transfer_voxel_conv_weights(torch_conv, jax_conv):
    """Transfer weights from a PyTorch VoxelConvolution to JAX.

    Parameters
    ----------
    torch_conv : torch.nn.Module
        PyTorch VoxelConvolution.
    jax_conv : models_jax.layers.VoxelConvolution
        JAX convolution.
    """
    # Transfer radial weights
    jax_conv.weight[...] = _torch_to_jax(torch_conv.weight)

    # Transfer SC weights (stored in conv.sc.weight via e3nn.o3.Linear)
    if hasattr(torch_conv, 'sc') and hasattr(torch_conv.sc, 'weight'):
        jax_conv.sc_weight[...] = _torch_to_jax(torch_conv.sc.weight).reshape(-1)


def transfer_gate_weights(torch_gate, jax_gate):
    """Transfer Gate weights (Gate has no learnable params, nothing to transfer)."""
    pass


def transfer_batchnorm_weights(torch_bn, jax_bn):
    """Transfer BatchNorm weights from PyTorch e3nn to JAX.

    Parameters
    ----------
    torch_bn : e3nn.nn.BatchNorm or EquivariantLayerNorm
    jax_bn : models_jax.layers.EquivariantBatchNorm or EquivariantLayerNorm
    """
    from irrepunet.models_jax.layers import EquivariantBatchNorm, EquivariantLayerNorm, Identity

    if isinstance(jax_bn, Identity):
        return

    if isinstance(jax_bn, EquivariantLayerNorm):
        # LayerNorm: weight and bias
        if hasattr(torch_bn, 'weight') and hasattr(jax_bn, 'weight'):
            jax_bn.weight[...] = _torch_to_jax(torch_bn.weight)
        if hasattr(torch_bn, 'bias') and hasattr(jax_bn, 'bias'):
            jax_bn.bias[...] = _torch_to_jax(torch_bn.bias)
        return

    if isinstance(jax_bn, EquivariantBatchNorm):
        # e3nn BatchNorm has .weight and .bias as Parameters
        if hasattr(torch_bn, 'weight') and hasattr(jax_bn, 'weight'):
            jax_bn.weight[...] = _torch_to_jax(torch_bn.weight)
        if hasattr(torch_bn, 'bias') and hasattr(jax_bn, 'bias'):
            jax_bn.bias[...] = _torch_to_jax(torch_bn.bias)
        # Running stats
        if hasattr(torch_bn, 'running_mean') and hasattr(jax_bn, 'running_mean'):
            jax_bn.running_mean[...] = _torch_to_jax(torch_bn.running_mean)
        if hasattr(torch_bn, 'running_var') and hasattr(jax_bn, 'running_var'):
            jax_bn.running_var[...] = _torch_to_jax(torch_bn.running_var)


def transfer_conv_block_weights(torch_block, jax_block):
    """Transfer ConvolutionBlock weights.

    Parameters
    ----------
    torch_block : models.layers.ConvolutionBlock
    jax_block : models_jax.layers.ConvolutionBlock
    """
    transfer_voxel_conv_weights(torch_block.conv1, jax_block.conv1)
    transfer_batchnorm_weights(torch_block.batchnorm1, jax_block.batchnorm1)
    transfer_voxel_conv_weights(torch_block.conv2, jax_block.conv2)
    transfer_batchnorm_weights(torch_block.batchnorm2, jax_block.batchnorm2)


def transfer_encoder_weights(torch_encoder, jax_encoder):
    """Transfer Encoder weights."""
    for t_block, j_block in zip(torch_encoder.down_blocks, jax_encoder.down_blocks):
        transfer_conv_block_weights(t_block, j_block)


def transfer_decoder_weights(torch_decoder, jax_decoder):
    """Transfer Decoder weights."""
    for t_block, j_block in zip(torch_decoder.up_blocks, jax_decoder.up_blocks):
        transfer_conv_block_weights(t_block, j_block)

    # Deep supervision heads
    if torch_decoder.ds_heads is not None and jax_decoder.ds_heads is not None:
        for t_head, j_head in zip(torch_decoder.ds_heads, jax_decoder.ds_heads):
            j_head.weight[...] = _torch_to_jax(t_head.weight).reshape(-1)


def transfer_weights(torch_model, jax_model):
    """Transfer all weights from PyTorch E3nnUNet to JAX E3nnUNet.

    Parameters
    ----------
    torch_model : models.unet.E3nnUNet
        Trained PyTorch model.
    jax_model : models_jax.unet.E3nnUNet
        JAX model with matching architecture.
    """
    # Encoder
    transfer_encoder_weights(torch_model.encoder, jax_model.encoder)

    # Decoder
    transfer_decoder_weights(torch_model.decoder, jax_model.decoder)

    # Output linear
    jax_model.out_weight[...] = _torch_to_jax(torch_model.out.weight).reshape(-1)

    # Bias
    jax_model.bias[...] = _torch_to_jax(torch_model.bias)


def load_pytorch_checkpoint(path, jax_model):
    """Load a PyTorch checkpoint and transfer weights to JAX model.

    Parameters
    ----------
    path : str
        Path to PyTorch checkpoint (.pt or .pth file).
    jax_model : models_jax.unet.E3nnUNet
        JAX model to receive weights.
    """
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Build a temporary PyTorch model to load the state dict into,
    # then transfer weights
    from irrepunet.models.unet import E3nnUNet as TorchE3nnUNet

    # Reconstruct PyTorch model with matching config
    torch_model = TorchE3nnUNet(
        n_classes=jax_model.n_classes,
        in_channels=jax_model.in_channels,
        diameter=jax_model.diameter,
        num_radial_basis=jax_model.num_radial_basis,
        spacing=jax_model.spacing,
        normalization=jax_model.normalization,
        n_base_filters=jax_model.n_base_filters,
        n_downsample=jax_model.n_downsample,
        equivariance=jax_model.equivariance,
        lmax=jax_model.lmax,
        pool_mode=jax_model.pool_mode,
        scale=jax_model.scale,
        dropout_prob=jax_model.dropout_prob,
        scalar_upsampling=jax_model.scalar_upsampling,
        cutoff=jax_model.cutoff,
        deep_supervision=jax_model.deep_supervision,
        max_features=jax_model.max_features,
        irrep_ratios=jax_model.irrep_ratios,
        fill_to_max=jax_model.fill_to_max,
    )
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    transfer_weights(torch_model, jax_model)
