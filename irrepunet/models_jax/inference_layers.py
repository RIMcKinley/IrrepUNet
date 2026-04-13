"""Distilled JAX layers for fast inference.

Precomputes fused convolution kernels (TP + self-connection) so that
inference requires no cuequivariance dependency and no tensor product
evaluation -- just conv3d -> norm -> activation.

Usage:
    from irrepunet.models_jax import E3nnUNet, distill, distill_at_spacing

    model = E3nnUNet(n_classes=3, rngs=nnx.Rngs(42))
    # ... training ...

    model = distill(model)           # or
    model = distill_at_spacing(model, spacing=(1.0, 1.0, 1.0))
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from irrepunet.models_jax.layers import (
    Buffer,
    _build_sc_weight_matrix_fast,
)


class DistilledVoxelConvolution(nnx.Module):
    """Voxel convolution with precomputed fused kernel.

    Stores the combined TP + self-connection kernel. Forward pass is a
    single conv_general_dilated call -- no TP evaluation, no CG constants,
    no spherical harmonics, no radial basis.
    """

    def __init__(self, kernel, padding):
        self.kernel = Buffer(jnp.array(kernel))  # (O, I, X, Y, Z)
        self.padding = padding                    # tuple of 3 ints

    @classmethod
    def from_trained(cls, conv):
        """Precompute fused kernel from a trained VoxelConvolution.

        Parameters
        ----------
        conv : VoxelConvolution
            Trained convolution layer.
        """
        # Build SC weight matrix
        if conv._has_sc_scatter:
            sc_w = _build_sc_weight_matrix_fast(
                conv.sc_weight[...],
                conv._sc_src[...], conv._sc_dst[...], conv._sc_alpha[...],
                conv.out_dim, conv.in_dim,
            )
        else:
            sc_w = jnp.zeros((conv.out_dim, conv.in_dim), dtype=conv.weight[...].dtype)

        # Sphere-norm: match TP kernel normalization for SC weight
        if conv.sphere_norm and conv._n_sphere_voxels_val:
            sc_w = sc_w / conv._n_sphere_voxels_val

        is_1x1x1 = all(s == 1 for s in conv.lattice[...].shape[:3])
        if is_1x1x1:
            # Only SC contributes (SH undefined at zero vector)
            kernel = sc_w[:, :, None, None, None]
        else:
            kernel = conv.kernel()
            cx = kernel.shape[2] // 2
            cy = kernel.shape[3] // 2
            cz = kernel.shape[4] // 2
            kernel = kernel.at[:, :, cx, cy, cz].add(sc_w)

        return cls(kernel, conv.padding)

    def __call__(self, x):
        """Forward pass: single conv3d with fused kernel.

        Parameters
        ----------
        x : jax.Array
            Shape (B, C_in, D, H, W).

        Returns
        -------
        jax.Array
            Shape (B, C_out, D, H, W).
        """
        return jax.lax.conv_general_dilated(
            x, self.kernel[...].astype(x.dtype),
            window_strides=(1, 1, 1),
            padding=[(p, p) for p in self.padding],
            dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW'),
        )


class DistilledLinearHead(nnx.Module):
    """Linear head with precomputed dense weight matrix.

    Replaces _DSHead and the UNet output projection. No SC scatter
    computation at inference.
    """

    def __init__(self, weight_matrix, bias=None):
        self.weight = Buffer(jnp.array(weight_matrix))  # (out_dim, in_dim)
        if bias is not None:
            self.bias = Buffer(jnp.array(bias))
        else:
            self.bias = None

    @classmethod
    def from_ds_head(cls, head):
        """Precompute from a trained _DSHead.

        Parameters
        ----------
        head : _DSHead
            Trained deep supervision head.
        """
        w_mat = _build_sc_weight_matrix_fast(
            head.weight[...],
            head._sc_src[...], head._sc_dst[...], head._sc_alpha[...],
            head.out_dim, head.in_dim,
        )
        return cls(w_mat)

    def __call__(self, x):
        """Forward: channels-first matmul.

        Parameters
        ----------
        x : jax.Array
            Shape (B, C_in, D, H, W).

        Returns
        -------
        jax.Array
            Shape (B, C_out, D, H, W).
        """
        x_cl = jnp.moveaxis(x, 1, -1)  # (B, D, H, W, C)
        out = x_cl @ self.weight[...].astype(x_cl.dtype).T
        if self.bias is not None:
            out = out + self.bias[...].astype(out.dtype)
        return jnp.moveaxis(out, -1, 1)


def distill(model):
    """Distill a JAX E3nnUNet for fast inference.

    Precomputes fused convolution kernels and replaces VoxelConvolution
    layers with simple conv wrappers. The resulting model:
    - Has no cuequivariance dependency at inference
    - Skips TP evaluation, SH, and radial basis on every forward call
    - Produces identical outputs to the original model

    Mutates the model in-place and returns it.

    Parameters
    ----------
    model : E3nnUNet
        Trained JAX UNet model.

    Returns
    -------
    E3nnUNet
        The same model with distilled layers.
    """
    # Replace VoxelConvolutions in encoder ConvBlocks
    for block in model.encoder.down_blocks:
        block.conv1 = DistilledVoxelConvolution.from_trained(block.conv1)
        block.conv2 = DistilledVoxelConvolution.from_trained(block.conv2)

    # Replace VoxelConvolutions in decoder ConvBlocks
    for block in model.decoder.up_blocks:
        block.conv1 = DistilledVoxelConvolution.from_trained(block.conv1)
        block.conv2 = DistilledVoxelConvolution.from_trained(block.conv2)

    # Replace deep supervision heads
    for i in range(len(model.decoder.ds_heads)):
        model.decoder.ds_heads[i] = DistilledLinearHead.from_ds_head(
            model.decoder.ds_heads[i]
        )

    # Precompute output projection weight matrix.
    # Use identity-scatter trick: store the dense matrix as out_weight and
    # set scatter indices to identity so _build_sc_weight_matrix_fast in
    # __call__ returns the precomputed matrix without recomputation.
    if model._has_out_sc:
        w_mat = _build_sc_weight_matrix_fast(
            model.out_weight[...],
            model._out_sc_src[...], model._out_sc_dst[...],
            model._out_sc_alpha[...],
            model._out_out_dim, model._out_in_dim,
        )
        n = model._out_out_dim * model._out_in_dim
        model.out_weight = Buffer(w_mat.reshape(-1))
        model._out_sc_src = Buffer(jnp.arange(n, dtype=jnp.int32))
        model._out_sc_dst = Buffer(jnp.arange(n, dtype=jnp.int32))
        model._out_sc_alpha = Buffer(jnp.ones(n, dtype=jnp.float32))

    # Convert bias from Param to Buffer (inference-only)
    model.bias = Buffer(model.bias[...])

    return model


def distill_at_spacing(model, spacing):
    """Distill a JAX E3nnUNet frozen to a specific spacing.

    Updates the model to the target spacing (rebuilds lattice, SH, radial
    basis), then precomputes all kernels. The resulting model works only
    at the specified spacing.

    Parameters
    ----------
    model : E3nnUNet
        Trained JAX UNet model (not yet distilled).
    spacing : tuple
        Target physical spacing (voxel size) per dimension.

    Returns
    -------
    E3nnUNet
        Distilled model frozen at the given spacing.
    """
    model.update_spacing(spacing)
    return distill(model)
