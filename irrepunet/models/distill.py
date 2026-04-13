"""Distilled equivariant layers for fast inference.

This module provides distilled versions of equivariant layers where:
1. Convolution kernels are precomputed and stored as regular Conv3d weights
2. Self-connections become regular 1x1 convolutions
3. Gate and BatchNorm are reimplemented in pure PyTorch
4. No tensor product computation happens at inference time

The distilled network uses only standard PyTorch operations (Conv3d, Linear)
and does NOT require e3nn at runtime.

Usage:
    from irrepunet.models import E3nnUNet, project_to_spacing

    model = E3nnUNet(n_classes=3)
    # ... training ...

    # Project to a specific spacing for deployment
    projected = project_to_spacing(model, spacing=(1.0, 1.0, 1.0))
    output = projected(input)  # No e3nn needed, no spacing arg needed
"""

import copy
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_sc_weight(e3nn_linear) -> torch.Tensor:
    """Extract dense weight matrix from e3nn Linear using Kronecker products.

    This is faster than passing an identity matrix through the forward pass,
    especially for large channel counts (4-9x speedup at 240-300 channels).

    Parameters
    ----------
    e3nn_linear : e3nn.o3.Linear
        The self-connection linear layer.

    Returns
    -------
    torch.Tensor
        Weight matrix of shape (out_features, in_features) such that
        output = W @ input, formatted for Conv3d 1x1 weight
        (out_features, in_features, 1, 1, 1).
    """
    d_in = e3nn_linear.irreps_in.dim
    d_out = e3nn_linear.irreps_out.dim
    mat = torch.zeros(d_out, d_in, dtype=e3nn_linear.weight.dtype,
                      device=e3nn_linear.weight.device)
    views = list(e3nn_linear.weight_views())

    for instr, w in zip(e3nn_linear.instructions, views):
        i_in = instr.i_in
        i_out = instr.i_out
        mul_in, ir_in = e3nn_linear.irreps_in[i_in]
        mul_out, ir_out = e3nn_linear.irreps_out[i_out]
        dim_ir = 2 * ir_in.l + 1

        # e3nn stores w as (mul_in, mul_out); transpose to (mul_out, mul_in)
        # for the weight matrix W where output = W @ input
        block = torch.kron(
            w.T.contiguous() * instr.path_weight,
            torch.eye(dim_ir, dtype=w.dtype, device=w.device),
        )

        in_start = sum(m * (2 * ir.l + 1) for m, ir in e3nn_linear.irreps_in[:i_in])
        out_start = sum(m * (2 * ir.l + 1) for m, ir in e3nn_linear.irreps_out[:i_out])

        mat[out_start:out_start + block.shape[0],
            in_start:in_start + block.shape[1]] = block

    return mat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def _extract_sc_weight_any(conv) -> torch.Tensor:
    """Extract SC weight matrix from a VoxelConvolution.

    Returns (out_features, in_features, 1, 1, 1) tensor.
    """
    if hasattr(conv, '_sc_layout') and conv._sc_layout is not None:
        from irrepunet.models.layers import _build_sc_weight_matrix
        w = _build_sc_weight_matrix(
            conv.sc.weight, conv._sc_layout,
            conv.irreps_out.dim, conv.irreps_in.dim,
        )
        return w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        return _extract_sc_weight(conv.sc)


def _is_voxel_conv(m):
    """Check if module is a VoxelConvolution."""
    from irrepunet.models.layers import VoxelConvolution
    return isinstance(m, VoxelConvolution)


class InferenceOnlyError(RuntimeError):
    """Raised when trying to backpropagate through inference-only layers."""
    pass


class NormalizedActivation(nn.Module):
    """Pure PyTorch normalized activation function.

    Applies: output = scale * activation(input)

    The scale is chosen so that the output has unit second moment when
    the input is standard normal. This matches e3nn's normalize2mom behavior.
    """

    def __init__(self, activation: Callable, scale: float):
        super().__init__()
        self.activation = activation
        self.register_buffer('scale', torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.activation(x)


class _DistilledActivation(nn.Module):
    """Pure PyTorch replacement for e3nn Activation."""

    def __init__(self, activations: List[Optional[nn.Module]], paths: List):
        super().__init__()
        self.activations = nn.ModuleList([a for a in activations if a is not None])
        # Store paths info: [(mul, dim, act_idx_or_None), ...]
        self.paths = []
        act_idx = 0
        for mul, (l, p), act in paths:
            if act is not None:
                self.paths.append((mul, 2 * l + 1, act_idx))
                act_idx += 1
            else:
                self.paths.append((mul, 2 * l + 1, None))

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        output = []
        index = 0

        for mul, ir_dim, has_act in self.paths:
            if has_act is not None:
                chunk = x.narrow(dim, index, mul)
                output.append(self.activations[has_act](chunk))
            else:
                output.append(x.narrow(dim, index, mul * ir_dim))
            index += mul * ir_dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(x)


class DistilledGate(nn.Module):
    """Pure PyTorch implementation of e3nn Gate.

    The gate activation splits input into:
    - scalars: passed through scalar activation (e.g., normalized relu)
    - gates: passed through gate activation (e.g., normalized sigmoid)
    - gated: multiplied element-wise by gates

    Output is concatenation of activated scalars and gated tensors.
    """

    def __init__(
        self,
        scalar_dims: int,
        gate_dims: int,
        gated_dims: int,
        gated_irreps_structure: List[Tuple[int, int]],
    ):
        super().__init__()
        self.scalar_dims = scalar_dims
        self.gate_dims = gate_dims
        self.gated_dims = gated_dims
        self.gated_irreps_structure = gated_irreps_structure

        self.act_scalars = None
        self.act_gates = None

    @classmethod
    def from_e3nn(cls, e3nn_gate: nn.Module) -> 'DistilledGate':
        """Create distilled gate from e3nn Gate."""
        scalar_dims = e3nn_gate.irreps_scalars.dim
        gate_dims = e3nn_gate.irreps_gates.dim
        gated_dims = e3nn_gate.irreps_gated.dim

        gated_irreps_structure = []
        for mul, ir in e3nn_gate.irreps_gated:
            gated_irreps_structure.append((mul, 2 * ir.l + 1))

        instance = cls(
            scalar_dims=scalar_dims,
            gate_dims=gate_dims,
            gated_dims=gated_dims,
            gated_irreps_structure=gated_irreps_structure,
        )

        instance.act_scalars = cls._extract_activations(e3nn_gate.act_scalars)
        instance.act_gates = cls._extract_activations(e3nn_gate.act_gates)

        try:
            device = next(iter(e3nn_gate.act_scalars.acts[0].parameters())).device
        except (StopIteration, IndexError, AttributeError):
            device = 'cpu'

        return instance.to(device=device)

    @staticmethod
    def _extract_activations(e3nn_activation) -> nn.Module:
        """Extract pure PyTorch activations from e3nn Activation module."""
        activations = []
        for act in e3nn_activation.acts:
            if act is not None:
                scale = float(act.cst)
                func = act.f
                activations.append(NormalizedActivation(func, scale))
            else:
                activations.append(None)

        return _DistilledActivation(activations, e3nn_activation.paths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalars = x[..., :self.scalar_dims]
        gates = x[..., self.scalar_dims:self.scalar_dims + self.gate_dims]
        gated = x[..., self.scalar_dims + self.gate_dims:]

        scalars = self.act_scalars(scalars)

        if self.gate_dims > 0:
            gates = self.act_gates(gates)

            gated_out_parts = []
            gated_idx = 0
            gate_idx = 0

            for mul, dim in self.gated_irreps_structure:
                for m in range(mul):
                    g = gates[..., gate_idx:gate_idx + 1]
                    v = gated[..., gated_idx:gated_idx + dim]
                    gated_out_parts.append(g * v)
                    gated_idx += dim
                    gate_idx += 1

            gated = torch.cat(gated_out_parts, dim=-1) if gated_out_parts else gated[..., :0]
            return torch.cat([scalars, gated], dim=-1)
        else:
            return scalars


class DistilledFusedGate(nn.Module):
    """Pure PyTorch implementation of FusedGate.

    Derives gate values from l=0 scalar features via a linear projection,
    rather than requiring extra gate channels from the convolution.
    """

    def __init__(
        self,
        n_scalars: int,
        n_gate_values: int,
        gated_irreps_structure: List[Tuple[int, int]],
    ):
        super().__init__()
        self.n_scalars = n_scalars
        self.n_gate_values = n_gate_values
        self.gated_irreps_structure = gated_irreps_structure

        self.scalar_activation = None  # Set by from_fused_gate
        if n_gate_values > 0:
            self.gate_linear = nn.Linear(n_scalars, n_gate_values, bias=False)
            self.gate_linear.weight.requires_grad = False
        else:
            self.gate_linear = None

    @classmethod
    def from_fused_gate(cls, fused_gate: nn.Module) -> 'DistilledFusedGate':
        """Create distilled FusedGate from e3nn-based FusedGate."""
        instance = cls(
            n_scalars=fused_gate.n_scalars,
            n_gate_values=fused_gate.n_gate_values,
            gated_irreps_structure=fused_gate.gated_irreps_structure,
        )

        # Store scalar activation as-is (already a pure PyTorch callable)
        instance.scalar_activation = fused_gate.scalar_activation

        # Copy gate linear weights
        if fused_gate.gate_linear is not None:
            with torch.no_grad():
                instance.gate_linear.weight.copy_(fused_gate.gate_linear.weight)

        try:
            device = fused_gate.gate_linear.weight.device if fused_gate.gate_linear is not None else 'cpu'
        except AttributeError:
            device = 'cpu'

        return instance.to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalars = x[..., :self.n_scalars]
        activated_scalars = self.scalar_activation(scalars)

        if self.gate_linear is None:
            return activated_scalars

        gated_features = x[..., self.n_scalars:]

        gate_values = torch.sigmoid(self.gate_linear(scalars))

        gated_parts = []
        gated_idx = 0
        gate_idx = 0
        for mul, dim in self.gated_irreps_structure:
            for _ in range(mul):
                g = gate_values[..., gate_idx:gate_idx + 1]
                v = gated_features[..., gated_idx:gated_idx + dim]
                gated_parts.append(g * v)
                gated_idx += dim
                gate_idx += 1

        gated = torch.cat(gated_parts, dim=-1)
        return torch.cat([activated_scalars, gated], dim=-1)


class DistilledBatchNorm(nn.Module):
    """Pure PyTorch implementation of e3nn BatchNorm.

    Normalizes each irrep type separately:
    - Scalars: batch/instance norm with mean subtraction and variance normalization
    - Higher-order irreps: normalize by component-wise variance (no mean subtraction)

    Supports both batch norm (uses running statistics) and instance norm (per-sample).
    """

    def __init__(
        self,
        irreps_structure: List[Tuple[int, int, bool]],
        eps: float = 1e-5,
        instance: bool = False,
    ):
        super().__init__()
        self.irreps_structure = irreps_structure
        self.eps = eps
        self.instance = instance

        num_scalar = sum(mul for mul, dim, is_scalar in irreps_structure if is_scalar)
        num_features = sum(mul for mul, dim, is_scalar in irreps_structure)

        if instance:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
        else:
            self.register_buffer('running_mean', torch.zeros(num_scalar))
            self.register_buffer('running_var', torch.ones(num_features))

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_scalar))

        self.weight.requires_grad = False
        self.bias.requires_grad = False

    @classmethod
    def from_e3nn(cls, e3nn_bn: nn.Module) -> 'DistilledBatchNorm':
        """Create distilled batch norm from e3nn BatchNorm."""
        irreps_structure = []
        for mul, ir in e3nn_bn.irreps:
            irreps_structure.append((mul, 2 * ir.l + 1, ir.l == 0))

        instance = cls(
            irreps_structure=irreps_structure,
            eps=e3nn_bn.eps,
            instance=e3nn_bn.instance,
        )

        device = e3nn_bn.weight.device if e3nn_bn.weight is not None else 'cpu'
        dtype = e3nn_bn.weight.dtype if e3nn_bn.weight is not None else torch.float32

        with torch.no_grad():
            if e3nn_bn.running_mean is not None:
                instance.running_mean.copy_(e3nn_bn.running_mean)
            if e3nn_bn.running_var is not None:
                instance.running_var.copy_(e3nn_bn.running_var)
            if e3nn_bn.weight is not None:
                instance.weight.copy_(e3nn_bn.weight)
            if e3nn_bn.bias is not None:
                instance.bias.copy_(e3nn_bn.bias)

        return instance.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        batch = x.shape[0]
        dim = x.shape[-1]
        x = x.reshape(batch, -1, dim)

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for mul, d, is_scalar in self.irreps_structure:
            field = x[:, :, ix:ix + mul * d]
            ix += mul * d

            field = field.reshape(batch, -1, mul, d)

            if is_scalar:
                if self.instance:
                    field_mean = field.mean(1).mean(-1)
                    field = field - field_mean.reshape(batch, 1, mul, 1)
                else:
                    field_mean = self.running_mean[irm:irm + mul]
                    field = field - field_mean.reshape(1, 1, mul, 1)
                irm += mul

            if self.instance:
                field_norm = field.pow(2).mean(3).mean(1)
            else:
                field_norm = self.running_var[irv:irv + mul]
            irv += mul

            field_norm = (field_norm + self.eps).pow(-0.5)

            weight = self.weight[iw:iw + mul]
            iw += mul

            if self.instance:
                field_norm = field_norm * weight.reshape(1, mul)
                field = field * field_norm.reshape(batch, 1, mul, 1)
            else:
                field_norm = field_norm * weight
                field = field * field_norm.reshape(1, 1, mul, 1)

            if is_scalar:
                bias = self.bias[ib:ib + mul]
                ib += mul
                field = field + bias.reshape(1, 1, mul, 1)

            fields.append(field.reshape(batch, -1, mul * d))

        output = torch.cat(fields, dim=2)
        return output.reshape(orig_shape)


class DistilledLinear(nn.Module):
    """Pure PyTorch implementation of e3nn Linear.

    e3nn Linear respects irrep structure by only connecting irreps of the
    same type. This is equivalent to a block-diagonal weight matrix.
    For inference, we extract the effective weight matrix and use nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False

    @classmethod
    def from_e3nn(cls, e3nn_linear: nn.Module) -> 'DistilledLinear':
        """Create distilled linear from e3nn Linear."""
        in_features = e3nn_linear.irreps_in.dim
        out_features = e3nn_linear.irreps_out.dim

        instance = cls(in_features, out_features)

        with torch.no_grad():
            # _extract_sc_weight returns (out, in, 1, 1, 1); squeeze for nn.Linear
            sc_weight = _extract_sc_weight(e3nn_linear)
            instance.linear.weight.copy_(sc_weight.squeeze(-1).squeeze(-1).squeeze(-1))

        device = e3nn_linear.weight.device
        dtype = e3nn_linear.weight.dtype
        return instance.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _detect_2d_kernel(kernel: torch.Tensor) -> Optional[Tuple[int, Tuple[int, int], None]]:
    """Detect if a Conv3d kernel is effectively 2D and return optimization info.

    Parameters
    ----------
    kernel : torch.Tensor
        Shape (out_channels, in_channels, k1, k2, k3)

    Returns
    -------
    Optional[Tuple[int, Tuple[int, int], None]]
        If 2D, returns (axis_to_remove, (k_a, k_b), None)
        If 3D, returns None
    """
    shape = kernel.shape
    if len(shape) != 5:
        return None

    out_c, in_c, k1, k2, k3 = shape

    for axis, size in enumerate([k1, k2, k3]):
        if size == 1:
            other_dims = [k1, k2, k3]
            other_dims.pop(axis)
            return (axis, tuple(other_dims), None)

    return None


class DistilledVoxelConvolution2D(nn.Module):
    """Optimized 2D convolution for anisotropic kernels.

    When a Conv3d kernel has a 1-sized dimension (e.g., nxnx1 for thick slices),
    this layer uses Conv2d for better performance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_2d: Tuple[int, int],
        padding_2d: Tuple[int, int],
        axis_to_remove: int,
        sc_mode: str = "parallel",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.axis_to_remove = axis_to_remove
        self.sc_mode = sc_mode

        sc_in = out_channels if sc_mode in ("conv_first", "conv_first_res") else in_channels
        self.sc = nn.Conv3d(sc_in, out_channels, kernel_size=1, bias=False)

        conv_in = out_channels if sc_mode in ("sc_first", "sc_first_res") else in_channels
        self.conv = nn.Conv2d(
            conv_in, out_channels,
            kernel_size=kernel_size_2d,
            padding=padding_2d,
            bias=False
        )

        self.conv.weight.requires_grad = False
        self.sc.weight.requires_grad = False

    @classmethod
    def from_e3nn(cls, e3nn_conv: nn.Module, axis_to_remove: int,
                  skip_kernels: bool = False,
                  precomputed_kernel: torch.Tensor = None) -> 'DistilledVoxelConvolution2D':
        """Create a 2D distilled convolution from e3nn VoxelConvolution.

        If skip_kernels=True, creates the correct architecture but leaves
        weights uninitialized (for loading cached state_dict into).
        If precomputed_kernel is provided, uses it instead of calling kernel().
        """
        sc_mode = getattr(e3nn_conv, 'sc_mode', 'parallel')
        in_channels = e3nn_conv.irreps_in.dim
        out_channels = e3nn_conv.irreps_out.dim

        kernel_size_3d = tuple(e3nn_conv.lattice.shape[:3])
        kernel_size_2d = list(kernel_size_3d)
        kernel_size_2d.pop(axis_to_remove)
        kernel_size_2d = tuple(kernel_size_2d)

        padding_2d = tuple(s // 2 for s in kernel_size_2d)

        instance = cls(in_channels, out_channels, kernel_size_2d, padding_2d, axis_to_remove, sc_mode=sc_mode)

        if not skip_kernels:
            with torch.no_grad():
                kernel_3d = precomputed_kernel if precomputed_kernel is not None else e3nn_conv.kernel()
                if axis_to_remove == 0:
                    kernel_2d = kernel_3d[:, :, 0, :, :]
                elif axis_to_remove == 1:
                    kernel_2d = kernel_3d[:, :, :, 0, :]
                else:
                    kernel_2d = kernel_3d[:, :, :, :, 0]
                instance.conv.weight.copy_(kernel_2d)

            with torch.no_grad():
                sc_w = _extract_sc_weight_any(e3nn_conv)
                if getattr(e3nn_conv, 'sphere_norm', False):
                    # Scale sc weight by voxels inside kernel radius to match
                    # the normalization applied to convolution weights in kernel().
                    n_voxels = e3nn_conv._n_kernel_voxels()
                    sc_w = sc_w / n_voxels
                instance.sc.weight.copy_(sc_w)

        return instance.to(device=e3nn_conv.weight.device, dtype=e3nn_conv.weight.dtype)

    def _conv2d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D convolution to 5D input by merging the removed axis into batch."""
        batch, channels, sx, sy, sz = x.shape
        conv_in_c = x.shape[1]

        if self.axis_to_remove == 0:
            x_2d = x.permute(0, 2, 1, 3, 4).reshape(batch * sx, conv_in_c, sy, sz)
            conv_out_2d = self.conv(x_2d)
            out_c = conv_out_2d.shape[1]
            return conv_out_2d.reshape(batch, sx, out_c, sy, sz).permute(0, 2, 1, 3, 4)
        elif self.axis_to_remove == 1:
            x_2d = x.permute(0, 3, 1, 2, 4).reshape(batch * sy, conv_in_c, sx, sz)
            conv_out_2d = self.conv(x_2d)
            out_c = conv_out_2d.shape[1]
            return conv_out_2d.reshape(batch, sy, out_c, sx, sz).permute(0, 2, 3, 1, 4)
        else:  # axis_to_remove == 2
            x_2d = x.permute(0, 4, 1, 2, 3).reshape(batch * sz, conv_in_c, sx, sy)
            conv_out_2d = self.conv(x_2d)
            out_c = conv_out_2d.shape[1]
            return conv_out_2d.reshape(batch, sz, out_c, sx, sy).permute(0, 2, 3, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad:
            raise InferenceOnlyError(
                "Backpropagation is not supported with distilled inference layers. "
                "This model is for inference only."
            )

        if self.sc_mode == "sc_first":
            return self._conv2d(self.sc(x))
        elif self.sc_mode == "sc_first_res":
            sc = self.sc(x)
            return sc + self._conv2d(sc)
        elif self.sc_mode == "conv_first":
            return self.sc(self._conv2d(x))
        elif self.sc_mode == "conv_first_res":
            conv_out = self._conv2d(x)
            return conv_out + self.sc(conv_out)
        else:
            return self.sc(x) + self._conv2d(x)


class DistilledVoxelConvolution(nn.Module):
    """Distilled voxel convolution using precomputed kernel.

    Stores the precomputed convolution kernel as a regular Conv3d weight.
    At inference time, it's just a standard convolution plus a 1x1 conv
    for the self-connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        sc_mode: str = "parallel",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sc_mode = sc_mode

        # In conv_first modes, sc operates on conv output (out_channels → out_channels)
        sc_in = out_channels if sc_mode in ("conv_first", "conv_first_res") else in_channels
        self.sc = nn.Conv3d(sc_in, out_channels, kernel_size=1, bias=False)

        # In sc_first modes, conv operates on sc output (out_channels → out_channels)
        conv_in = out_channels if sc_mode in ("sc_first", "sc_first_res") else in_channels
        self.conv = nn.Conv3d(
            conv_in, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.conv.weight.requires_grad = False
        self.sc.weight.requires_grad = False

    @classmethod
    def from_e3nn(cls, e3nn_conv: nn.Module, use_2d: bool = True,
                  skip_kernels: bool = False,
                  precomputed_kernel: torch.Tensor = None) -> nn.Module:
        """Create a distilled convolution from a trained e3nn VoxelConvolution.

        If use_2d=True and kernel is effectively 2D, returns DistilledVoxelConvolution2D.
        If skip_kernels=True, creates the correct architecture but leaves
        weights uninitialized (for loading cached state_dict into).
        If precomputed_kernel is provided, uses it instead of calling kernel().
        """
        sc_mode = getattr(e3nn_conv, 'sc_mode', 'parallel')

        # Handle 1x1x1 kernels (self-connection only, no spatial conv)
        if all(s == 1 for s in e3nn_conv.lattice.shape[:3]):
            return _DistilledSelfConnectionOnly.from_e3nn(e3nn_conv, skip_kernels=skip_kernels)

        kernel_size = tuple(e3nn_conv.lattice.shape[:3])

        # Detect 2D from lattice shape (any dim == 1)
        if use_2d:
            for axis, size in enumerate(kernel_size):
                if size == 1:
                    return DistilledVoxelConvolution2D.from_e3nn(
                        e3nn_conv, axis, skip_kernels=skip_kernels,
                        precomputed_kernel=precomputed_kernel)

        in_channels = e3nn_conv.irreps_in.dim
        out_channels = e3nn_conv.irreps_out.dim
        padding = tuple(s // 2 for s in kernel_size)

        instance = cls(in_channels, out_channels, kernel_size, padding, sc_mode=sc_mode)

        if not skip_kernels:
            with torch.no_grad():
                kernel = precomputed_kernel if precomputed_kernel is not None else e3nn_conv.kernel()
                instance.conv.weight.copy_(kernel)

            with torch.no_grad():
                sc_w = _extract_sc_weight_any(e3nn_conv)
                if getattr(e3nn_conv, 'sphere_norm', False):
                    # Scale sc weight by voxels inside kernel radius to match
                    # the normalization applied to convolution weights in kernel().
                    n_voxels = e3nn_conv._n_kernel_voxels()
                    sc_w = sc_w / n_voxels
                instance.sc.weight.copy_(sc_w)

        return instance.to(device=e3nn_conv.weight.device, dtype=e3nn_conv.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad:
            raise InferenceOnlyError(
                "Backpropagation is not supported with distilled inference layers. "
                "This model is for inference only."
            )

        if self.sc_mode == "sc_first":
            return self.conv(self.sc(x))
        elif self.sc_mode == "sc_first_res":
            sc = self.sc(x)
            return sc + self.conv(sc)
        elif self.sc_mode == "conv_first":
            return self.sc(self.conv(x))
        elif self.sc_mode == "conv_first_res":
            conv_out = self.conv(x)
            return conv_out + self.sc(conv_out)
        else:
            return self.sc(x) + self.conv(x)


class _DistilledSelfConnectionOnly(nn.Module):
    """Distilled version of a VoxelConvolution with 1x1x1 kernel.

    When the kernel is 1x1x1, the radial basis is zero at the origin,
    so the conv3d output is exactly zero. Only the self-connection contributes.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.sc = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.sc.weight.requires_grad = False

    @classmethod
    def from_e3nn(cls, e3nn_conv: nn.Module,
                  skip_kernels: bool = False) -> '_DistilledSelfConnectionOnly':
        in_channels = e3nn_conv.irreps_in.dim
        out_channels = e3nn_conv.irreps_out.dim

        instance = cls(in_channels, out_channels)

        if not skip_kernels:
            with torch.no_grad():
                instance.sc.weight.copy_(_extract_sc_weight_any(e3nn_conv))

        return instance.to(device=e3nn_conv.weight.device, dtype=e3nn_conv.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sc(x)


def _distill_gate(gate_module: nn.Module) -> nn.Module:
    """Distill a gate module (FusedGate or e3nn Gate)."""
    from irrepunet.models.layers import FusedGate

    if isinstance(gate_module, FusedGate):
        return DistilledFusedGate.from_fused_gate(gate_module)
    else:
        # Legacy e3nn Gate
        return DistilledGate.from_e3nn(gate_module)


def _distill_normalization(norm_module: nn.Module) -> nn.Module:
    """Distill a normalization module.

    Handles:
    - e3nn.nn.BatchNorm (batch or instance mode) -> DistilledBatchNorm
    - EquivariantLayerNorm -> keep as-is (already pure PyTorch)
    - Identity -> keep as-is
    """
    from irrepunet.models.layers import EquivariantLayerNorm, Identity
    from e3nn.nn import BatchNorm as E3nnBatchNorm

    if isinstance(norm_module, E3nnBatchNorm):
        return DistilledBatchNorm.from_e3nn(norm_module)
    elif isinstance(norm_module, (EquivariantLayerNorm, Identity)):
        return norm_module
    else:
        return norm_module


class DistilledConvolutionBlock(nn.Module):
    """Distilled convolution block with precomputed kernels.

    All e3nn layers are converted to pure PyTorch:
    - VoxelConvolution -> DistilledVoxelConvolution (Conv3d)
    - Gate -> DistilledGate (pure PyTorch)
    - e3nn BatchNorm -> DistilledBatchNorm (pure PyTorch)
    - EquivariantLayerNorm -> kept as-is (already pure PyTorch)
    - Identity -> kept as-is
    - e3nn Dropout -> kept as-is (already compatible)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = None
        self.conv2 = None
        self.gate1 = None
        self.gate2 = None
        self.batchnorm1 = None
        self.batchnorm2 = None
        self.dropout1 = None
        self.dropout2 = None
        self.irreps_out = None

    @classmethod
    def from_e3nn(cls, e3nn_block: nn.Module, use_2d: bool = True,
                  skip_kernels: bool = False,
                  precomputed_kernels: dict = None) -> 'DistilledConvolutionBlock':
        """Create distilled block from e3nn ConvolutionBlock.

        Parameters
        ----------
        precomputed_kernels : dict, optional
            Map from id(VoxelConvolution) -> precomputed kernel tensor.
            If provided, skips the expensive kernel() call.
        """
        instance = cls()

        pk = precomputed_kernels or {}
        instance.conv1 = DistilledVoxelConvolution.from_e3nn(
            e3nn_block.conv1, use_2d=use_2d, skip_kernels=skip_kernels,
            precomputed_kernel=pk.get(id(e3nn_block.conv1)))
        instance.conv2 = DistilledVoxelConvolution.from_e3nn(
            e3nn_block.conv2, use_2d=use_2d, skip_kernels=skip_kernels,
            precomputed_kernel=pk.get(id(e3nn_block.conv2)))

        instance.gate1 = _distill_gate(e3nn_block.gate1)
        instance.gate2 = _distill_gate(e3nn_block.gate2)

        instance.batchnorm1 = _distill_normalization(e3nn_block.batchnorm1)
        instance.batchnorm2 = _distill_normalization(e3nn_block.batchnorm2)

        instance.dropout1 = e3nn_block.dropout1
        instance.dropout2 = e3nn_block.dropout2

        instance.irreps_out = e3nn_block.irreps_out

        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.conv1(x)
        x = self.batchnorm1(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate1(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout1(x.transpose(1, 4)).transpose(1, 4)

        # Second conv block
        x = self.conv2(x)
        x = self.batchnorm2(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate2(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout2(x.transpose(1, 4)).transpose(1, 4)

        return x


def _precompute_kernels(module: nn.Module, max_workers: int = 8) -> dict:
    """Precompute all VoxelConvolution kernels in parallel using threads.

    Returns a dict mapping id(VoxelConvolution) -> kernel tensor.
    Threading works well here because e3nn tensor products release the GIL
    during the heavy torch operations.
    """
    # Collect all VoxelConvolutions (skip 1x1x1 which have no spatial kernel)
    convs = [
        m for m in module.modules()
        if _is_voxel_conv(m)
        and not all(s == 1 for s in m.lattice.shape[:3])
    ]

    if not convs:
        return {}

    def compute_kernel(conv):
        with torch.no_grad():
            return conv.kernel()

    kernels = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_kernel, c): c for c in convs}
        for future in futures:
            conv = futures[future]
            kernels[id(conv)] = future.result()

    return kernels


def _distill_module(module: nn.Module, use_2d: bool = True,
                    skip_kernels: bool = False,
                    precomputed_kernels: dict = None) -> None:
    """Recursively replace e3nn layers with distilled equivalents in-place.

    Handles:
    - ConvolutionBlock -> DistilledConvolutionBlock
    - VoxelConvolution -> DistilledVoxelConvolution (or 2D variant)
    - e3nn.o3.Linear -> DistilledLinear
    - e3nn.nn.Gate -> DistilledGate
    - e3nn.nn.BatchNorm -> DistilledBatchNorm

    Leaves alone: EquivariantLayerNorm, nn.Upsample, EquivariantPool3d, MultipolePool, Identity

    If skip_kernels=True, convolution weights are left uninitialized (for use
    as a skeleton to load cached state_dict fragments into). Non-convolution
    weights (norms, gates, linears) are still extracted normally since they're
    spacing-independent and cheap to compute.

    If precomputed_kernels is provided, uses pre-computed kernel tensors
    (keyed by id(VoxelConvolution)) instead of calling kernel() sequentially.
    """
    from irrepunet.models.layers import VoxelConvolution, ConvolutionBlock, FusedGate

    try:
        from e3nn.o3 import Linear as E3nnLinear
        from e3nn.nn import Gate as E3nnGate
        from e3nn.nn import BatchNorm as E3nnBatchNorm
        has_e3nn = True
    except ImportError:
        has_e3nn = False

    pk = precomputed_kernels or {}

    for name, child in list(module.named_children()):
        if isinstance(child, ConvolutionBlock):
            setattr(module, name, DistilledConvolutionBlock.from_e3nn(
                child, use_2d=use_2d, skip_kernels=skip_kernels,
                precomputed_kernels=pk))
        elif isinstance(child, VoxelConvolution):
            setattr(module, name, DistilledVoxelConvolution.from_e3nn(
                child, use_2d=use_2d, skip_kernels=skip_kernels,
                precomputed_kernel=pk.get(id(child))))
        elif isinstance(child, FusedGate):
            setattr(module, name, DistilledFusedGate.from_fused_gate(child))
        elif has_e3nn and isinstance(child, E3nnLinear):
            setattr(module, name, DistilledLinear.from_e3nn(child))
        elif has_e3nn and isinstance(child, E3nnGate):
            setattr(module, name, DistilledGate.from_e3nn(child))
        elif has_e3nn and isinstance(child, E3nnBatchNorm):
            setattr(module, name, DistilledBatchNorm.from_e3nn(child))
        else:
            # Recurse into children (Encoder, Decoder, etc.)
            _distill_module(child, use_2d, skip_kernels, pk)


def project_to_spacing(
    model: nn.Module,
    spacing: Tuple[float, float, float],
    use_2d: bool = True,
    scales: list = None,
    skip_kernels: bool = False,
) -> nn.Module:
    """Project an equivariant model to a specific voxel spacing.

    Creates a pure PyTorch model with precomputed Conv3d kernels frozen to
    the given spacing. The resulting model:
    - Uses only standard PyTorch operations (Conv3d, Linear, etc.)
    - Does NOT require e3nn at runtime
    - Does NOT accept a spacing argument in forward()
    - Produces identical outputs to the original model at that spacing

    Unlike the cuequivariance distill_at_spacing(), this correctly handles
    UNet models where deeper encoder levels operate at coarser effective
    spacings (due to pooling). The per-level spacing computation is delegated
    to the model's _rebuild_for_spacing() method.

    Parameters
    ----------
    model : nn.Module
        A trained equivariant model (E3nnUNet).
    spacing : tuple of float
        Target voxel spacing (mm), e.g. (1.0, 1.0, 1.0).
    use_2d : bool
        If True (default), use Conv2d for anisotropic kernels with a
        1-sized dimension for better performance.
    scales : list, optional
        Override pooling/upsampling scales. Use with
        ``optimal_scales_for_spacing()`` to avoid degenerate architecture
        boundaries and maximize TTA jitter range.
    skip_kernels : bool
        If True, create the distilled architecture with correct tensor
        shapes but skip expensive kernel computation. Use this to create
        a skeleton model for loading cached per-level state_dict fragments.

    Returns
    -------
    nn.Module
        Distilled model frozen at the specified spacing.

    Examples
    --------
    >>> from irrepunet.models import E3nnUNet, project_to_spacing
    >>> model = E3nnUNet(n_classes=2, ...)
    >>> # ... train ...
    >>> projected = project_to_spacing(model, spacing=(1.0, 1.0, 1.0))
    >>> output = projected(input_tensor)  # No spacing arg, no e3nn needed
    """
    projected = copy.deepcopy(model)
    projected.eval()

    # Apply custom scales permanently for this projection (not temporary)
    if scales is not None:
        projected.scales = list(scales)
        for i, pool in enumerate(projected.encoder.down_pool):
            pool.scale = scales[i]
        projected.decoder.scales = list(scales[::-1])

    # Set per-level spacings correctly through encoder/decoder
    spacing = tuple(float(s) for s in spacing)
    projected._rebuild_for_spacing(spacing)

    # Precompute all convolution kernels in parallel using threads
    precomputed_kernels = {} if skip_kernels else _precompute_kernels(projected)

    # Walk all modules, replacing e3nn layers with distilled equivalents
    _distill_module(projected, use_2d, skip_kernels=skip_kernels,
                    precomputed_kernels=precomputed_kernels)

    # Freeze spacing: set default_spacing so forward() never tries to rebuild.
    # After distillation, the model can only run at this spacing.
    projected.default_spacing = spacing
    projected._current_spacing = spacing

    return projected


def architecture_spacing_range(
    model: nn.Module,
    spacing: Tuple[float, float, float],
) -> Tuple[Tuple[float, float], ...]:
    """Compute per-dimension spacing range where architecture stays identical.

    Two spacings produce the same architecture (same Conv3d kernel sizes,
    pool kernel sizes, upsample factors) when all floor() operations in the
    lattice and pooling computations yield the same integer values.

    The valid range for a constraint ``floor(a / s) == k`` is
    ``s in (a / (k + 1), a / k]``.

    Parameters
    ----------
    model : nn.Module
        A trained equivariant model (E3nnUNet).
    spacing : tuple of float
        Reference voxel spacing (mm).

    Returns
    -------
    tuple of (lo, hi) pairs
        Per-dimension half-open interval ``(lo, hi]`` where any spacing
        produces the same architecture as ``spacing``.
    """
    n_downsample = model.n_downsample
    diameters = model.diameters    # [diameter * 2^i for i in range(n_downsample + 1)]
    scales = model.scales          # [scale * 2^i for i in range(n_downsample)]

    spacing = tuple(float(s) for s in spacing)

    # _compute_steps_array computes effective spacing at each level directly
    # from the input spacing (NOT cascaded):
    #   steps_array[0] = spacing
    #   steps_array[i+1][d] = floor(scales[i] / s[d]) * s[d]  if s[d] < scales[i]
    #                        = s[d]                              otherwise
    steps_array = model._compute_steps_array(spacing)

    result = []
    for d in range(3):
        s = spacing[d]
        lo = 0.0
        hi = math.inf

        for level in range(n_downsample + 1):
            eff = steps_array[level][d]
            r = diameters[level] / 2.0

            # --- Effective spacing constraint (level > 0) ---
            # _compute_steps_array computes:
            #   eff = floor(scales[level-1] / s) * s   if s < scales[level-1]
            #       = s                                  otherwise
            # For eff to stay the same integer multiple of s, we need
            # floor(scales[level-1] / s) to stay constant.
            if level > 0:
                sc_prev = scales[level - 1]
                if s < sc_prev:
                    m = math.floor(sc_prev / s)  # m >= 1
                    lo = max(lo, sc_prev / (m + 1))
                    hi = min(hi, sc_prev / m)
                else:
                    # m would be 0 (no pooling), need s >= sc_prev
                    lo = max(lo, sc_prev)

            # --- Conv lattice constraint ---
            # Lattice half-extent: floor(r / eff) must stay constant.
            # eff = m * s for level > 0 (m constrained above), s for level 0.
            m = round(eff / s) if level > 0 else 1
            k = math.floor(r / eff)
            if k > 0:
                lo = max(lo, r / ((k + 1) * m))
                hi = min(hi, r / (k * m))
            else:
                lo = max(lo, r / m)

            # --- Pool kernel constraint (levels 0..n_downsample-1) ---
            # Pool at level i uses eff_i and scales[i].
            # Kernel = floor(scales[i] / eff_i) when eff_i < scales[i].
            if level < n_downsample:
                scale_l = scales[level]
                if eff < scale_l:
                    q = math.floor(scale_l / eff)
                    lo = max(lo, scale_l / ((q + 1) * m))
                    hi = min(hi, scale_l / (q * m))
                else:
                    lo = max(lo, scale_l / m)

        result.append((lo, hi))

    return tuple(result)


def optimal_scales_for_spacing(
    model: nn.Module,
    spacing: Tuple[float, float, float],
) -> list:
    """Compute pooling scales that maximize architecture range width.

    For each pooling scale, checks whether any dimension of ``spacing``
    sits near a degenerate boundary (where the architecture range has
    zero or near-zero width). If so, shifts that scale to center the
    spacing within the resulting architecture range.

    This is useful at inference time: projecting with optimal scales
    avoids degenerate architectures and maximizes the range of TTA
    spacing jitter that stays within one architecture.

    Parameters
    ----------
    model : nn.Module
        A trained equivariant model (E3nnUNet).
    spacing : tuple of float
        The case's exact voxel spacing (mm).

    Returns
    -------
    list of float
        Adjusted pooling scales (same length as ``model.scales``).
    """
    base_scales = list(model.scales)
    adjusted = list(base_scales)
    spacing = tuple(float(s) for s in spacing)

    for i, scale in enumerate(base_scales):
        for s in spacing:
            # The effective spacing at this pool level depends on earlier
            # pooling. For level i, eff = m * s where m = floor(scales[j]/s)
            # for earlier levels j. For the pool-kernel constraint at level i,
            # the boundary is at eff == scale, i.e. s == scale / m.
            # The simplest degenerate case: s == scale (m=1, no earlier pooling).
            # More generally, s == scale / k for integer k.

            # Check: does floor(scale / s) sit right at a boundary?
            if s <= 0:
                continue
            ratio = scale / s
            k = math.floor(ratio)

            # Distance to nearest floor() boundary
            # Boundaries at ratio = k and ratio = k+1
            dist_lo = ratio - k       # distance to lower boundary
            dist_hi = (k + 1) - ratio  # distance to upper boundary
            min_dist = min(dist_lo, dist_hi)

            # If close to a boundary (within 5% of the step), adjust
            if min_dist < 0.05:
                if k == 0:
                    # s >= scale: no pooling in this dim at this level.
                    # Boundary is at s == scale. Shift scale down so s is
                    # safely in the "no pooling" region.
                    # Target: center of the no-pooling range.
                    # No-pooling range for this constraint: s >= scale.
                    # We want scale < s, with margin.
                    adjusted[i] = min(adjusted[i], s * 0.9)
                else:
                    # s < scale, kernel = k. Center s in (scale/(k+1), scale/k].
                    # Center = scale * (2k+1) / (2*k*(k+1))
                    # Solve for scale: scale = s * 2*k*(k+1) / (2k+1)
                    optimal_scale = s * 2 * k * (k + 1) / (2 * k + 1)
                    adjusted[i] = max(adjusted[i], optimal_scale)

    return adjusted


def _collect_conv_pairs(projected, original):
    """Collect matching (DistilledConv, VoxelConv) pairs for weight updates."""
    from irrepunet.models.layers import VoxelConvolution

    pairs = []
    orig_children = dict(original.named_children())

    for name_p, child_p in projected.named_children():
        if name_p not in orig_children:
            continue
        child_o = orig_children[name_p]

        if isinstance(child_p, (DistilledVoxelConvolution, DistilledVoxelConvolution2D)):
            assert _is_voxel_conv(child_o)
            pairs.append((child_p, child_o))
        elif not isinstance(child_p, _DistilledSelfConnectionOnly):
            pairs.extend(_collect_conv_pairs(child_p, child_o))

    return pairs


def _update_weights_walk(
    projected: nn.Module,
    original: nn.Module,
) -> None:
    """Update Conv3d/Conv2d weights in projected model from original.

    Precomputes all kernels in parallel using threads, then copies them
    into the projected model's convolution weights.
    """
    pairs = _collect_conv_pairs(projected, original)

    if not pairs:
        return

    # Compute all kernels in parallel
    def compute_kernel(conv_o):
        with torch.no_grad():
            return conv_o.kernel()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(compute_kernel, conv_o): (conv_p, conv_o)
                   for conv_p, conv_o in pairs}

        for future in futures:
            conv_p, conv_o = futures[future]
            kernel_3d = future.result()

            with torch.no_grad():
                if isinstance(conv_p, DistilledVoxelConvolution2D):
                    axis = conv_p.axis_to_remove
                    if axis == 0:
                        kernel_2d = kernel_3d[:, :, 0, :, :]
                    elif axis == 1:
                        kernel_2d = kernel_3d[:, :, :, 0, :]
                    else:
                        kernel_2d = kernel_3d[:, :, :, :, 0]
                    conv_p.conv.weight.copy_(kernel_2d)
                else:
                    conv_p.conv.weight.copy_(kernel_3d)

                # Update sc weight
                sc_w = _extract_sc_weight_any(conv_o)
                if getattr(conv_o, 'sphere_norm', False):
                    sc_w = sc_w / conv_o._n_kernel_voxels()
                conv_p.sc.weight.copy_(sc_w)


def update_projected_weights(
    projected: nn.Module,
    model: nn.Module,
    new_spacing: Tuple[float, float, float],
    verify: bool = True,
) -> None:
    """Recompute Conv3d/Conv2d weights for a new spacing without deep copy.

    This is much faster than calling ``project_to_spacing()`` again because
    it skips the deep copy and module replacement — only the convolution
    kernel weights are recomputed.

    The new spacing must produce the same architecture (same kernel sizes,
    pool sizes, upsample factors) as the spacing used for the original
    projection.

    Parameters
    ----------
    projected : nn.Module
        A previously projected model from ``project_to_spacing()``.
    model : nn.Module
        The original equivariant model (E3nnUNet).
    new_spacing : tuple of float
        New voxel spacing (mm) to recompute weights for.
    verify : bool
        If True (default), verify the new spacing is within the
        architecture-equivalent range of the current projected spacing.
        Raises ValueError if not.
    """
    new_spacing = tuple(float(s) for s in new_spacing)

    if verify:
        ranges = architecture_spacing_range(model, projected._current_spacing)
        for d, (lo, hi) in enumerate(ranges):
            if not (lo < new_spacing[d] <= hi):
                raise ValueError(
                    f"Spacing {new_spacing} dimension {d} ({new_spacing[d]}) "
                    f"is outside architecture-equivalent range ({lo}, {hi}] "
                    f"for current spacing {projected._current_spacing}. "
                    f"Use project_to_spacing() instead."
                )

    # Save original model's current spacing so we can restore it
    old_spacing = model._current_spacing

    # Rebuild original model for the new spacing (updates lattices/pools)
    model._rebuild_for_spacing(new_spacing)
    try:
        _update_weights_walk(projected, model)
    finally:
        # Restore original model to its previous spacing
        model._rebuild_for_spacing(old_spacing)

    projected.default_spacing = new_spacing
    projected._current_spacing = new_spacing


def compute_kernel_sizes(
    diameter: float,
    spacing: Tuple[float, float, float],
    scales: List[float],
    kernel_trim_threshold: float = 1.0,
    kernel_growth: float = 2.0,
    kernel_trim_cross_section: float = 0.0,
) -> List[Tuple[int, ...]]:
    """Compute per-level kernel sizes from raw parameters (no model needed).

    Parameters
    ----------
    diameter : float
        Base kernel diameter in physical units (mm).
    spacing : tuple of float
        Input voxel spacing (mm).
    scales : list of float
        Pooling scale at each level (length = n_downsample).
    kernel_trim_threshold : float
        Legacy trim: trim outer shell when s*step/r > threshold (1.0 = no trim).
        Only used when kernel_trim_cross_section == 0.
    kernel_growth : float
        Multiplicative factor for kernel diameter per level (default 2.0).
        Kernel radius at level k = ``diameter * kernel_growth^k / 2``.
    kernel_trim_cross_section : float
        Cross-section fraction trim threshold (0.0 = disabled).
        When > 0, trims shells where sqrt(1 - (s*step/r)^2) < threshold.
        Takes priority over kernel_trim_threshold.

    Returns
    -------
    list of (int, int, int)
        Kernel size per axis at each encoder level (len = n_downsample + 1).
    """
    from irrepunet.models.layers import _trim_half_extent

    n_downsample = len(scales)
    spacing = tuple(float(s) for s in spacing)

    # Effective spacing at each level after pooling
    from .layers import _pool_factor
    steps_array = [spacing]
    for i in range(n_downsample):
        output_steps = []
        for step in spacing:
            k = _pool_factor(scales[i], step)
            output_steps.append(k * step)
        steps_array.append(tuple(output_steps))

    # Kernel size per level
    result = []
    for lvl, sp in enumerate(steps_array):
        r = diameter * (kernel_growth ** lvl) / 2.0
        level_sizes = []
        for s in sp:
            half = math.floor(r / s)
            half = _trim_half_extent(half, s, r,
                                     kernel_trim_cross_section,
                                     kernel_trim_threshold)
            level_sizes.append(2 * half + 1)
        result.append(tuple(level_sizes))
    return result


def _compute_steps_array(spacing, scales):
    """Compute effective spacing at each encoder level (model-free).

    Replicates the logic of E3nnUNet._compute_steps_array without
    needing a model object.
    """
    steps = [tuple(float(s) for s in spacing)]
    for sc in scales:
        level = []
        for s in spacing:
            if s < sc:
                level.append(math.floor(sc / s) * s)
            else:
                level.append(s)
        steps.append(tuple(level))
    return steps


def optimize_pooling_scales(
    spacings: List[Tuple[float, float, float]],
    n_downsample: int,
    patch_size_mm: Optional[float] = None,
    diameter: Optional[float] = None,
    kernel_trim_threshold: float = 1.0,
    min_bottleneck_kernel: int = 3,
    kernel_trim_cross_section: float = 0.0,
) -> Tuple[float, float, List[float]]:
    """Compute diameter, kernel growth, and pooling scales.

    Jointly satisfies three constraints:

    1. **L0 >= 3x3x1**: The two finest axes of every case must have
       kernel size >= 3 at the input level.  Requires
       ``diameter >= 2 * max(second_finest_spacing)``.
    2. **Bottleneck >= K³** (pre-trim): At the deepest level *n*, every
       axis of every case has kernel size >= ``min_bottleneck_kernel``
       before trimming.  For K=3: ``scales[-1] = r_n`` (floor=1).
       For K=5: ``scales[-1] = r_n / 2`` (floor>=2).  Generally
       ``scales[-1] = r_n / bottleneck_half`` where
       ``bottleneck_half = K // 2``.
    3. **Bottleneck resolution <= patch/8**: At least ~8 spatial positions
       per axis at the deepest level.  Requires
       ``scales[-1] <= patch_size_mm / 8``.

    With geometric kernel growth (factor 2), constraints 1 and 2 often
    conflict.  This function resolves it by computing the optimal
    ``kernel_growth`` factor that satisfies all three.

    Parameters
    ----------
    spacings : list of (float, float, float)
        Per-case voxel spacings.
    n_downsample : int
        Number of pooling levels.
    patch_size_mm : float, optional
        Physical patch size in mm (minimum dimension if anisotropic).
        One of ``patch_size_mm`` or ``diameter`` must be given.
    diameter : float, optional
        Base kernel diameter in mm.  If not given, derived from the L0
        constraint (``2 * max(second_finest_spacing)``).
    kernel_trim_threshold : float
        Kernel trim threshold (for diagnostics).
    min_bottleneck_kernel : int
        Minimum kernel size (pre-trim) guaranteed at the deepest level
        on every axis of every case.  Must be odd.  Default 3.

    Returns
    -------
    (diameter, kernel_growth, scales) : (float, float, list of float)
        Kernel diameter, per-level growth factor, and pooling scales.
    """
    import numpy as np

    n = n_downsample
    max_sp = max(v for sp in spacings for v in sp)

    # Bottleneck half-extent: floor(r_n / eff) >= bottleneck_half
    # guarantees kernel size >= 2*bottleneck_half + 1 (pre-trim).
    bottleneck_half = max(min_bottleneck_kernel // 2, 1)

    # --- Step 1: Determine diameter from L0 constraint ---
    # 3x3x1 at L0: floor(r_0/s) >= 1 for the two finest axes
    # r_0 = diameter/2, so diameter >= 2 * max(second_finest)
    # (Trim can't reduce half=1 to 0, so no threshold adjustment needed.)
    second_finest = [sorted(sp)[1] for sp in spacings]
    d_l0 = 2.0 * max(second_finest) * (1.0 + 1e-9)

    if diameter is not None:
        if diameter < d_l0:
            print(f"  Warning: diameter {diameter:.2f}mm gives kernel < 3 on "
                  f"some in-plane axes at L0 (need >= {d_l0:.2f}mm).")
    else:
        diameter = d_l0

    # --- Step 2: Determine kernel_growth ---
    # Kernel radius at level k: r_k = diameter * growth^k / 2
    # At the deepest level n: r_n = diameter * growth^n / 2
    # scales[-1] = r_n / bottleneck_half
    #
    # From r_n >= bottleneck_half * max_spacing:
    #   growth >= (2 * bottleneck_half * max_spacing / diameter)^(1/n)
    # Epsilon ensures r_n strictly > bottleneck_half * max_sp
    min_growth = ((2.0 * bottleneck_half * max_sp / diameter) *
                  (1.0 + 1e-9)) ** (1.0 / n) if n > 0 else 1.0

    if patch_size_mm is not None:
        # From scales[-1] = r_n / bottleneck_half <= patch/8:
        #   r_n <= bottleneck_half * patch / 8
        #   growth <= (2 * bottleneck_half * patch / (8 * diameter))^(1/n)
        #           = (bottleneck_half * patch / (4 * diameter))^(1/n)
        max_growth = (bottleneck_half * patch_size_mm /
                      (4.0 * diameter)) ** (1.0 / n) if n > 0 else 1.0

        if min_growth > max_growth:
            # Constraints conflict: patch too small for this dataset's
            # max spacing.  Satisfy kernel guarantee, relax patch/8.
            print(f"  Warning: patch/8={patch_size_mm/8:.1f}mm < "
                  f"{bottleneck_half}*max_spacing="
                  f"{bottleneck_half * max_sp:.1f}mm.  Cannot satisfy "
                  f"both {min_bottleneck_kernel}³ and patch/8 constraints."
                  f"  Using growth={min_growth:.3f} "
                  f"(< 8 positions at bottleneck).")
            kernel_growth = min_growth
        else:
            # Prefer growth <= 2.0, but never below min_growth
            kernel_growth = max(min_growth, min(2.0, max_growth))
    else:
        kernel_growth = min(2.0, max(min_growth, 1.5))

    # --- Step 3: Set scales ---
    r_n = diameter * kernel_growth ** n / 2.0

    # Last scale = r_n / bottleneck_half:
    # For bottleneck_half=1 (kernel 3): scales[-1] = r_n, floor(r_n/eff) >= 1
    # For bottleneck_half=2 (kernel 5): scales[-1] = r_n/2, floor(r_n/eff) >= 2
    hi = r_n / bottleneck_half

    # First scale: pool kernel >= 2 for median fine axis
    min_per_case = [min(sp) for sp in spacings]
    lo = 2.0 * float(np.median(min_per_case))

    if lo >= hi:
        lo = hi / (2.0 ** (n - 1))

    if n == 1:
        scales = [hi]
    else:
        scales = list(np.geomspace(lo, hi, n))

    # Ensure strictly increasing
    for i in range(1, len(scales)):
        if scales[i] <= scales[i - 1]:
            scales[i] = scales[i - 1] * 2.0

    # --- Step 4: Report ---
    # Encoder RF: sum of kernel diameters = diameter * sum(growth^k, k=0..n)
    rf = diameter * sum(kernel_growth ** k for k in range(n + 1))
    positions = patch_size_mm / hi if patch_size_mm else None
    print(f"\n  Diameter: {diameter:.3f} mm, kernel_growth: {kernel_growth:.3f}")
    print(f"  Encoder RF: {rf:.1f} mm" +
          (f" ({rf / patch_size_mm:.0%} of patch)" if patch_size_mm else ""))
    print(f"  Bottleneck r_n: {r_n:.2f} mm, scales[-1]: {hi:.2f} mm"
          f" (max spacing: {max_sp:.1f} mm)" +
          (f"  [{positions:.1f} positions/axis]" if positions else ""))

    _print_scale_diagnostics(
        spacings, scales, diameter, kernel_trim_threshold, kernel_growth,
        kernel_trim_cross_section=kernel_trim_cross_section)

    return diameter, kernel_growth, scales


def _print_scale_diagnostics(
    spacings: List[Tuple[float, float, float]],
    scales: List[float],
    diameter: float,
    kernel_trim_threshold: float = 1.0,
    kernel_growth: float = 2.0,
    kernel_trim_cross_section: float = 0.0,
) -> None:
    """Print a diagnostic table evaluating a set of pooling scales.

    For each encoder level shows:
    - Number of distinct effective-spacing tuples (spacing groups)
    - Number of distinct per-level kernel-size tuples
    - Mean and max anisotropy ratio across cases
    """
    n_downsample = len(scales)

    print(f"\n  Pooling scales: {['%.2f' % s for s in scales]}")
    print(f"  {'Level':<22s} {'Groups':>7s} {'Kernels':>8s} "
          f"{'Aniso mean':>11s} {'Aniso max':>10s}")
    print(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*11} {'-'*10}")

    for k in range(n_downsample + 1):
        if k == 0:
            effs = [tuple(float(v) for v in sp) for sp in spacings]
            label = "L0 (input)"
        else:
            effs = []
            for sp in spacings:
                eff = []
                for v in sp:
                    if v < scales[k - 1]:
                        eff.append(math.floor(scales[k - 1] / v) * v)
                    else:
                        eff.append(float(v))
                effs.append(tuple(eff))
            label = f"L{k} (pool {scales[k-1]:.2f})"

        unique_effs = len(set(effs))

        # Per-level kernel sizes
        from irrepunet.models.layers import _trim_half_extent
        kernel_set = set()
        for sp in spacings:
            steps = _compute_steps_array(sp, scales)
            r = diameter * (kernel_growth ** k) / 2.0
            ks = []
            for s in steps[k]:
                half = math.floor(r / s)
                half = _trim_half_extent(half, s, r,
                                         kernel_trim_cross_section,
                                         kernel_trim_threshold)
                ks.append(2 * half + 1)
            kernel_set.add(tuple(ks))
        unique_kernels = len(kernel_set)

        ratios = [max(e) / min(e) for e in effs]
        mean_r = sum(ratios) / len(ratios)
        max_r = max(ratios)

        print(f"  {label:<22s} {unique_effs:>7d} {unique_kernels:>8d} "
              f"{mean_r:>11.2f} {max_r:>10.2f}")

    # Full architecture keys
    arch_keys = set()
    for sp in spacings:
        key = tuple(compute_kernel_sizes(
            diameter, sp, scales, kernel_trim_threshold, kernel_growth,
            kernel_trim_cross_section))
        arch_keys.add(key)
    print(f"\n  Unique architecture keys: {len(arch_keys)}")


def optimize_bottleneck_kernels(
    spacings: List[Tuple[float, float, float]],
    n_downsample: int,
    patch_size_mm: float,
    diameter: Optional[float] = None,
    target_bottleneck_kernel: int = 3,
    kernel_growth: float = 2.0,
) -> dict:
    """Find (diameter, scale, kernel_trim_threshold) for target-size bottleneck kernels.

    Achieves 5^3 pre-trim kernels (floor(r/eff)=2) at the bottleneck, trimmed
    to 3^3 via kernel_trim_threshold, on ALL axes of ALL spacing groups.
    This requires jointly searching over diameter and scale.

    Increasing diameter beyond the L0 minimum ensures that even thick-slice
    axes (e.g. 5mm) can achieve floor=2 at the bottleneck.

    Parameters
    ----------
    spacings : list of (float, float, float)
        All unique spacing tuples across resolution groups.
    n_downsample : int
        Number of pooling levels.
    patch_size_mm : float
        Minimum physical patch dimension in mm.
    diameter : float, optional
        Base kernel diameter.  If None, jointly optimized with scale.
    target_bottleneck_kernel : int
        Target kernel size at bottleneck (must be odd). Default 3.
    kernel_growth : float
        Per-level diameter growth factor (default 2.0).

    Returns
    -------
    dict with keys:
        diameter, scale, scales, kernel_trim_threshold, kernel_growth,
        kernel_sizes (per-group pre/post trim), diagnostics (str)
    """
    assert target_bottleneck_kernel % 2 == 1, "target must be odd"
    target_half = target_bottleneck_kernel // 2  # 1 for target=3
    target_pre_trim = target_half + 1  # 2 for target=3

    n = n_downsample
    a = kernel_growth ** n / 2.0  # r_n = d * a

    # --- Step 1: L0 minimum diameter ---
    second_finest = [sorted(sp)[1] for sp in spacings]
    d_min = 2.0 * max(second_finest) * (1.0 + 1e-9)

    if diameter is not None:
        if diameter < d_min:
            print(f"  Warning: diameter {diameter:.2f}mm < {d_min:.2f}mm "
                  f"(some L0 kernels will be < 3x3x1)")
        d_min = diameter

    # --- Step 2: Joint (d, scale_n) search ---
    # For each unique axis spacing s and pooling factor k, floor(r_n/(k*s)) = 2
    # requires d in [2*k*s/a, 3*k*s/a).  scale_n in [k*s, (k+1)*s).
    # Find smallest d >= d_min in the intersection over all axis spacings.
    all_axis_spacings = sorted(set(s for sp in spacings for s in sp))
    d_max = max(all_axis_spacings) * 8  # generous upper bound

    # Collect d-interval boundaries where floor values change
    d_boundaries = set()
    d_boundaries.add(d_min)
    for s in all_axis_spacings:
        for k in range(1, 500):
            d_lo = target_pre_trim * k * s / a
            d_hi = (target_pre_trim + 1) * k * s / a
            if d_lo > d_max:
                break
            d_boundaries.add(d_lo)
            d_boundaries.add(d_hi)
    d_boundaries = sorted(d_boundaries)

    # Test each d-interval for an all-axes solution
    best_d = None
    best_sn = None
    for d_cand in d_boundaries:
        d_test = d_cand + 1e-9
        if d_test < d_min:
            continue
        if d_test > d_max:
            break

        r_n_test = d_test * a

        # For each axis spacing, find valid scale_n intervals where floor=2
        sn_ranges = None  # running intersection
        all_valid = True

        for s in all_axis_spacings:
            s_ranges = []
            # Pooled: eff = k*s where k = floor(scale_n/s)
            for k in range(1, 500):
                eff = k * s
                if eff > r_n_test:
                    break
                if math.floor(r_n_test / eff) == target_pre_trim:
                    s_ranges.append((k * s, (k + 1) * s))
            # Unpooled: eff = s (when s >= scale_n)
            if s > 0 and math.floor(r_n_test / s) == target_pre_trim:
                s_ranges.append((0, s + 1e-9))

            if not s_ranges:
                all_valid = False
                break

            # Intersect with running result
            if sn_ranges is None:
                sn_ranges = s_ranges
            else:
                new_ranges = []
                for r1 in sn_ranges:
                    for r2 in s_ranges:
                        lo = max(r1[0], r2[0])
                        hi = min(r1[1], r2[1])
                        if lo < hi - 1e-9:
                            new_ranges.append((lo, hi))
                sn_ranges = new_ranges
                if not sn_ranges:
                    all_valid = False
                    break

        if all_valid and sn_ranges:
            best_sn = sn_ranges[0][0] + 1e-9  # smallest valid scale_n
            best_d = d_test
            break

    if best_d is not None:
        diameter = best_d
        scale_n = best_sn
    else:
        # Fallback: no all-axes solution found. Use d_min with fine-axes-only.
        diameter = d_min
        r_n_test = diameter * a
        # Binary search for smallest scale_n where max_floor <= target_pre_trim
        # on the two finest axes per group.
        fine_spacings = set()
        for sp in spacings:
            sorted_sp = sorted(sp)
            fine_spacings.update(sorted_sp[:2])
        lo_sn = min(fine_spacings) * 0.5
        hi_sn = r_n_test * 2.0
        for _ in range(200):
            mid = (lo_sn + hi_sn) / 2.0
            max_f = 0
            for sp in spacings:
                for s in sorted(sp)[:2]:
                    eff = math.floor(mid / s) * s if s < mid else s
                    max_f = max(max_f, math.floor(r_n_test / eff) if eff > 0 else 0)
            if max_f > target_pre_trim:
                lo_sn = mid
            else:
                hi_sn = mid
            if hi_sn - lo_sn < 1e-6:
                break
        scale_n = hi_sn
        print(f"  Note: no all-axes solution found; using fine-axes-only "
              f"(d={diameter:.4f}, scale_n={scale_n:.3f})")

    r_n = diameter * a
    scale = scale_n / (2 ** (n - 1)) if n > 0 else scale_n
    scales = [scale * (2 ** i) for i in range(n)]

    # --- Step 3: Compute cross-section trim threshold ---
    # For axes where pre-trim half == target_pre_trim (e.g. 2), compute the
    # cross-section fraction at that shell. Set threshold just above the max
    # so these shells get trimmed, while the inner shell (half-1) survives.
    cross_fracs_at_target = []
    for sp in spacings:
        steps = _compute_steps_array(sp, scales)
        for s in steps[n]:
            half = math.floor(r_n / s)
            if half == target_pre_trim:
                ratio_sq = (half * s / r_n) ** 2
                cross_fracs_at_target.append(math.sqrt(max(0.0, 1.0 - ratio_sq)))

    if cross_fracs_at_target:
        kernel_trim_cross_section = max(cross_fracs_at_target) + 1e-6
    else:
        kernel_trim_cross_section = 0.0

    # Also keep legacy threshold for backward compat reporting
    kernel_trim_threshold = 1.0

    # --- Step 4: Verify ---
    group_results = {}
    all_ok = True

    for sp in spacings:
        sizes_post = compute_kernel_sizes(
            diameter, sp, scales, kernel_trim_threshold, kernel_growth,
            kernel_trim_cross_section)
        sizes_pre = compute_kernel_sizes(
            diameter, sp, scales, 1.0, kernel_growth)
        group_results[sp] = {'pre': sizes_pre, 'post': sizes_post}

        # Check bottleneck: all axes should be target or 1
        bn = sizes_post[n]
        for k in bn:
            if k != 1 and k != target_bottleneck_kernel:
                all_ok = False

        # Check L0: two finest axes >= 3
        sorted_l0 = sorted(sizes_post[0])
        if sorted_l0[1] < 3:
            all_ok = False

    # --- Step 5: Diagnostics ---
    diag_lines = []
    diag_lines.append(f"Bottleneck optimization: target={target_bottleneck_kernel}^3")
    diag_lines.append(f"  diameter={diameter:.4f}mm, scale={scale:.4f}, "
                       f"kernel_trim_cross_section={kernel_trim_cross_section:.6f}")
    diag_lines.append(f"  r_n={r_n:.3f}mm, scale_n={scale_n:.3f}mm")
    diag_lines.append(f"  Verification: {'PASS' if all_ok else 'FAIL'}")
    diag_lines.append("")

    n_levels = n + 1
    header = "  " + f"{'Spacing':<22s}"
    for k in range(n_levels):
        label = f"L{k}" + (" (bn)" if k == n else "")
        header += f" {label:>12s}"
    diag_lines.append(header)
    diag_lines.append("  " + "-" * (22 + 13 * n_levels))

    for sp in sorted(group_results.keys()):
        res = group_results[sp]
        sp_str = f"({sp[0]:.2f},{sp[1]:.2f},{sp[2]:.2f})"
        row = f"  {sp_str:<22s}"
        for k in range(n_levels):
            post = res['post'][k]
            row += f" {post[0]:>3d}x{post[1]}x{post[2]}"
        diag_lines.append(row)

    diagnostics = "\n".join(diag_lines)
    print(diagnostics)

    return {
        'diameter': diameter,
        'scale': scale,
        'scales': scales,
        'kernel_trim_threshold': kernel_trim_threshold,
        'kernel_trim_cross_section': kernel_trim_cross_section,
        'kernel_growth': kernel_growth,
        'kernel_sizes': group_results,
        'diagnostics': diagnostics,
        'verified': all_ok,
    }


def compute_architecture_key(
    model: nn.Module,
    spacing: Tuple[float, float, float],
) -> Tuple[Tuple[int, ...], ...]:
    """Compute per-level kernel sizes for a given spacing.

    Two spacings with the same architecture key produce projected models
    with identical tensor shapes, so one projection can serve both.

    Returns a tuple of per-level kernel size tuples, e.g.:
    ((5, 1, 5), (5, 1, 5), (3, 3, 3), (3, 5, 3), (3, 3, 3), (3, 3, 3))
    """
    return tuple(compute_kernel_sizes(
        diameter=model.diameter,
        spacing=spacing,
        scales=model.scales,
        kernel_trim_threshold=getattr(model, 'kernel_trim_threshold', 1.0),
        kernel_growth=getattr(model, 'kernel_growth', 2.0),
        kernel_trim_cross_section=getattr(model, 'kernel_trim_cross_section', 0.0),
    ))


def _decompose_state_dict(state_dict, n_downsample, arch_key):
    """Decompose a projected model state_dict into per-level fragments.

    Returns a dict of {(block_prefix, kernel_size): sub_state_dict} for each
    encoder/decoder block, plus an "other" key for remaining parameters
    (output heads, etc.).

    Parameters
    ----------
    state_dict : dict
        Full state_dict from a projected model.
    n_downsample : int
        Number of pooling levels in the model.
    arch_key : tuple of tuple
        Per-level kernel sizes from compute_architecture_key().

    Returns
    -------
    dict
        Mapping from fragment key to sub-state_dict.
    """
    block_prefixes = set()
    for i in range(n_downsample + 1):
        block_prefixes.add(f"encoder.down_blocks.{i}.")
    for j in range(n_downsample):
        block_prefixes.add(f"decoder.up_blocks.{j}.")

    fragments = {}

    # Encoder blocks
    for i in range(n_downsample + 1):
        prefix = f"encoder.down_blocks.{i}."
        frag = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
        fragments[(f"encoder.down_blocks.{i}", arch_key[i])] = frag

    # Decoder blocks — up_blocks[j] operates at the same resolution as
    # encoder.down_blocks[n_downsample - 1 - j]
    for j in range(n_downsample):
        prefix = f"decoder.up_blocks.{j}."
        dec_level = n_downsample - 1 - j
        frag = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
        fragments[(f"decoder.up_blocks.{j}", arch_key[dec_level])] = frag

    # Everything else (output heads, input projections, etc.)
    other = {k: v for k, v in state_dict.items()
             if not any(k.startswith(p) for p in block_prefixes)}
    fragments["other"] = other

    return fragments


def _assemble_state_dict(level_fragments, arch_key, n_downsample):
    """Assemble a full state_dict from per-level fragments.

    Returns the assembled state_dict, or None if any fragment is missing.

    Parameters
    ----------
    level_fragments : dict
        Fragment cache: {(block_prefix, kernel_size): sub_state_dict, ...}
    arch_key : tuple of tuple
        Per-level kernel sizes for the target spacing.
    n_downsample : int
        Number of pooling levels.

    Returns
    -------
    dict or None
        Assembled state_dict, or None if incomplete coverage.
    """
    assembled = {}

    for i in range(n_downsample + 1):
        frag_key = (f"encoder.down_blocks.{i}", arch_key[i])
        if frag_key not in level_fragments:
            return None
        assembled.update(level_fragments[frag_key])

    for j in range(n_downsample):
        dec_level = n_downsample - 1 - j
        frag_key = (f"decoder.up_blocks.{j}", arch_key[dec_level])
        if frag_key not in level_fragments:
            return None
        assembled.update(level_fragments[frag_key])

    if "other" not in level_fragments:
        return None
    assembled.update(level_fragments["other"])

    return assembled


def export_hierarchical_bundle(
    model: nn.Module,
    checkpoint: dict,
    resolution_levels: List[Tuple[float, float, float]],
    output_path: str,
    use_2d: bool = True,
    native_spacings: Optional[List[Tuple[float, float, float]]] = None,
    min_cases: int = 1,
) -> str:
    """Export a checkpoint with pre-projected models for hierarchical inference.

    Projects the model at each coarse resolution level and stores the full
    projected model objects. Optionally pre-projects native spacings using
    per-level state_dict deduplication — two architectures sharing the same
    kernel size at a given level share that level's fragment, reducing storage
    from O(N_arch * model_size) to O(sum of unique fragments).

    Parameters
    ----------
    model : nn.Module
        Trained e3nn model (E3nnUNet).
    checkpoint : dict
        Original checkpoint dict (from torch.load).
    resolution_levels : list of (float, float, float)
        Coarse spacings to pre-project, e.g. [(5.0, 5.0, 5.0), (2.0, 2.0, 2.0)].
    output_path : str
        Path to save the bundle (.pt file).
    use_2d : bool
        Use Conv2d optimization for anisotropic kernels.
    native_spacings : list of (float, float, float), optional
        Native spacings to pre-project (e.g. from dataset scan).
        Grouped by architecture with per-level deduplication.
    min_cases : int
        Only bundle architectures seen by at least this many native spacings.

    Returns
    -------
    str
        Path to the saved bundle.
    """
    import time
    from collections import defaultdict

    bundle = dict(checkpoint)  # shallow copy
    projected_levels = {}

    # --- Coarse resolution levels (stored as full models) ---
    for spacing in resolution_levels:
        spacing = tuple(float(s) for s in spacing)
        t0 = time.time()
        projected = project_to_spacing(model, spacing, use_2d=use_2d)
        dt = time.time() - t0
        n_params = sum(p.numel() for p in projected.parameters()) / 1e6
        projected_levels[spacing] = projected.cpu()
        print(f"  Projected {spacing} in {dt:.1f}s ({n_params:.1f}M params)")

    bundle['projected_levels'] = projected_levels
    bundle['hierarchical_resolution_levels'] = resolution_levels

    # --- Native spacings with per-level dedup ---
    if native_spacings:
        n_downsample = model.n_downsample
        arch_groups = defaultdict(list)
        for sp in native_spacings:
            sp = tuple(float(s) for s in sp)
            arch = compute_architecture_key(model, sp)
            arch_groups[arch].append(sp)

        # Filter by min_cases
        arch_groups = {
            arch: sps for arch, sps in arch_groups.items()
            if len(sps) >= min_cases
        }
        print(f"\n  Native spacings: {len(native_spacings)} total, "
              f"{len(arch_groups)} unique architectures (min_cases={min_cases})")

        level_fragments = {}  # {(block_prefix, kernel_size): sub_state_dict}
        arch_registry = {}   # {arch_key: representative_spacing}
        n_projections = 0
        n_reused_full = 0
        total_blocks = (n_downsample + 1) + n_downsample  # enc + dec blocks

        for arch, sps in sorted(arch_groups.items(), key=lambda x: -len(x[1])):
            representative = sps[0]
            arch_registry[arch] = representative

            # Count how many fragments already exist for this architecture
            existing = 0
            for i in range(n_downsample + 1):
                if (f"encoder.down_blocks.{i}", arch[i]) in level_fragments:
                    existing += 1
            for j in range(n_downsample):
                dec_level = n_downsample - 1 - j
                if (f"decoder.up_blocks.{j}", arch[dec_level]) in level_fragments:
                    existing += 1

            if existing == total_blocks and "other" in level_fragments:
                n_reused_full += 1
                print(f"    arch={arch[0]}..{arch[-1]} n={len(sps):3d} "
                      f"-> all {total_blocks} block fragments reused")
                continue

            # Project to get weights for new fragments
            t0 = time.time()
            projected = project_to_spacing(model, representative, use_2d=use_2d)
            dt = time.time() - t0
            sd = projected.state_dict()
            n_projections += 1

            # Decompose and deduplicate
            new_fragments = _decompose_state_dict(sd, n_downsample, arch)
            new_count = 0
            for frag_key, frag_sd in new_fragments.items():
                if frag_key not in level_fragments:
                    level_fragments[frag_key] = frag_sd
                    new_count += 1

            del projected
            print(f"    arch={arch[0]}..{arch[-1]} n={len(sps):3d} "
                  f"projected in {dt:.1f}s ({new_count} new, {existing} reused)")

        bundle['level_fragments'] = level_fragments
        bundle['arch_registry'] = arch_registry
        bundle['n_downsample'] = n_downsample

        # Remove old-format key if present
        bundle.pop('native_projections', None)

        # Stats
        total_frag_params = sum(
            sum(v.numel() for v in frag.values())
            for frag in level_fragments.values()
            if isinstance(frag, dict)
        )
        n_covered = sum(len(sps) for sps in arch_groups.values())
        print(f"\n  Per-level dedup: {len(level_fragments)} unique fragments, "
              f"{total_frag_params / 1e6:.1f}M total params")
        print(f"  {n_projections} projections needed, "
              f"{n_reused_full} architectures fully reused")
        print(f"  Covers {n_covered}/{len(native_spacings)} spacings")

    torch.save(bundle, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved hierarchical bundle to {output_path} ({size_mb:.0f}MB)")
    return output_path
