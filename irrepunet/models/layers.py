"""Equivariant layers for e3nnUNet."""

import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from e3nn import o3
from e3nn.nn import BatchNorm, Dropout, Gate
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace, soft_unit_step


def _precompute_sc_layout(irreps_in, irreps_out):
    """Precompute block-diagonal layout for self-connection weight matrix.

    Matches e3nn.o3.Linear instruction generation: all (i_in, i_out) pairs
    where irreps_in[i_in].ir == irreps_out[i_out].ir, ordered by (i_in, i_out).

    Normalization: alpha = 1 / sqrt(total_mul_in) per output group, where
    total_mul_in sums multiplicities of ALL input groups connecting to it.

    Returns list of (w_offset, in_offset, out_offset, mul_in, mul_out, ir_dim, alpha)
    """
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)

    # Build offset tables
    in_offsets, offset = [], 0
    for mul, ir in irreps_in:
        in_offsets.append(offset)
        offset += mul * ir.dim

    out_offsets, offset = [], 0
    for mul, ir in irreps_out:
        out_offsets.append(offset)
        offset += mul * ir.dim

    # Compute total_mul_in for each output group (for normalization)
    total_mul_in = {}
    for i_in, (mul_in, ir_in) in enumerate(irreps_in):
        for i_out, (mul_out, ir_out) in enumerate(irreps_out):
            if ir_in == ir_out:
                total_mul_in[i_out] = total_mul_in.get(i_out, 0) + mul_in

    # Generate instructions in e3nn order (outer: i_in, inner: i_out)
    layout = []
    w_offset = 0
    for i_in, (mul_in, ir_in) in enumerate(irreps_in):
        for i_out, (mul_out, ir_out) in enumerate(irreps_out):
            if ir_in == ir_out:
                ir_dim = ir_in.dim
                alpha = 1.0 / math.sqrt(total_mul_in[i_out])
                layout.append((
                    w_offset, in_offsets[i_in], out_offsets[i_out],
                    mul_in, mul_out, ir_dim, alpha,
                ))
                w_offset += mul_in * mul_out
    return layout


def _build_sc_weight_matrix(weight, layout, out_dim, in_dim):
    """Build dense (out_dim, in_dim) SC weight matrix from block-diagonal layout.

    For each instruction: places alpha * kron(W.T, I_{ir_dim}) at the
    appropriate (out_offset, in_offset) block. Multiple instructions may
    target the same output rows but different input columns.
    """
    w = weight.reshape(-1)
    out = w.new_zeros(out_dim, in_dim)
    for w_off, in_off, out_off, mul_in, mul_out, ir_dim, alpha in layout:
        W = w[w_off:w_off + mul_in * mul_out].reshape(mul_in, mul_out)
        WT = alpha * W.T  # (mul_out, mul_in)
        if ir_dim == 1:
            out[out_off:out_off + mul_out, in_off:in_off + mul_in] = WT
        else:
            for c in range(ir_dim):
                out[out_off + c:out_off + mul_out * ir_dim:ir_dim,
                    in_off + c:in_off + mul_in * ir_dim:ir_dim] = WT
    return out


def get_voxel_convolution(backend="e3nn", pyramid=False, **kwargs):
    """Factory function to create a voxel convolution.

    Parameters
    ----------
    backend : str
        Backend to use. Only ``"e3nn"`` is supported.
    pyramid : bool or dict
        If True, use PyramidVoxelConvolution with default settings.
        If a dict, use as keyword arguments for pyramid-specific params
        (num_pyramid_levels, pyramid_decay, pyramid_mode).
    **kwargs
        Arguments passed to the convolution constructor.

    Returns
    -------
    nn.Module
        VoxelConvolution instance.
    """
    if backend != "e3nn":
        raise ValueError(f"Unknown backend: {backend}. Only 'e3nn' is supported.")

    if pyramid:
        pyramid_kwargs = pyramid if isinstance(pyramid, dict) else {}
        return PyramidVoxelConvolution(**kwargs, **pyramid_kwargs)

    return VoxelConvolution(**kwargs)


def _pool_factor(scale, step):
    """Compute integer pooling factor for one axis, robust to float imprecision.

    Uses round() when scale/step is within 1% of an integer to prevent
    math.floor(1.99999984) = 1 when the true ratio is 2.0.
    """
    if step >= scale:
        return 1
    raw = scale / step
    rounded = round(raw)
    if rounded >= 1 and abs(raw - rounded) < 0.01 * rounded:
        return rounded
    return max(math.floor(raw), 1)


class NormSoftClamp(nn.Module):
    """Self-normalizing activation for l>0 irrep features.

    Uses tanh-based soft clamping on log(norm_sq / target) to provide
    bidirectional contraction with bounded output. l=0 scalars pass through
    unchanged (standard SELU in the Gate handles those).

    For each l>0 irrep instance with dimension d = 2l+1:
        norm_sq = ||f||^2
        target_ns = d * exp(log_target)
        log_ratio = log(norm_sq / target_ns)
        compressed = tanh(steepness * log_ratio) / steepness
        new_ns = target_ns * exp(compressed)
        f_out = f * sqrt(new_ns / norm_sq)

    Key properties:
    - Bidirectional contraction (dampens large, amplifies small)
    - Bounded output: max log-deviation = 1/steepness
    - Near-identity near equilibrium (preserves gradient information)
    - Learnable target per irrep l-type (adapts to actual network dynamics)
    - Fixed point at var/comp ~1.03 with init_target_var=2.7

    Input shape: (batch, ..., irreps.dim)  [channel-last, as used by e3nn]

    Parameters
    ----------
    irreps : Irreps
        Irreducible representations (post-Gate output).
    steepness : float
        Controls contraction strength. Higher = more precise but less smooth.
    init_target_var : float
        Initial target per-component variance. 2.7 is calibrated for sigmoid gating.
    eps : float
        Small constant for numerical stability.
    """

    def __init__(self, irreps, steepness: float = 1.0, init_target_var: float = 2.7,
                 eps: float = 1e-8):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.steepness = steepness
        self.eps = eps

        # Build vectorized group info: (start, end, mul, dim, target_idx)
        # Groups by (mul, ir) block instead of per-instance, reducing iterations
        # from O(total_instances) to O(num_irrep_types)
        groups = []
        l_types_seen = {}
        log_target_list = []
        idx = 0

        for mul, ir in self.irreps:
            total = mul * ir.dim
            if ir.l == 0:
                groups.append((idx, idx + total, mul, ir.dim, -1))
            else:
                if ir.l not in l_types_seen:
                    l_types_seen[ir.l] = len(log_target_list)
                    log_target_list.append(math.log(init_target_var))
                groups.append((idx, idx + total, mul, ir.dim, l_types_seen[ir.l]))
            idx += total

        self._groups = groups

        if log_target_list:
            self.log_target = nn.Parameter(
                torch.tensor(log_target_list, dtype=torch.float32)
            )
        else:
            self.log_target = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps}, s={self.steepness})"

    def forward(self, x):
        """x: (batch, ..., C) channel-last."""
        chunks = []
        for start, end, mul, dim, tidx in self._groups:
            if dim == 1:
                # l=0 scalars: passthrough
                chunks.append(x[..., start:end])
            else:
                # l>0: reshape to (..., mul, dim), process all instances at once
                feat = x[..., start:end].unflatten(-1, (mul, dim))
                target_var = torch.exp(self.log_target[tidx])
                target_ns = dim * target_var
                norm_sq = (feat ** 2).sum(dim=-1, keepdim=True)
                log_ratio = torch.log(norm_sq.clamp(min=self.eps) / target_ns)
                compressed = torch.tanh(self.steepness * log_ratio) / self.steepness
                new_ns = target_ns * torch.exp(compressed)
                scale = torch.sqrt(new_ns / norm_sq.clamp(min=self.eps))
                chunks.append((feat * scale).flatten(-2))
        return torch.cat(chunks, dim=-1)


# Backward-compatible alias
NormSELU = NormSoftClamp


class FusedGate(nn.Module):
    """Gate activation that derives gate values from l=0 features via linear projection.

    Unlike e3nn's Gate, which requires the preceding conv to output extra l=0 channels
    solely for gating l>0 features, FusedGate derives gate values from the existing
    l=0 scalar features via a small learned linear projection. This means the conv
    outputs only irreps_hidden (no extra gate channels).

    The module is equivariant because l=0 features are rotation-invariant, so the
    linear projection and sigmoid produce rotation-invariant gates, which scale l>0
    features without breaking equivariance.

    Parameters
    ----------
    irreps : Irreps
        Input (and output) irreps. Must contain l=0 scalars.
    scalar_activation : callable
        Activation function for l=0 scalars (e.g., SiLU, softplus).
    """

    def __init__(self, irreps, scalar_activation):
        super().__init__()
        self.irreps_in = Irreps(irreps)
        self.irreps_out = Irreps(irreps)

        # Split into scalar and gated parts
        self.irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_in if ir.l == 0])
        self.irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_in if ir.l > 0])

        self.n_scalars = self.irreps_scalars.dim
        self.n_gated = self.irreps_gated.dim
        self.n_gate_values = self.irreps_gated.num_irreps  # one gate per irrep instance

        # Store structure for per-irrep gate expansion: [(mul, 2l+1), ...]
        self.gated_irreps_structure = [
            (mul, 2 * ir.l + 1) for mul, ir in self.irreps_gated
        ]

        # Scalar activation (applied to l=0 features)
        self.scalar_activation = scalar_activation

        # Linear projection: l=0 scalars -> gate values (one per l>0 irrep instance)
        if self.n_gate_values > 0:
            self.gate_linear = nn.Linear(self.n_scalars, self.n_gate_values, bias=False)
        else:
            self.gate_linear = None

    def forward(self, x):
        """Forward pass. x shape: (..., irreps.dim) channel-last."""
        scalars = x[..., :self.n_scalars]
        activated_scalars = self.scalar_activation(scalars)

        if self.gate_linear is None:
            return activated_scalars

        gated_features = x[..., self.n_scalars:]

        # Derive gate values from raw scalars (not activated)
        gate_values = torch.sigmoid(self.gate_linear(scalars))

        # Expand each gate value to cover the (2l+1) components of its irrep
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


class EquivariantLayerNorm(nn.Module):
    """Per-irrep LayerNorm using torch.nn.LayerNorm.

    Each irrep type (mul, l) gets its own LayerNorm over its mul*(2l+1)
    features, matching how e3nn's BatchNorm handles irreps separately.
    No running statistics — identical at train and eval time.

    Parameters
    ----------
    irreps : o3.Irreps
        Irreducible representations to normalize.
    eps : float
        Small constant for numerical stability.
    affine : bool
        If True, apply learnable per-element scale and bias.
    normalization : str
        Kept for API compatibility; ignored.
    """

    def __init__(
        self,
        irreps,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = 'component',
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.slices = []
        lns = []
        ix = 0
        for mul, ir in self.irreps:
            d = mul * ir.dim
            self.slices.append((ix, ix + d))
            lns.append(nn.LayerNorm(d, eps=eps, elementwise_affine=affine))
            ix += d
        self.lns = nn.ModuleList(lns)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps})"

    def forward(self, input):
        """Normalize input tensor. Shape ``(batch, ..., irreps.dim)``."""
        chunks = []
        for (start, end), ln in zip(self.slices, self.lns):
            chunks.append(ln(input[..., start:end]))
        return torch.cat(chunks, dim=-1)


def _trim_half_extent(s, step, r, cross_section_threshold, old_threshold):
    """Trim kernel half-extent based on cross-section fraction or legacy threshold.

    For each axis, compute the sphere's cross-section fraction at the outermost
    voxel: cross_frac = sqrt(1 - (s * step / r)^2). Trim (decrement s) while
    cross_frac < cross_section_threshold.

    Parameters
    ----------
    s : int
        Current half-extent (kernel size = 2*s + 1).
    step : float
        Effective voxel spacing on this axis (mm).
    r : float
        Kernel radius (mm).
    cross_section_threshold : float
        Minimum cross-section fraction to keep a shell (0.0 = disabled).
        When > 0, takes priority over old_threshold.
    old_threshold : float
        Legacy trim threshold (trim when s*step/r > old_threshold).
        Only used when cross_section_threshold == 0.

    Returns
    -------
    int
        Trimmed half-extent.
    """
    if cross_section_threshold > 0:
        while s > 0:
            ratio_sq = (s * step / r) ** 2
            cross_frac = math.sqrt(max(0.0, 1.0 - ratio_sq))
            if cross_frac < cross_section_threshold:
                s -= 1
            else:
                break
    elif old_threshold < 1.0:
        if s > 0 and (s * step) / r > old_threshold:
            s -= 1
    return s


class VoxelConvolution(nn.Module):
    """Equivariant convolution on voxels with physical units.

    Parameters
    ----------
    irreps_in : str or Irreps
        Input irreducible representations
    irreps_out : str or Irreps
        Output irreducible representations
    irreps_sh : str or Irreps
        Spherical harmonics irreps (typically o3.Irreps.spherical_harmonics(lmax))
    diameter : float
        Diameter of the filter in physical units
    num_radial_basis : int
        Number of radial basis functions
    steps : tuple of float
        Size of the voxel in physical units (spacing)
    cutoff : bool
        Whether to apply cutoff to basis functions
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        diameter: float,
        num_radial_basis: int,
        steps: tuple = (1.0, 1.0, 1.0),
        cutoff: bool = True,
        in1_var: list = None,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sequential_sc: bool = False,
        sc_mode: str = None,
        sphere_norm: bool = True,
        **kwargs
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.num_radial_basis = num_radial_basis
        self.diameter = diameter
        self.cutoff = cutoff
        self.kernel_trim_threshold = kernel_trim_threshold
        self.kernel_trim_cross_section = kernel_trim_cross_section
        self.sphere_norm = sphere_norm
        # sc_mode: "parallel", "sc_first", "conv_first", "sc_first_res", "conv_first_res", "none"
        if sc_mode is not None:
            self.sc_mode = sc_mode
        elif sequential_sc:
            self.sc_mode = "sc_first"
        else:
            self.sc_mode = "parallel"

        # When sc_mode="none", drop self-connection entirely.
        # Requires cutoff=False so RBF(0)!=0 and center weight is learnable.
        if self.sc_mode == "none":
            self.cutoff = False
            self.sc = None
        elif self.sc_mode in ("conv_first", "conv_first_res"):
            # Conv operates on irreps_in, sc operates on conv output (irreps_out)
            self.sc = Linear(self.irreps_out, self.irreps_out)
        else:
            self.sc = Linear(self.irreps_in, self.irreps_out)

        # Build the lattice and buffers for the initial spacing
        self._build_lattice_buffers(steps, kwargs)
        self.kwargs = kwargs

        # Tensor product for combining input with spherical harmonics
        # In sc_first/sc_first_res, conv operates on sc output (irreps_out).
        # In parallel/conv_first/conv_first_res, conv operates on raw input (irreps_in).
        sc_first_modes = ("sc_first", "sc_first_res")
        tp_in = self.irreps_out if self.sc_mode in sc_first_modes else self.irreps_in
        tp_kwargs = dict(shared_weights=False, compile_right=True)
        if in1_var is not None and self.sc_mode not in sc_first_modes:
            tp_kwargs['in1_var'] = in1_var
        self.tp = FullyConnectedTensorProduct(
            tp_in,
            self.irreps_sh,
            self.irreps_out,
            **tp_kwargs
        )

        # Learnable weights
        self.weight = nn.Parameter(torch.randn(self.num_radial_basis, self.tp.weight_numel))

    def _build_lattice_buffers(self, steps, kwargs=None):
        """Build or rebuild lattice-dependent buffers for given spacing.

        Respects the dtype of existing buffers (e.g., FP16) when rebuilding.
        """
        r = self.diameter / 2

        # Detect target dtype and device from existing buffers or parameters
        target_dtype = None
        target_device = None
        if hasattr(self, 'lattice') and self.lattice is not None:
            target_dtype = self.lattice.dtype
            target_device = self.lattice.device
        elif hasattr(self, 'weight') and self.weight is not None:
            target_dtype = self.weight.dtype
            target_device = self.weight.device

        axes = []
        for i in range(3):
            s = math.floor(r / steps[i])
            s = _trim_half_extent(s, steps[i], r,
                                  self.kernel_trim_cross_section,
                                  self.kernel_trim_threshold)
            axes.append(torch.arange(-s, s + 1.0) * steps[i])
        x, y, z = axes

        lattice = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)

        # Update or register the buffer (preserve dtype)
        if hasattr(self, 'lattice') and self.lattice is not None:
            lattice = lattice.to(device=target_device, dtype=target_dtype)
            self.lattice = lattice
        else:
            self.register_buffer('lattice', lattice)

        # Update padding in kwargs
        if kwargs is not None and 'padding' not in kwargs:
            kwargs['padding'] = tuple(s // 2 for s in lattice.shape[:3])
        elif hasattr(self, 'kwargs'):
            self.kwargs['padding'] = tuple(s // 2 for s in lattice.shape[:3])

        # Radial basis function embedding
        emb = soft_one_hot_linspace(
            x=lattice.float().norm(dim=-1),  # Compute in FP32 for precision
            start=0.0,
            end=r,
            number=self.num_radial_basis,
            basis='smooth_finite',
            cutoff=self.cutoff,
        )
        if hasattr(self, 'emb') and self.emb is not None:
            emb = emb.to(device=target_device, dtype=target_dtype)
            self.emb = emb
        else:
            self.register_buffer('emb', emb)

        # Spherical harmonics (compute in FP32 for precision, then convert)
        sh = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=lattice.float(),  # Compute in FP32 for precision
            normalize=True,
            normalization='component'
        )
        if hasattr(self, 'sh') and self.sh is not None:
            sh = sh.to(device=target_device, dtype=target_dtype)
            self.sh = sh
        else:
            self.register_buffer('sh', sh)

    def update_spacing(self, steps: tuple):
        """Update lattice buffers for new spacing without changing learnable params."""
        self._build_lattice_buffers(steps)

    def kernel(self):
        """Compute the convolution kernel."""
        weight = self.emb @ self.weight
        if self.sphere_norm:
            weight = weight / self._n_kernel_voxels()
        else:
            weight = weight / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2])
        kernel = self.tp.right(self.sh, weight)
        kernel = torch.einsum('xyzio->oixyz', kernel)
        return kernel

    def _n_kernel_voxels(self):
        """Count voxels inside the kernel radius (sphere, not cuboid)."""
        r = self.diameter / 2
        dists = self.lattice.float().norm(dim=-1)  # (x, y, z)
        return (dists <= r).sum().item()

    def _sc_weight_matrix(self):
        """Extract dense weight matrix from self-connection (differentiable).

        Passes identity through the e3nn Linear. Returns (out_dim, in_dim)
        matrix W where sc(x) = W @ x. Autograd-compatible for training.
        """
        in_dim = self.sc.irreps_in.dim
        identity = torch.eye(
            in_dim,
            device=self.sc.weight.device,
            dtype=self.sc.weight.dtype,
        )
        return self.sc(identity).T

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, irreps_in.dim, x, y, z)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, irreps_out.dim, x, y, z)
        """
        # 1x1x1 kernel: radial basis is zero at origin (when cutoff=True),
        # so conv3d output is exactly zero. Skip kernel computation and conv3d entirely.
        # With cutoff=False (sc_mode="none"), center has nonzero RBF so kernel is valid.
        is_1x1x1 = all(s == 1 for s in self.lattice.shape[:3])

        if self.sc_mode == "none":
            # No self-connection: cutoff=False ensures center weight is nonzero
            if is_1x1x1 and self.cutoff:
                # Shouldn't happen (sc_mode=none forces cutoff=False), but safety
                b = x.shape[0]
                spatial = x.shape[2:]
                return x.new_zeros(b, self.irreps_out.dim, *spatial)
            return F.conv3d(x, self.kernel(), **self.kwargs)
        elif self.sc_mode == "sc_first":
            sc = self.sc(x.transpose(1, 4)).transpose(1, 4)
            if is_1x1x1:
                return sc
            return F.conv3d(sc, self.kernel(), **self.kwargs)
        elif self.sc_mode == "sc_first_res":
            # sc then conv, with residual skip around the conv
            sc = self.sc(x.transpose(1, 4)).transpose(1, 4)
            if is_1x1x1:
                return sc
            return sc + F.conv3d(sc, self.kernel(), **self.kwargs)
        elif self.sc_mode == "conv_first":
            if is_1x1x1:
                # Conv output is zero (RBF=0 at origin), sc(0)=0 (no bias)
                b = x.shape[0]
                spatial = x.shape[2:]
                return x.new_zeros(b, self.irreps_out.dim, *spatial)
            conv_out = F.conv3d(x, self.kernel(), **self.kwargs)
            return self.sc(conv_out.transpose(1, 4)).transpose(1, 4)
        elif self.sc_mode == "conv_first_res":
            # conv then sc, with residual skip around the sc
            if is_1x1x1:
                b = x.shape[0]
                spatial = x.shape[2:]
                return x.new_zeros(b, self.irreps_out.dim, *spatial)
            conv_out = F.conv3d(x, self.kernel(), **self.kwargs)
            return conv_out + self.sc(conv_out.transpose(1, 4)).transpose(1, 4)
        else:
            # Parallel: fuse sc weights into kernel center.
            # Instead of computing sc(x) and conv(x) separately — which
            # requires two B×C_out×D×H×W tensors alive for the addition —
            # we add the sc weight matrix to the kernel's center voxel and
            # run a single conv3d.  Mathematically equivalent because with
            # cutoff=True the RBF is zero at the origin, so the kernel
            # center is zero and the sc fills it in.
            sc_w = self._sc_weight_matrix()  # (out_dim, in_dim)
            if is_1x1x1:
                # Kernel is all zeros. Just apply sc as 1x1x1 conv.
                return F.conv3d(x, sc_w[:, :, None, None, None], **self.kwargs)
            kernel = self.kernel()
            cx = kernel.shape[2] // 2
            cy = kernel.shape[3] // 2
            cz = kernel.shape[4] // 2
            # Differentiable center injection: broadcast sc_w into a
            # kernel-shaped tensor that is nonzero only at the center,
            # then add to the spatial kernel.
            mask = kernel.new_zeros(kernel.shape[2:])
            mask[cx, cy, cz] = 1.0
            if self.sphere_norm:
                # Scale sc_w by kernel voxel count (inside radius) to match
                # the normalization applied to convolution weights in kernel().
                n_voxels = self._n_kernel_voxels()
                kernel = kernel + (sc_w[:, :, None, None, None] / n_voxels) * mask
            else:
                kernel = kernel + sc_w[:, :, None, None, None] * mask
            return F.conv3d(x, kernel, **self.kwargs)


class PyramidVoxelConvolution(VoxelConvolution):
    """VoxelConvolution with multi-scale kernel pyramid.

    Constructs a pyramid of kernels at decreasing resolution — e.g. for
    a 21³ native kernel the pyramid might be 3³ → 5³ → 11³ → 21³.
    Each coarse level is obtained by **subsampling the native lattice at
    integer strides**, so the coarse lattice points are an exact subset
    of the native grid.

    Two modes control how the coarse levels are fused back:

    ``"scatter"`` (default)
        Each voxel in the native kernel is weighted by the sum of
        ``alpha_k`` for every pyramid level whose stride lands on it.
        This is a precomputed per-voxel weight mask applied as a
        pointwise multiply — equivalent to a sum of dilated convolutions,
        but with zero loop overhead.  ``tp.right`` runs once; the mask
        multiply is the only extra cost.

    ``"interp"``
        Slice the native kernel at stride ``d``, trilinear-upscale back.
        Produces a smooth, low-pass-filtered version at each level.
        Slightly more expensive (K interpolation ops) but avoids the
        sparse zeros between strided samples.

    The pyramid is anisotropic: per-axis strides are clamped so that axes
    with small half-extent (e.g. the slice direction in thick-slice data)
    are not subsampled beyond half-extent 1.

    A single ``F.conv3d`` is applied in both modes.

    Parameters
    ----------
    (all VoxelConvolution parameters, plus:)
    num_pyramid_levels : int
        Requested number of kernel scales (including native).  Actual
        count may be lower after deduplication.  Default 4.
    pyramid_decay : float
        Geometric decay per level away from the coarsest.  Level 0
        (coarsest) gets weight 1.0; level k gets ``decay ** k``.
        Default 1.0 (equal weight per level).  Weights are non-learnable.
    pyramid_mode : str
        ``"scatter"`` (default) or ``"interp"``.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        diameter: float,
        num_radial_basis: int,
        steps: tuple = (1.0, 1.0, 1.0),
        cutoff: bool = True,
        in1_var: list = None,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sequential_sc: bool = False,
        sc_mode: str = None,
        sphere_norm: bool = True,
        # ---- pyramid-specific ----
        num_pyramid_levels: int = 4,
        pyramid_decay: float = 1.0,
        pyramid_mode: str = "scatter",
        **kwargs,
    ):
        if pyramid_mode not in ("interp", "scatter"):
            raise ValueError(
                f"pyramid_mode must be 'interp' or 'scatter', got '{pyramid_mode}'"
            )
        self.num_pyramid_levels = num_pyramid_levels
        self.pyramid_mode = pyramid_mode
        self._pyramid_steps = steps

        super().__init__(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            irreps_sh=irreps_sh,
            diameter=diameter,
            num_radial_basis=num_radial_basis,
            steps=steps,
            cutoff=cutoff,
            in1_var=in1_var,
            kernel_trim_threshold=kernel_trim_threshold,
            kernel_trim_cross_section=kernel_trim_cross_section,
            sequential_sc=sequential_sc,
            sc_mode=sc_mode,
            sphere_norm=sphere_norm,
            **kwargs,
        )

        # Allocate fixed (non-learnable) per-level weights.  Always non-learnable:
        # learned weights duplicated the network's existing radial-basis flexibility
        # and didn't measurably help.
        K_max = num_pyramid_levels
        # Level 0 = coarsest (dominant), level K-1 = native (residual).
        init_alpha = [pyramid_decay ** k for k in range(K_max)]
        logits = [math.log(math.exp(w) - 1) if w > 0.05 else w
                  for w in init_alpha]
        self.register_buffer('pyramid_logits', torch.tensor(logits))

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_pyramid_levels(native_half, num_levels):
        """Compute per-axis integer strides for each pyramid level.

        Returns
        -------
        level_strides : list of (d0, d1, d2) tuples, coarsest first
        level_slices  : list of (slice0, slice1, slice2) tuples for
                        indexing into the native kernel
        """
        seen = set()
        level_strides = []
        level_slices = []

        for d_global in range(num_levels, 0, -1):
            per_stride = []
            per_slice = []
            for h in native_half:
                d = min(d_global, h) if h > 1 else 1
                ch = h // d
                ch = max(ch, 1)
                start = h - ch * d
                per_stride.append(d)
                per_slice.append(slice(start, start + 2 * ch * d + 1, d))
            key = tuple(per_stride)
            if key in seen:
                continue
            seen.add(key)
            level_strides.append(key)
            level_slices.append(tuple(per_slice))

        return level_strides, level_slices

    # ------------------------------------------------------------------
    @staticmethod
    def _build_scatter_mask(native_shape, level_slices):
        """Build the per-voxel weight-index mask for scatter mode.

        Returns an integer tensor of shape ``(Nx, Ny, Nz)`` where each
        voxel stores a bitmask of which pyramid levels land on it.
        At runtime: ``weight = sum of alpha_k for each set bit k``.

        Actually, we store a float membership matrix
        ``(Nx, Ny, Nz, K)`` so the alpha dot product is differentiable.
        """
        K = len(level_slices)
        membership = torch.zeros(*native_shape, K)
        for level, sl in enumerate(level_slices):
            membership[sl[0], sl[1], sl[2], level] = 1.0
        return membership

    # ------------------------------------------------------------------
    def _build_lattice_buffers(self, steps, kwargs=None):
        """Build native lattice buffers, pyramid slices, and scatter mask."""
        super()._build_lattice_buffers(steps, kwargs)
        self._pyramid_steps = steps

        if not hasattr(self, 'num_pyramid_levels'):
            self._actual_K = 1
            self._level_strides = [(1, 1, 1)]
            self._level_slices = [(slice(None), slice(None), slice(None))]
            return

        native_half = [(self.lattice.shape[i] - 1) // 2 for i in range(3)]
        strides, slices = self._compute_pyramid_levels(
            native_half, self.num_pyramid_levels)
        self._actual_K = len(strides)
        self._level_strides = strides
        self._level_slices = slices

        # Precompute scatter membership: (Nx, Ny, Nz, K)
        native_shape = tuple(self.lattice.shape[:3])
        membership = self._build_scatter_mask(native_shape, slices)
        membership = membership.to(dtype=self.lattice.dtype,
                                   device=self.lattice.device)
        if hasattr(self, '_scatter_membership') and self._scatter_membership is not None:
            self._scatter_membership = membership
        else:
            self.register_buffer('_scatter_membership', membership)

    # ------------------------------------------------------------------
    @staticmethod
    def _upscale_kernel_to_native(kernel_k, target_shape):
        """Trilinear-upscale a coarse kernel to native spatial shape."""
        coarse_shape = kernel_k.shape[:3]
        if coarse_shape == target_shape:
            return kernel_k
        C_in, C_out = kernel_k.shape[3], kernel_k.shape[4]
        k = (kernel_k.permute(3, 4, 0, 1, 2)
             .reshape(C_in * C_out, 1, *coarse_shape))
        k_up = F.interpolate(
            k.float(), size=target_shape, mode='trilinear', align_corners=True,
        ).to(kernel_k.dtype)
        return k_up.reshape(C_in, C_out, *target_shape).permute(2, 3, 4, 0, 1)

    def kernel(self):
        """Compute the fused pyramid kernel at native resolution.

        ``tp.right`` runs **once** on the full native lattice.
        """
        # Clamp K to available logits (spacing changes can increase K
        # beyond the parameter size allocated at __init__).
        K = min(self._actual_K, self.pyramid_logits.shape[0])
        alpha = F.softplus(self.pyramid_logits[:K])
        native_shape = tuple(self.lattice.shape[:3])

        # Single TP evaluation
        weight = self.emb @ self.weight
        native_kernel = self.tp.right(self.sh, weight)

        if self.pyramid_mode == "scatter":
            # Dot the per-voxel membership with alpha → per-voxel scalar weight
            # membership: (Nx, Ny, Nz, K_actual),  alpha: (K,)
            mem = self._scatter_membership[..., :K]
            w = mem @ alpha                             # (Nx, Ny, Nz)
            # Normalise w so its RMS over sphere voxels = 1.
            # This preserves output variance: a concentrated (capped) kernel
            # and a full kernel produce outputs with the same scale.
            r = self.diameter / 2
            sphere_mask = self.lattice.float().norm(dim=-1) <= r
            n_sphere = sphere_mask.sum()
            if n_sphere > 0:
                w_rms = ((w ** 2 * sphere_mask.float()).sum() / n_sphere).sqrt()
                w = w / (w_rms + 1e-8)
            fused = native_kernel * w[..., None, None]
        else:
            fused = self._kernel_interp(native_kernel, native_shape, K, alpha)

        # Standard sphere_norm — same as VoxelConvolution.
        # For scatter mode the w-normalisation already ensures the pyramid
        # weighting averages to 1 over the sphere, so this division has the
        # same effect as in the base class.
        if self.sphere_norm:
            fused = fused / self._n_kernel_voxels()
        else:
            fused = fused / (native_shape[0] * native_shape[1] * native_shape[2])

        return torch.einsum('xyzio->oixyz', fused)

    def _kernel_interp(self, native_kernel, native_shape, K, alpha):
        """Trilinear upscaling mode: slice → upscale → accumulate."""
        fused = None
        # Also accumulate the per-voxel weight map for normalisation
        w_map = native_kernel.new_zeros(*native_shape)
        ones = native_kernel.new_ones(*native_shape)
        for level in range(K):
            sl = self._level_slices[level]
            if all(d == 1 for d in self._level_strides[level]):
                kernel_k = native_kernel
                w_k = ones
            else:
                kernel_k = native_kernel[sl[0], sl[1], sl[2]].contiguous()
                kernel_k = self._upscale_kernel_to_native(kernel_k, native_shape)
                ones_k = ones[sl[0], sl[1], sl[2]].contiguous()
                w_k = self._upscale_kernel_to_native(
                    ones_k[..., None, None], native_shape).squeeze(-1).squeeze(-1)
            w_map = w_map + alpha[level] * w_k
            if fused is None:
                fused = alpha[level] * kernel_k
            else:
                fused = fused + alpha[level] * kernel_k
        # Normalise so mean weight over sphere = 1
        r = self.diameter / 2
        sphere_mask = self.lattice.float().norm(dim=-1) <= r
        n_sphere = sphere_mask.sum()
        if n_sphere > 0:
            w_mean = (w_map * sphere_mask.float()).sum() / n_sphere
            fused = fused / (w_mean + 1e-8)
        return fused

    # ------------------------------------------------------------------
    def update_spacing(self, steps: tuple):
        """Update all lattice buffers for new spacing."""
        self._pyramid_steps = steps
        self._build_lattice_buffers(steps)


class NaivePyramidVoxelConvolution(nn.Module):
    """Reference multi-convolution pyramid (for correctness testing).

    At each of ``num_pyramid_levels`` scales:
      1. Downsample the input by avg-pooling (factor = sub_k).
      2. Convolve with a kernel whose lattice step is ``sub_k * step``
         (same physical diameter → fewer voxels).
      3. Upsample back to native resolution (nearest).
      4. Weight by alpha_k.

    The learned weight tensor and TP are shared across all levels.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        diameter: float,
        num_radial_basis: int,
        steps: tuple = (1.0, 1.0, 1.0),
        cutoff: bool = True,
        in1_var: list = None,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sequential_sc: bool = False,
        sc_mode: str = None,
        sphere_norm: bool = True,
        # ---- pyramid-specific ----
        num_pyramid_levels: int = 3,
        pyramid_decay: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.diameter = diameter
        self.num_radial_basis = num_radial_basis
        self.cutoff = cutoff
        self.kernel_trim_threshold = kernel_trim_threshold
        self.kernel_trim_cross_section = kernel_trim_cross_section
        self.sphere_norm = sphere_norm
        self.num_pyramid_levels = num_pyramid_levels

        if sc_mode is not None:
            self.sc_mode = sc_mode
        elif sequential_sc:
            self.sc_mode = "sc_first"
        else:
            self.sc_mode = "parallel"

        # Self-connection (same logic as VoxelConvolution)
        if self.sc_mode == "none":
            self.cutoff = False
            self.sc = None
        elif self.sc_mode in ("conv_first", "conv_first_res"):
            self.sc = Linear(self.irreps_out, self.irreps_out)
        else:
            self.sc = Linear(self.irreps_in, self.irreps_out)

        # Shared TP and weights
        sc_first_modes = ("sc_first", "sc_first_res")
        tp_in = self.irreps_out if self.sc_mode in sc_first_modes else self.irreps_in
        tp_kwargs = dict(shared_weights=False, compile_right=True)
        if in1_var is not None and self.sc_mode not in sc_first_modes:
            tp_kwargs['in1_var'] = in1_var
        self.tp = FullyConnectedTensorProduct(
            tp_in, self.irreps_sh, self.irreps_out, **tp_kwargs,
        )
        self.weight = nn.Parameter(
            torch.randn(self.num_radial_basis, self.tp.weight_numel)
        )

        # Per-level lattice buffers
        K = num_pyramid_levels
        self._subsample_factors = self._compute_subsample_factors(K, steps)
        self._build_all_level_buffers(steps, kwargs)

        # Level weights (always non-learnable)
        init_alpha = [pyramid_decay ** (K - 1 - k) for k in range(K)]
        logits = [math.log(math.exp(w) - 1) if w > 0.05 else w
                  for w in init_alpha]
        self.register_buffer('pyramid_logits', torch.tensor(logits))

        self.kwargs = kwargs

    def _compute_subsample_factors(self, K, steps):
        """Compute integer subsample factor per level.

        Level K-1 (finest) has factor 1.  Coarser levels have
        progressively larger integer factors chosen so the subsampled
        kernel still covers the diameter.
        """
        r = self.diameter / 2
        # Native half-extent on the coarsest axis
        max_step = max(steps)
        native_half = math.floor(r / max_step)
        if native_half < K:
            K = max(native_half, 1)
        # Evenly spaced subsample factors from K down to 1
        factors = []
        for k in range(K):
            # Level 0 → largest subsample, level K-1 → 1
            f = max(1, round((K - k)))
            factors.append(f)
        # Ensure monotonically decreasing and last is 1
        factors[-1] = 1
        for i in range(len(factors) - 2, -1, -1):
            factors[i] = max(factors[i], factors[i + 1] + 1)
        return factors

    def _build_all_level_buffers(self, steps, kwargs=None):
        """Build lattice/emb/sh buffers for each pyramid level."""
        r = self.diameter / 2
        for k, sub in enumerate(self._subsample_factors):
            eff_steps = tuple(s * sub for s in steps)
            axes = []
            for i in range(3):
                s = math.floor(r / eff_steps[i])
                s = _trim_half_extent(s, eff_steps[i], r,
                                      self.kernel_trim_cross_section,
                                      self.kernel_trim_threshold)
                axes.append(torch.arange(-s, s + 1.0) * eff_steps[i])
            lattice = torch.stack(
                torch.meshgrid(axes[0], axes[1], axes[2], indexing='ij'),
                dim=-1,
            )
            emb = soft_one_hot_linspace(
                x=lattice.float().norm(dim=-1),
                start=0.0, end=r,
                number=self.num_radial_basis,
                basis='smooth_finite', cutoff=self.cutoff,
            )
            sh = o3.spherical_harmonics(
                l=self.irreps_sh, x=lattice.float(),
                normalize=True, normalization='component',
            )
            self.register_buffer(f'lattice_{k}', lattice)
            self.register_buffer(f'emb_{k}', emb)
            self.register_buffer(f'sh_{k}', sh)

        # Padding from finest (last) level
        finest = getattr(self, f'lattice_{len(self._subsample_factors) - 1}')
        if kwargs is not None and 'padding' not in kwargs:
            kwargs['padding'] = tuple(s // 2 for s in finest.shape[:3])

    def _kernel_for_level(self, k):
        """Compute kernel for a single pyramid level."""
        emb = getattr(self, f'emb_{k}')
        sh = getattr(self, f'sh_{k}')
        lattice = getattr(self, f'lattice_{k}')
        weight = emb @ self.weight
        if self.sphere_norm:
            r = self.diameter / 2
            dists = lattice.float().norm(dim=-1)
            n_vox = (dists <= r).sum().item()
            weight = weight / max(n_vox, 1)
        else:
            n = lattice.shape[0] * lattice.shape[1] * lattice.shape[2]
            weight = weight / n
        kernel = self.tp.right(sh, weight)
        return torch.einsum('xyzio->oixyz', kernel)

    def _sc_weight_matrix(self):
        in_dim = self.sc.irreps_in.dim
        identity = torch.eye(in_dim, device=self.sc.weight.device,
                             dtype=self.sc.weight.dtype)
        return self.sc(identity).T

    def update_spacing(self, steps):
        self._subsample_factors = self._compute_subsample_factors(
            self.num_pyramid_levels, steps)
        self._build_all_level_buffers(steps)

    def forward(self, x):
        """Multi-convolution pyramid forward pass."""
        alpha = F.softplus(self.pyramid_logits)
        spatial = x.shape[2:]
        out = x.new_zeros(x.shape[0], self.irreps_out.dim, *spatial)

        for k, sub in enumerate(self._subsample_factors):
            kernel_k = self._kernel_for_level(k)
            if sub > 1:
                x_down = F.avg_pool3d(x, kernel_size=sub, stride=sub)
            else:
                x_down = x
            pad_k = tuple(s // 2 for s in kernel_k.shape[2:])
            y_k = F.conv3d(x_down, kernel_k, padding=pad_k)
            if sub > 1:
                y_k = F.interpolate(y_k, size=spatial, mode='nearest')
            out = out + alpha[k] * y_k

        # Self-connection (parallel mode only for simplicity)
        if self.sc is not None:
            sc_out = self.sc(x.transpose(1, 4)).transpose(1, 4)
            out = out + sc_out

        return out


class EquivariantPool3d(nn.Module):
    """Equivariant 3D pooling that preserves irrep structure.

    For scalar irreps (l=0), uses standard max pooling.
    For higher-order irreps (l>0), pools based on norm.
    """

    def __init__(self, scale: float, steps: tuple, mode: str, irreps):
        super().__init__()

        self.scale = scale
        self.steps = steps
        self.mode = mode
        self.kernel_size = tuple(_pool_factor(self.scale, s) for s in self.steps)
        self.irreps = irreps

    def update_spacing(self, steps: tuple):
        """Update kernel size for new spacing."""
        self.steps = steps
        self.kernel_size = tuple(_pool_factor(self.scale, s) for s in self.steps)

    def forward(self, x):
        if self.mode == 'maxpool3d':
            return self._equivariant_max_pool(x)
        elif self.mode == 'average':
            return F.avg_pool3d(x, self.kernel_size, stride=self.kernel_size)
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")

    def _equivariant_max_pool(self, x):
        """Vectorized equivariant max pooling.

        Processes all instances of each irrep type in a single batched op
        instead of looping per-instance.
        """
        assert x.shape[1] == self.irreps.dim, "Shape mismatch"
        B = x.shape[0]
        spatial = x.shape[2:]

        results = []
        start = 0

        for mul, ir in self.irreps:
            dim = ir.dim
            end = start + mul * dim

            if ir.l == 0:
                # All scalars at once: (B, mul, D, H, W) — standard max pool
                pooled = F.max_pool3d(
                    x[:, start:end, ...],
                    self.kernel_size, stride=self.kernel_size,
                )
                results.append(pooled)
            else:
                # Reshape to (B, mul, dim, D, H, W)
                feat = x[:, start:end, ...].reshape(B, mul, dim, *spatial)

                # Pool on norms: (B, mul, D, H, W) → (B, mul, D', H', W')
                norms = feat.norm(dim=2)
                _, indices = F.max_pool3d_with_indices(
                    norms, self.kernel_size, stride=self.kernel_size,
                    return_indices=True,
                )

                # Gather all components using spatial indices
                out_spatial = indices.shape[2:]
                flat_feat = feat.reshape(B, mul, dim, -1)
                flat_idx = indices.reshape(B, mul, 1, -1).expand(-1, -1, dim, -1)
                gathered = torch.gather(flat_feat, 3, flat_idx)
                gathered = gathered.reshape(B, mul * dim, *out_spatial)
                results.append(gathered)

            start = end

        return torch.cat(results, dim=1)


def compute_s2d_output_irreps(irreps_in, lmax_expand=1, max_output_l=2,
                              parity=-1):
    """Compute output irreps of MultipolePool without building the full layer.

    For O3 (parity=-1) with input a×0e + b×1o + c×2e, lmax_expand=1:
        0e: a+b, 1o: a+b+c, 2e: b+c

    For SO3 (parity=1) with input a×0e + b×1e + c×2e, lmax_expand=1:
        0e: a+b, 1e: a+2b+c, 2e: b+2c
    (more terms retained because no parity filtering)

    Parameters
    ----------
    irreps_in : Irreps
        Input irreps.
    lmax_expand : int
        Max l for SH expansion (typically 1).
    max_output_l : int
        Maximum l to keep in output.
    parity : int
        Parity convention for SH: 1 for SO3 (all-even), -1 for O3 (natural).

    Returns
    -------
    Irreps
        Filtered output irreps.
    """
    irreps_in = o3.Irreps(irreps_in)
    irreps_sh = o3.Irreps.spherical_harmonics(lmax_expand, p=parity)

    # Use FullTensorProduct to get exact CG output irreps
    tp = o3.FullTensorProduct(irreps_in, irreps_sh)
    full_out = tp.irreps_out

    # Filter by max_output_l.
    # For O3 (parity=-1): also filter to natural parity (p = (-1)^l).
    # For SO3 (parity=1): all outputs are even, no parity filtering needed.
    kept = []
    for mul, ir in full_out:
        if ir.l > max_output_l:
            continue
        if parity == -1 and ir.p != (-1) ** ir.l:
            continue
        kept.append((mul, ir))

    return o3.Irreps(kept).simplify() if kept else o3.Irreps("0x0e")


def compute_fixed_point_irreps(dim_target, parity=-1):
    """Compute irreps at the CG fixed-point ratio for MultipolePool.

    Iterates the CG expansion to find the converged multiplicity ratio,
    then scales to fit dim_target.

    For O3 (parity=-1): fixed-point ratio is 1:sqrt(2):1 → a×0e + b×1o + a×2e
    For SO3 (parity=1): fixed-point ratio is ~1:2.25:1.80 → a×0e + b×1e + c×2e

    Parameters
    ----------
    dim_target : int
        Target total dimension.
    parity : int
        Parity convention: 1 for SO3 (all-even), -1 for O3 (natural).

    Returns
    -------
    Irreps
        Fixed-point irreps.
    """
    p_label = 'e' if parity == 1 else 'o'

    # Find fixed-point ratio by iterating CG expansion
    irreps = o3.Irreps(f"1x0e + 1x1{p_label} + 1x2e")
    for _ in range(15):  # converges in ~10 iterations
        irreps = compute_s2d_output_irreps(irreps, parity=parity)

    # Extract converged multiplicities
    muls = {}
    for mul, ir in irreps:
        muls[str(ir)] = mul
    m0 = muls.get('0e', 1)
    m1 = muls.get(f'1{p_label}', 1)
    m2 = muls.get('2e', 1)

    # Scale to target dimension
    dim_per_unit = m0 * 1 + m1 * 3 + m2 * 5
    scale = dim_target / dim_per_unit
    a = max(1, round(m0 * scale))
    b = max(1, round(m1 * scale))
    c = max(1, round(m2 * scale))
    return o3.Irreps(f"{a}x0e + {b}x1{p_label} + {c}x2e")


class MultipolePool(nn.Module):
    """Equivariant space-to-depth via spherical harmonic multipole expansion.

    For each block of voxels with features f_i at sub-voxel positions r_i:
        output = sum_i TP(f_i, Y(r_i))

    where Y are spherical harmonics and TP is the parameter-free full tensor product.
    Output irreps are filtered to l <= max_output_l. For O3 (parity=-1),
    unnatural parity terms are also removed.

    Spacing-adaptive: the pool factor per axis is computed as floor(scale / step),
    matching the mm-scale approach used by EquivariantPool3d. When spacing changes,
    both the factors and the projection matrix are recomputed.

    Parameters
    ----------
    irreps_in : Irreps
        Input irreps.
    scale : float
        Pooling scale in physical units (mm). Per-axis factor = floor(scale / step).
    lmax_expand : int
        Max l for SH expansion (default 1).
    steps : tuple
        Voxel spacing in physical units.
    max_output_l : int
        Maximum l to keep in output (default 2).
    parity : int
        Parity convention for SH: 1 for SO3 (all-even), -1 for O3 (natural).
    """

    def __init__(self, irreps_in, scale=2.0, lmax_expand=1, steps=(1., 1., 1.),
                 max_output_l=2, parity=-1):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.scale = scale
        self.lmax_expand = lmax_expand
        self.max_output_l = max_output_l
        self.parity = parity

        # Compute filtered output irreps
        self._irreps_out = compute_s2d_output_irreps(
            self.irreps_in, lmax_expand, max_output_l, parity=parity)

        # Cache projection matrices by steps tuple to avoid expensive recomputation
        self._proj_cache = {}

        # Build projection matrix for initial spacing (also sets self.factors)
        self._build_projection(steps)

    @property
    def irreps_out(self):
        return self._irreps_out

    def _build_projection(self, steps):
        """Build or rebuild the projection matrix for given spacing.

        Results are cached by factors tuple to avoid expensive recomputation
        of CG tensor products and SH basis during multi-resolution training.
        Factors (not raw steps) determine projection matrix structure; tiny SH
        variations from step float imprecision are negligible.
        """
        # Compute factors using robust near-integer rounding
        self.factors = tuple(_pool_factor(self.scale, s) for s in steps)

        # Cache by factors — structure depends on factors, SH variation is negligible
        cache_key = self.factors

        if cache_key in self._proj_cache:
            M = self._proj_cache[cache_key]
            if hasattr(self, 'proj') and self.proj is not None:
                self.proj = M.to(device=self.proj.device)
            else:
                self.register_buffer('proj', M)
            return
        n_pos = self.factors[0] * self.factors[1] * self.factors[2]

        # Sub-voxel displacement vectors (physical units)
        positions = []
        for idx in itertools.product(range(self.factors[0]),
                                     range(self.factors[1]),
                                     range(self.factors[2])):
            offset = torch.tensor([
                (idx[0] - (self.factors[0] - 1) / 2.0) * steps[0],
                (idx[1] - (self.factors[1] - 1) / 2.0) * steps[1],
                (idx[2] - (self.factors[2] - 1) / 2.0) * steps[2],
            ], dtype=torch.float64)
            positions.append(offset)
        positions = torch.stack(positions)  # (n_pos, 3)

        # SH at each sub-voxel position (parity convention matches network)
        irreps_sh = o3.Irreps.spherical_harmonics(self.lmax_expand, p=self.parity)
        sh_values = o3.spherical_harmonics(irreps_sh, positions, normalize=True)

        # Build full TP output irreps and identify which to keep
        tp = o3.FullTensorProduct(self.irreps_in, irreps_sh)
        full_out = tp.irreps_out

        # Identify indices of kept irreps in the full output
        # SO3 (parity=1): all outputs are even, just filter by max_output_l
        # O3 (parity=-1): also filter to natural parity (p = (-1)^l)
        kept_slices = []
        idx = 0
        for mul, ir in full_out:
            dim = mul * ir.dim
            keep = (ir.l <= self.max_output_l)
            if self.parity == -1 and ir.p != (-1) ** ir.l:
                keep = False
            if keep:
                kept_slices.append((idx, idx + dim))
            idx += dim

        # Build projection matrix in float64 for precision
        d_in = self.irreps_in.dim
        tp64 = o3.FullTensorProduct(self.irreps_in, irreps_sh).double()
        sh64 = sh_values.double()
        d_full = tp64.irreps_out.dim
        d_out = self._irreps_out.dim

        eye = torch.eye(d_in, dtype=torch.float64)
        M_full = torch.zeros(d_full, n_pos * d_in, dtype=torch.float64)
        for i in range(n_pos):
            sh_i = sh64[i].unsqueeze(0).expand(d_in, -1)
            M_i = tp64(eye, sh_i)  # (d_in, d_full)
            M_full[:, i * d_in:(i + 1) * d_in] = M_i.t()

        # Slice rows to keep only filtered output irreps
        kept_rows = []
        for start, end in kept_slices:
            kept_rows.append(M_full[start:end])
        M = torch.cat(kept_rows, dim=0)
        assert M.shape[0] == d_out, f"Projection matrix row mismatch: {M.shape[0]} vs {d_out}"

        # Cache the result (CPU tensor, moved to device on demand)
        self._proj_cache[cache_key] = M

        # Register or update buffer
        if hasattr(self, 'proj') and self.proj is not None:
            self.proj = M.to(device=self.proj.device)
        else:
            self.register_buffer('proj', M)

    def update_spacing(self, steps):
        """Recompute projection matrix for new spacing."""
        self._build_projection(steps)

    def forward(self, x):
        B, C, D, H, W = x.shape
        f0, f1, f2 = self.factors
        assert D % f0 == 0 and H % f1 == 0 and W % f2 == 0, \
            f"Spatial dims ({D},{H},{W}) must be divisible by factors={self.factors}"
        D2, H2, W2 = D // f0, H // f1, W // f2

        # Reshape into f0×f1×f2 blocks
        x = x.reshape(B, C, D2, f0, H2, f1, W2, f2)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # (B, D', H', W', f0, f1, f2, C)
        x = x.reshape(-1, f0 * f1 * f2 * C)  # (N, n_pos*C)

        # Project and reshape
        proj = self.proj.to(x.dtype)
        out = x @ proj.t()  # (N, d_out)
        return out.reshape(B, D2, H2, W2, -1).permute(0, 4, 1, 2, 3)

    def extra_repr(self):
        return (f"irreps_in={self.irreps_in}, irreps_out={self._irreps_out}, "
                f"scale={self.scale}, factors={self.factors}, lmax={self.lmax_expand}")


class Identity(nn.Module):
    """Identity module for optional normalization."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ConvolutionBlock(nn.Module):
    """Double convolution block with gating and normalization."""

    def __init__(
        self,
        irreps_in,
        irreps_hidden,
        activation,
        irreps_sh,
        normalization: str,
        diameter: float,
        num_radial_basis: int,
        steps: tuple,
        dropout_prob: float,
        cutoff: bool,
        pre_norm: bool = False,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sequential_sc: bool = False,
        sc_mode: str = None,
        fused_gate: bool = True,
        sphere_norm: bool = True,
        backend: str = "e3nn",
        pyramid=False,
    ):
        super().__init__()

        # Resolve sc_mode from either new param or legacy bool
        if sc_mode is None:
            sc_mode = "sc_first" if sequential_sc else "parallel"

        # Setup normalization
        if normalization == 'None':
            BN = Identity
        elif normalization == 'batch':
            BN = BatchNorm
        elif normalization == 'instance':
            BN = partial(BatchNorm, instance=True)
        elif normalization == 'layer':
            BN = EquivariantLayerNorm
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        # When NormSoftClamp is active, l>0 inputs have var=init_target_var (not 1.0).
        # Tell the TP so its weight normalization accounts for this.
        softclamp_target_var = 2.7  # NormSoftClamp default init_target_var
        def _in1_var_for(irreps):
            """Compute per-group in1_var: 1.0 for l=0, target_var for l>0."""
            return [softclamp_target_var if ir.l > 0 else 1.0 for _, ir in Irreps(irreps)]

        if fused_gate:
            # FusedGate: conv outputs irreps_hidden directly, no extra gate channels
            scalar_act = activation[0]
            self.gate1 = FusedGate(irreps_hidden, scalar_act)
            conv1_out_irreps = irreps_hidden
        else:
            # Legacy e3nn Gate: conv must output extra l=0 channels for gating
            irreps_scalars = Irreps([(mul, ir) for mul, ir in Irreps(irreps_hidden) if ir.l == 0])
            irreps_gated = Irreps([(mul, ir) for mul, ir in Irreps(irreps_hidden) if ir.l > 0])
            irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")
            self.gate1 = Gate(irreps_scalars, activation, irreps_gates, [torch.sigmoid], irreps_gated)
            conv1_out_irreps = self.gate1.irreps_in

        in1_var_conv1 = _in1_var_for(irreps_in) if pre_norm else None
        self.conv1 = get_voxel_convolution(
            backend=backend,
            pyramid=pyramid,
            irreps_in=irreps_in, irreps_out=conv1_out_irreps, irreps_sh=irreps_sh,
            diameter=diameter, num_radial_basis=num_radial_basis, steps=steps,
            cutoff=cutoff, in1_var=in1_var_conv1,
            kernel_trim_threshold=kernel_trim_threshold,
            kernel_trim_cross_section=kernel_trim_cross_section,
            sc_mode=sc_mode,
            sphere_norm=sphere_norm,
        )
        self.batchnorm1 = BN(conv1_out_irreps)
        self.dropout1 = Dropout(irreps_hidden, dropout_prob)

        # Post-gate NormSoftClamp (self-normalizing activation for l>0)
        self.norm_selu1 = NormSoftClamp(irreps_hidden) if pre_norm else None

        if fused_gate:
            self.gate2 = FusedGate(irreps_hidden, scalar_act)
            conv2_out_irreps = irreps_hidden
        else:
            self.gate2 = Gate(irreps_scalars, activation, irreps_gates, [torch.sigmoid], irreps_gated)
            conv2_out_irreps = self.gate2.irreps_in

        in1_var_conv2 = _in1_var_for(irreps_hidden) if pre_norm else None
        self.conv2 = get_voxel_convolution(
            backend=backend,
            pyramid=pyramid,
            irreps_in=irreps_hidden, irreps_out=conv2_out_irreps, irreps_sh=irreps_sh,
            diameter=diameter, num_radial_basis=num_radial_basis, steps=steps,
            cutoff=cutoff, in1_var=in1_var_conv2,
            kernel_trim_threshold=kernel_trim_threshold,
            kernel_trim_cross_section=kernel_trim_cross_section,
            sc_mode=sc_mode,
            sphere_norm=sphere_norm,
        )
        self.batchnorm2 = BN(conv2_out_irreps)
        self.dropout2 = Dropout(irreps_hidden, dropout_prob)

        # Post-gate NormSoftClamp for second gate
        self.norm_selu2 = NormSoftClamp(irreps_hidden) if pre_norm else None

        self.irreps_out = irreps_hidden

    def update_spacing(self, steps: tuple):
        """Update convolution lattices for new spacing."""
        self.conv1.update_spacing(steps)
        self.conv2.update_spacing(steps)

    def forward(self, x):
        dtype = x.dtype
        # First conv block
        x = self.conv1(x)
        x = self.batchnorm1(x.transpose(1, 4)).to(dtype).transpose(1, 4)
        x = self.gate1(x.transpose(1, 4)).to(dtype).transpose(1, 4)
        if self.norm_selu1 is not None:
            x = self.norm_selu1(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout1(x.transpose(1, 4)).transpose(1, 4)

        # Second conv block
        x = self.conv2(x)
        x = self.batchnorm2(x.transpose(1, 4)).to(dtype).transpose(1, 4)
        x = self.gate2(x.transpose(1, 4)).to(dtype).transpose(1, 4)
        if self.norm_selu2 is not None:
            x = self.norm_selu2(x.transpose(1, 4)).transpose(1, 4)
        x = self.dropout2(x.transpose(1, 4)).transpose(1, 4)

        return x


def _parse_ratios(ratios):
    """Parse ratios into (even_ratios, odd_ratios) tuples.

    Accepts:
      - tuple: (4, 2, 1) - same ratios for even/odd
      - dict: {'e': (4, 2, 1), 'o': (2, 1, 0)} - separate ratios
    """
    if isinstance(ratios, dict):
        return tuple(ratios.get('e', ())), tuple(ratios.get('o', ()))
    return tuple(ratios), tuple(ratios)


def _build_irreps(ne: int, no: int, ratios, fill_to: int = 0) -> Irreps:
    """Build irreps string from ne, no and ratios for each l.

    If fill_to > 0, add extra scalar irreps to reach exactly fill_to features.
    """
    even_ratios, odd_ratios = _parse_ratios(ratios)
    parts = []
    for l, r in enumerate(even_ratios):
        if r * ne > 0:
            parts.append(f"{r * ne}x{l}e")
    for l, r in enumerate(odd_ratios):
        if r * no > 0:
            parts.append(f"{r * no}x{l}o")

    irreps = Irreps(" + ".join(parts)).simplify() if parts else Irreps("0x0e")

    # Top up with scalars if fill_to is specified
    if fill_to > 0 and irreps.dim < fill_to:
        extra_scalars = fill_to - irreps.dim
        irreps = (irreps + Irreps(f"{extra_scalars}x0e")).sort().irreps.simplify()

    return irreps


def _features_per_ne(ratios, has_odd: bool) -> int:
    """Compute features per ne unit from ratios. Each l has (2l+1) components."""
    even_ratios, odd_ratios = _parse_ratios(ratios)
    total = sum(r * (2 * l + 1) for l, r in enumerate(even_ratios))
    if has_odd:
        total += sum(r * (2 * l + 1) for l, r in enumerate(odd_ratios))
    return total


class Encoder(nn.Module):
    """UNet encoder (downsampling path)."""

    def __init__(
        self,
        n_downsample: int,
        activation,
        irreps_sh,
        ne: int,
        no: int,
        normalization: str,
        irreps_in,
        diameters: list,
        num_radial_basis: int,
        steps_array: list,
        down_op: str,
        scales: list,
        dropout_prob: float,
        cutoff: bool,
        max_features: int = 320,
        irrep_ratios: tuple = (4, 2, 1),
        fill_to_max: bool = False,
        pre_norm: bool = False,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sequential_sc: bool = False,
        sc_mode: str = None,
        fused_gate: bool = True,
        sphere_norm: bool = True,
        backend: str = "e3nn",
        pyramid=False,
    ):
        super().__init__()

        # Resolve sc_mode
        if sc_mode is None:
            sc_mode = "sc_first" if sequential_sc else "parallel"

        features_per_ne = _features_per_ne(irrep_ratios, no > 0)
        ne_max = max_features // features_per_ne

        self.down_op = down_op
        blocks = []
        self.down_irreps_out = []
        self.ne_per_level = []
        self.no_per_level = []

        # Infer parity convention from SH irreps:
        # SO3 uses all-even (parity=1), O3 uses natural parity (parity=-1)
        sh_parity = 1 if all(ir.p == 1 for _, ir in o3.Irreps(irreps_sh)) else -1

        if down_op == 's2d':
            # S2D path: same ne-doubling channel schedule as maxpool, but
            # MultipolePool + Linear replaces max-pooling between levels.
            # MultipolePool expands features via CG (captures spatial structure),
            # then a Linear projection compresses back to the target irreps.
            #
            # Precompute all level irreps so we know block input dims.
            # For level n>0, the projection from the previous pool outputs
            # level_irreps[n], so irreps_in = irreps_out for that block.
            ne_tmp, no_tmp = ne, no
            level_irreps = []
            for n in range(n_downsample + 1):
                ne_c = min(ne_tmp, ne_max)
                no_c = min(no_tmp, ne_max) if no_tmp > 0 else 0
                ft = max_features if (fill_to_max and ne_tmp >= ne_max) else 0
                level_irreps.append(_build_irreps(ne_c, no_c, irrep_ratios, ft))
                ne_tmp *= 2
                no_tmp *= 2

            for n in range(n_downsample + 1):
                ne_capped = min(ne, ne_max)
                no_capped = min(no, ne_max) if no > 0 else 0

                self.ne_per_level.append(ne_capped)
                self.no_per_level.append(no_capped)

                irreps_hidden = level_irreps[n]
                # For n>0, pool+proj from previous level outputs level_irreps[n]
                block_irreps_in = irreps_in if n == 0 else level_irreps[n]

                block = ConvolutionBlock(
                    block_irreps_in, irreps_hidden, activation, irreps_sh, normalization,
                    diameters[n], num_radial_basis, steps_array[n], dropout_prob, cutoff,
                    pre_norm=pre_norm,
                    kernel_trim_threshold=kernel_trim_threshold,
                    kernel_trim_cross_section=kernel_trim_cross_section,
                    sc_mode=sc_mode,
                    fused_gate=fused_gate,
                    sphere_norm=sphere_norm,
                    backend=backend,
                    pyramid=pyramid,
                )
                blocks.append(block)
                self.down_irreps_out.append(block.irreps_out)
                ne *= 2
                no *= 2
        else:
            for n in range(n_downsample + 1):
                ne_capped = min(ne, ne_max)
                no_capped = min(no, ne_max) if no > 0 else 0

                self.ne_per_level.append(ne_capped)
                self.no_per_level.append(no_capped)

                # Fill to max_features with scalars if at the cap
                fill_to = max_features if (fill_to_max and ne >= ne_max) else 0
                irreps_hidden = _build_irreps(ne_capped, no_capped, irrep_ratios, fill_to)

                block = ConvolutionBlock(
                    irreps_in, irreps_hidden, activation, irreps_sh, normalization,
                    diameters[n], num_radial_basis, steps_array[n], dropout_prob, cutoff,
                    pre_norm=pre_norm,
                    kernel_trim_threshold=kernel_trim_threshold,
                    kernel_trim_cross_section=kernel_trim_cross_section,
                    sc_mode=sc_mode,
                    fused_gate=fused_gate,
                    sphere_norm=sphere_norm,
                    backend=backend,
                    pyramid=pyramid,
                )
                blocks.append(block)
                self.down_irreps_out.append(block.irreps_out)
                irreps_in = block.irreps_out
                ne *= 2
                no *= 2

        self.down_blocks = nn.ModuleList(blocks)

        # Pooling / s2d layers
        pooling = []
        projections = []
        for n in range(n_downsample):
            if down_op == 's2d':
                pool = MultipolePool(
                    self.down_irreps_out[n], scale=scales[n],
                    steps=steps_array[n], parity=sh_parity)
                pooling.append(pool)
                # Linear projection: compress CG expansion back to next level's irreps
                proj = Linear(pool.irreps_out, self.down_irreps_out[n + 1])
                projections.append(proj)
            else:
                pooling.append(EquivariantPool3d(
                    scales[n], steps_array[n], down_op, self.down_irreps_out[n]
                ))

        self.down_pool = nn.ModuleList(pooling)
        self.down_proj = nn.ModuleList(projections) if projections else None

    def update_spacing(self, steps_array: list):
        """Update all blocks and pooling layers for new spacing."""
        for i, block in enumerate(self.down_blocks):
            block.update_spacing(steps_array[i])
        for i, pool in enumerate(self.down_pool):
            pool.update_spacing(steps_array[i])

    def forward(self, x):
        """Returns list of features at each level."""
        features = []

        for i, block in enumerate(self.down_blocks):
            x = block(x)
            features.append(x)
            if i < len(self.down_blocks) - 1:
                x = self.down_pool[i](x)
                if self.down_proj is not None:
                    # e3nn Linear expects channel-last; data is channel-first
                    x = self.down_proj[i](x.transpose(1, 4)).transpose(1, 4)

        return features


class Decoder(nn.Module):
    """UNet decoder (upsampling path) with optional deep supervision."""

    def __init__(
        self,
        n_blocks: int,
        activation,
        irreps_sh,
        ne_per_level: list,
        no_per_level: list,
        normalization: str,
        encoder_irreps: list,
        diameters: list,
        num_radial_basis: int,
        steps_array: list,
        scales: list,
        dropout_prob: float,
        scalar_upsampling: bool,
        cutoff: bool,
        deep_supervision: bool = False,
        n_classes: int = None,
        max_features: int = 320,
        irrep_ratios: tuple = (4, 2, 1),
        fill_to_max: bool = False,
        pre_norm: bool = False,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sequential_sc: bool = False,
        sc_mode: str = None,
        fused_gate: bool = True,
        irreps_per_level: list = None,
        pool_mode: str = 'maxpool3d',
        sphere_norm: bool = True,
        backend: str = "e3nn",
        pyramid=False,
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.deep_supervision = deep_supervision
        self.scales = scales  # Store for update_spacing
        self.pool_mode = pool_mode

        # Resolve sc_mode
        if sc_mode is None:
            sc_mode = "sc_first" if sequential_sc else "parallel"

        has_odd = no_per_level is not None and any(n > 0 for n in no_per_level)
        features_per_ne = _features_per_ne(irrep_ratios, has_odd)
        ne_max = max_features // features_per_ne

        # For scalar upsampling, use only l=0 with summed multipliers
        even_ratios, odd_ratios = _parse_ratios(irrep_ratios)
        scalar_ratios = {'e': (sum(even_ratios) * 2,), 'o': (sum(odd_ratios) * 2,)}

        irreps_in = encoder_irreps[-1]
        blocks = []
        upsample_ops = []
        block_irreps_out = []

        for n in range(n_blocks):
            if irreps_per_level is not None:
                # Direct irreps specification (used with s2d encoder)
                irreps_hidden = o3.Irreps(irreps_per_level[n])
            else:
                ne = ne_per_level[n]
                no = no_per_level[n] if no_per_level is not None else 0
                ne_capped = min(ne, ne_max)
                no_capped = min(no, ne_max) if no > 0 else 0

                # Fill to max_features with scalars if at the cap
                fill_to = max_features if (fill_to_max and ne >= ne_max) else 0

                if scalar_upsampling:
                    irreps_hidden = _build_irreps(ne_capped, no_capped, scalar_ratios, fill_to)
                else:
                    irreps_hidden = _build_irreps(ne_capped, no_capped, irrep_ratios, fill_to)

            # Concatenate with skip connection
            skip_irreps = encoder_irreps[::-1][n + 1]

            block = ConvolutionBlock(
                irreps_in + skip_irreps, irreps_hidden, activation, irreps_sh,
                normalization, diameters[n], num_radial_basis, steps_array[n],
                dropout_prob, cutoff, pre_norm=pre_norm,
                kernel_trim_threshold=kernel_trim_threshold,
                kernel_trim_cross_section=kernel_trim_cross_section,
                sc_mode=sc_mode,
                fused_gate=fused_gate,
                sphere_norm=sphere_norm,
                backend=backend,
                pyramid=pyramid,
            )
            blocks.append(block)
            block_irreps_out.append(block.irreps_out)
            irreps_in = block.irreps_out

            # Upsample scale factor (same formula for maxpool and s2d)
            scale_factor = tuple(
                _pool_factor(scales[n], step) for step in steps_array[n]
            )
            upsample_ops.append(nn.Upsample(
                scale_factor=scale_factor, mode='trilinear', align_corners=True
            ))

        self.up_blocks = nn.ModuleList(blocks)
        self.upsample_ops = nn.ModuleList(upsample_ops)

        # Deep supervision output heads (one for each decoder level except the last)
        if deep_supervision and n_classes is not None:
            from e3nn.o3 import Linear
            output_irreps = Irreps(f"{n_classes}x0e")
            self.ds_heads = nn.ModuleList([
                Linear(block_irreps_out[i], output_irreps)
                for i in range(n_blocks - 1)  # All but last (last uses main output head)
            ])
        else:
            self.ds_heads = None

    def update_spacing(self, steps_array: list):
        """Update all blocks and upsample layers for new spacing."""
        for i, block in enumerate(self.up_blocks):
            block.update_spacing(steps_array[i])
        # Update upsample scale factors
        for i in range(self.n_blocks):
            scale_factor = tuple(
                _pool_factor(self.scales[i], step) for step in steps_array[i]
            )
            self.upsample_ops[i].scale_factor = scale_factor

    def forward(self, x, encoder_features):
        """Forward pass with skip connections from encoder.

        Returns
        -------
        If deep_supervision is False:
            x : tensor of shape (B, C, D, H, W)
        If deep_supervision is True:
            list of tensors at different scales, from coarsest to finest
        """
        ds_outputs = []

        for i in range(self.n_blocks):
            x = self.upsample_ops[i](x)
            skip = encoder_features[::-1][i + 1]
            if x.shape[2:] != skip.shape[2:]:
                raise RuntimeError(
                    f"Decoder level {i}: upsampled shape {x.shape[2:]} != "
                    f"encoder skip shape {skip.shape[2:]}. "
                    f"upsample_factor={self.upsample_ops[i].scale_factor}, "
                    f"scales={self.scales}, "
                    f"all encoder shapes={[f.shape for f in encoder_features]}"
                )
            x = torch.cat([x, skip], dim=1)
            x = self.up_blocks[i](x)

            # Collect deep supervision outputs (all but last level)
            if self.deep_supervision and self.ds_heads is not None and i < self.n_blocks - 1:
                ds_out = self.ds_heads[i](x.transpose(1, 4)).to(x.dtype).transpose(1, 4)
                ds_outputs.append(ds_out)

        if self.deep_supervision and self.ds_heads is not None:
            # Return list: [coarsest, ..., finest (main output to be added by caller)]
            return x, ds_outputs
        return x

    def forward_superres(self, x, encoder_features):
        """Forward pass for super-resolution: interpolate skip connections when
        spatial dimensions don't match between encoder features and decoder.

        No deep supervision for super-res batches (simplifies implementation).

        Parameters
        ----------
        x : torch.Tensor
            Bottleneck features from encoder
        encoder_features : list of torch.Tensor
            Encoder features at each level (from finest to coarsest)

        Returns
        -------
        torch.Tensor
            Decoded features at full resolution
        """
        for i in range(self.n_blocks):
            x = self.upsample_ops[i](x)
            skip = encoder_features[::-1][i + 1]
            # Interpolate skip if spatial dims don't match decoder
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(
                    skip, size=x.shape[2:],
                    mode='trilinear', align_corners=True
                )
            x = torch.cat([x, skip], dim=1)
            x = self.up_blocks[i](x)
        return x
