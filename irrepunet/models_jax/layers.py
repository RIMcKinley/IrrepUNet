"""Equivariant layers for JAX E3nnUNet.

Port of models/layers.py using Flax NNX and cuequivariance_jax.
"""

import math
import functools
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

import cuequivariance as cue
import cuequivariance_jax as cuex

from irrepunet.models_jax.radial_basis import soft_one_hot_linspace
from irrepunet.models_jax.tp_right import build_tp_right


def _pool_factor(scale, step):
    """Compute integer pooling factor for one axis, robust to float imprecision.

    Uses round() when scale/step is within 1% of an integer to prevent
    math.floor(1.99999984) = 1 when the true ratio is 2.0.

    Matches PyTorch implementation in models/layers.py.
    """
    if step >= scale:
        return 1
    raw = scale / step
    rounded = round(raw)
    if rounded >= 1 and abs(raw - rounded) < 0.01 * rounded:
        return rounded
    return max(math.floor(raw), 1)


# ---------------------------------------------------------------------------
# Pure NumPy lattice buffer computation (bypass JAX/cuequivariance dispatch)
# ---------------------------------------------------------------------------

def _soft_unit_step_numpy(x):
    """C-infinity smooth unit step in NumPy."""
    safe_x = np.where(x > 0, x, 1.0)
    return np.where(x > 0, np.exp(-1.0 / safe_x), 0.0)


def _soft_one_hot_linspace_numpy(x, start, end, number, cutoff=True):
    """Port of soft_one_hot_linspace using NumPy ops."""
    if cutoff:
        values = np.linspace(start, end, number + 2)
        step = values[1] - values[0]
        values = values[1:-1]
    else:
        values = np.linspace(start, end, number)
        step = values[1] - values[0]
    diff = (x[..., None] - values) / step
    return 1.14136 * np.e**2 * _soft_unit_step_numpy(diff + 1) * _soft_unit_step_numpy(1 - diff)


def _spherical_harmonics_numpy(ls, vecs):
    """Compute real SH in component normalization using pure NumPy polynomials.

    Matches ``cuequivariance_jax.spherical_harmonics(ls, vecs, normalize=True)``
    for l <= 2.  Raises ValueError for l > 2.

    Parameters
    ----------
    ls : list of int
        Angular momentum values (unique, sorted).
    vecs : np.ndarray, shape (N, 3)
        Input 3D vectors (not necessarily unit-length; normalized internally).
    """
    N = vecs.shape[0]
    x = vecs[:, 0]
    y = vecs[:, 1]
    z = vecs[:, 2]

    r = np.sqrt(x * x + y * y + z * z)
    safe_r = np.where(r > 0, r, 1.0)
    nx = x / safe_r
    ny = y / safe_r
    nz = z / safe_r
    # Zero out at origin (cuequivariance returns 0 for l>=1 at origin)
    mask = (r > 0).astype(vecs.dtype)

    sh_parts = []
    for l in ls:
        if l == 0:
            sh_parts.append(np.ones((N, 1), dtype=vecs.dtype))
        elif l == 1:
            sqrt3 = np.float64(np.sqrt(3.0)).astype(vecs.dtype)
            sh_parts.append(np.stack([sqrt3 * nx, sqrt3 * ny, sqrt3 * nz], axis=-1))
        elif l == 2:
            sqrt15 = np.sqrt(np.float64(15.0))
            sqrt5_h = np.sqrt(np.float64(5.0)) / 2.0
            sqrt15_h = sqrt15 / 2.0
            # cuequivariance component ordering (y-polar axis):
            # [xz, xy, 3y²-1, yz, z²-x²]
            block = np.stack([
                sqrt15 * nx * nz,
                sqrt15 * nx * ny,
                sqrt5_h * (3 * ny * ny - 1),
                sqrt15 * ny * nz,
                sqrt15_h * (nz * nz - nx * nx),
            ], axis=-1).astype(vecs.dtype)
            # Zero all l=2 components at origin (m=0 has constant term)
            sh_parts.append(block * mask[:, None])
        else:
            raise ValueError(
                f"_spherical_harmonics_numpy only supports l <= 2, got l={l}"
            )

    return np.concatenate(sh_parts, axis=-1)


def _trim_half_extent(s, step, r, cross_section_threshold, old_threshold):
    """Trim kernel half-extent based on cross-section fraction or legacy threshold.

    Mirrors the PyTorch implementation in models/layers.py.
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


def _compute_lattice_buffers_numpy(diameter, steps, ls, num_radial_basis, cutoff,
                                   kernel_trim_cross_section=0.0,
                                   kernel_trim_threshold=1.0):
    """Compute lattice, radial basis, and SH entirely in NumPy.

    Returns (lattice, emb, sh, padding) as NumPy arrays.
    """
    r = diameter / 2
    axes = []
    for s in steps:
        half = math.floor(r / s)
        half = _trim_half_extent(half, s, r,
                                 kernel_trim_cross_section,
                                 kernel_trim_threshold)
        axes.append(np.arange(-half, half + 1.0) * s)
    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    lattice = np.stack([gx, gy, gz], axis=-1).astype(np.float32)
    padding = tuple(s // 2 for s in lattice.shape[:3])

    norms = np.sqrt(np.sum(lattice ** 2, axis=-1))
    emb = _soft_one_hot_linspace_numpy(norms, 0.0, r, num_radial_basis, cutoff)

    lattice_flat = lattice.reshape(-1, 3)
    sh_flat = _spherical_harmonics_numpy(ls, lattice_flat)
    sh = sh_flat.reshape(*lattice.shape[:3], -1)

    return lattice, emb, sh, padding


# ---------------------------------------------------------------------------
# Irreps utilities (mirrors e3nn Irreps via cuequivariance)
# ---------------------------------------------------------------------------

def _parse_irreps(irreps) -> cue.Irreps:
    """Accept string or cue.Irreps, return cue.Irreps for O3."""
    if isinstance(irreps, cue.Irreps):
        return irreps
    return cue.Irreps("O3", str(irreps))


def _irreps_dim(irreps: cue.Irreps) -> int:
    """Total dimension of an irreps."""
    return sum(mul * ir.dim for mul, ir in irreps)


def _irreps_ls(irreps: cue.Irreps) -> List[int]:
    """List of l values, one per irrep component (expanded by multiplicity)."""
    ls = []
    for mul, ir in irreps:
        for _ in range(mul):
            ls.append(ir.l)
    return ls


def spherical_harmonics_irreps(lmax: int, p: int = 1) -> cue.Irreps:
    """Build SH irreps up to lmax.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum.
    p : int
        Parity convention: 1 for SO3 (all even parity: (-1)^l -> e),
        -1 for O3 (alternating parity: (-1)^l).
    """
    parts = []
    for l in range(lmax + 1):
        if p == 1:
            parity = "e"
        else:
            parity = "e" if l % 2 == 0 else "o"
        parts.append(f"1x{l}{parity}")
    return cue.Irreps("O3", " + ".join(parts))


# ---------------------------------------------------------------------------
# Self-connection layout (port of _precompute_sc_layout / _build_sc_weight_matrix)
# ---------------------------------------------------------------------------

def _precompute_sc_layout(
    irreps_in: cue.Irreps, irreps_out: cue.Irreps
) -> List[Tuple[int, int, int, int, int, int, float]]:
    """Precompute block-diagonal layout for self-connection weight matrix.

    Returns list of (w_offset, in_offset, out_offset, mul_in, mul_out, ir_dim, alpha).
    """
    in_offsets, offset = [], 0
    for mul, ir in irreps_in:
        in_offsets.append(offset)
        offset += mul * ir.dim

    out_offsets, offset = [], 0
    for mul, ir in irreps_out:
        out_offsets.append(offset)
        offset += mul * ir.dim

    total_mul_in = {}
    for i_in, (mul_in, ir_in) in enumerate(irreps_in):
        for i_out, (mul_out, ir_out) in enumerate(irreps_out):
            if ir_in == ir_out:
                total_mul_in[i_out] = total_mul_in.get(i_out, 0) + mul_in

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


def _sc_weight_numel(layout):
    """Total number of SC weight elements."""
    if not layout:
        return 0
    last = layout[-1]
    return last[0] + last[3] * last[4]


def _build_sc_weight_matrix(
    weight: jax.Array,
    layout: List[Tuple[int, int, int, int, int, int, float]],
    out_dim: int,
    in_dim: int,
) -> jax.Array:
    """Build dense (out_dim, in_dim) SC weight matrix from block-diagonal layout."""
    w = weight.reshape(-1)
    out = jnp.zeros((out_dim, in_dim), dtype=w.dtype)
    for w_off, in_off, out_off, mul_in, mul_out, ir_dim, alpha in layout:
        W = w[w_off:w_off + mul_in * mul_out].reshape(mul_in, mul_out)
        WT = alpha * W.T  # (mul_out, mul_in)
        if ir_dim == 1:
            out = out.at[out_off:out_off + mul_out, in_off:in_off + mul_in].add(WT)
        else:
            for c in range(ir_dim):
                out = out.at[
                    out_off + c:out_off + mul_out * ir_dim:ir_dim,
                    in_off + c:in_off + mul_in * ir_dim:ir_dim
                ].add(WT)
    return out


def _precompute_sc_scatter(layout, out_dim, in_dim):
    """Precompute scatter indices for vectorized SC weight matrix construction.

    Returns (src_indices, dst_flat_indices, alphas) as JAX arrays,
    or None if layout is empty.  Uses flat dest indices (row * in_dim + col)
    so only three arrays are needed.
    """
    src_list = []
    dst_list = []
    alpha_list = []

    for w_off, in_off, out_off, mul_in, mul_out, ir_dim, alpha in layout:
        for i in range(mul_in):
            for j in range(mul_out):
                src = w_off + i * mul_out + j
                for c in range(ir_dim):
                    src_list.append(src)
                    row = out_off + j * ir_dim + c
                    col = in_off + i * ir_dim + c
                    dst_list.append(row * in_dim + col)
                    alpha_list.append(alpha)

    if not src_list:
        return None

    return (
        jnp.array(src_list, dtype=jnp.int32),
        jnp.array(dst_list, dtype=jnp.int32),
        jnp.array(alpha_list, dtype=jnp.float32),
    )


def _build_sc_weight_matrix_fast(weight, sc_src, sc_dst, sc_alpha, out_dim, in_dim):
    """Build SC weight matrix using precomputed scatter indices (single scatter op)."""
    w = weight.reshape(-1)
    values = w[sc_src] * sc_alpha.astype(w.dtype)
    out = jnp.zeros(out_dim * in_dim, dtype=w.dtype)
    out = out.at[sc_dst].add(values)
    return out.reshape(out_dim, in_dim)


# ---------------------------------------------------------------------------
# Buffer: non-trainable variable for Flax NNX
# ---------------------------------------------------------------------------

class Buffer(nnx.Variable):
    """Non-trainable state variable (like PyTorch register_buffer)."""
    pass


# ---------------------------------------------------------------------------
# VoxelConvolution
# ---------------------------------------------------------------------------

class VoxelConvolution(nnx.Module):
    """Equivariant voxel convolution in JAX.

    Uses cuequivariance for spherical harmonics and CG coefficients.
    Static spacing only (set at construction time).

    Parameters
    ----------
    irreps_in, irreps_out, irreps_sh : str or cue.Irreps
    diameter : float
    num_radial_basis : int
    steps : tuple
    cutoff : bool
    normalize_by_lattice : bool
    remat_kernel : bool
        If True, discard the computed kernel during forward and recompute
        it from the small radial weights during backward. The kernel is
        large (O, I, X, Y, Z) but computed from tiny inputs (radial
        weights + SH + CG coefficients), so recomputation is cheap
        relative to the memory saved.
    rngs : nnx.Rngs
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        diameter: float,
        num_radial_basis: int,
        steps: Tuple[float, ...] = (1.0, 1.0, 1.0),
        cutoff: bool = True,
        normalize_by_lattice: bool = True,
        remat_kernel: bool = False,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sphere_norm: bool = True,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.irreps_in = _parse_irreps(irreps_in)
        self.irreps_out = _parse_irreps(irreps_out)
        self.irreps_sh = _parse_irreps(irreps_sh)
        self.num_radial_basis = num_radial_basis
        self.diameter = diameter
        self.cutoff = cutoff
        self.normalize_by_lattice = normalize_by_lattice
        self.remat_kernel = remat_kernel
        self.kernel_trim_threshold = kernel_trim_threshold
        self.kernel_trim_cross_section = kernel_trim_cross_section
        self.sphere_norm = sphere_norm
        self.in_dim = _irreps_dim(self.irreps_in)
        self.out_dim = _irreps_dim(self.irreps_out)

        # Build tensor product right-evaluation
        self._tp_right_fn, self._weight_numel, tp_constants = build_tp_right(
            self.irreps_in, self.irreps_sh, self.irreps_out,
        )

        # Store CG constants as individual Buffer attributes (Flax NNX pytree requirement)
        self._tp_const_keys = sorted(tp_constants.keys())
        for k in self._tp_const_keys:
            setattr(self, f'_tpc_{k}', Buffer(tp_constants[k]))

        # Self-connection layout
        self._sc_layout = _precompute_sc_layout(self.irreps_in, self.irreps_out)
        sc_numel = _sc_weight_numel(self._sc_layout)
        sc_scatter = _precompute_sc_scatter(
            self._sc_layout, self.out_dim, self.in_dim
        )
        if sc_scatter is not None:
            self._sc_src = Buffer(sc_scatter[0])
            self._sc_dst = Buffer(sc_scatter[1])
            self._sc_alpha = Buffer(sc_scatter[2])
            self._has_sc_scatter = True
        else:
            self._has_sc_scatter = False

        # Learnable parameters
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (num_radial_basis, self._weight_numel))
        )
        if sc_numel > 0:
            self.sc_weight = nnx.Param(
                jax.random.normal(rngs.params(), (sc_numel,))
            )
        else:
            self.sc_weight = nnx.Param(jnp.zeros((0,)))

        # Build lattice buffers (static)
        self._build_lattice_buffers(steps)

    def update_spacing(self, steps: tuple, _cache=None):
        """Update lattice buffers for new spacing without changing learnable params."""
        self._build_lattice_buffers(steps, _cache=_cache)

    def _build_lattice_buffers(self, steps, _cache=None):
        """Build lattice, radial basis, and spherical harmonics.

        Uses pure NumPy computation for l <= 2 (bypasses JAX/cuequivariance
        dispatch overhead). An optional ``_cache`` dict deduplicates
        computation across layers sharing the same (diameter, steps).
        """
        cache_key = (self.diameter, tuple(steps),
                     self.kernel_trim_cross_section, self.kernel_trim_threshold)
        if _cache is not None and cache_key in _cache:
            lattice, emb, sh, padding = _cache[cache_key]
        else:
            ls = [ir.l for _, ir in self.irreps_sh]
            max_l = max(ls) if ls else 0

            if max_l <= 2:
                # Fast NumPy path (microseconds, no JAX/cuequivariance overhead)
                np_lattice, np_emb, np_sh, padding = _compute_lattice_buffers_numpy(
                    self.diameter, steps, ls, self.num_radial_basis, self.cutoff,
                    kernel_trim_cross_section=self.kernel_trim_cross_section,
                    kernel_trim_threshold=self.kernel_trim_threshold,
                )
                lattice = jnp.array(np_lattice)
                emb = jnp.array(np_emb)
                sh = jnp.array(np_sh)
            else:
                # Fallback to cuequivariance for l > 2
                lattice, emb, sh, padding = self._build_lattice_buffers_cuex(steps, ls)

            if _cache is not None:
                _cache[cache_key] = (lattice, emb, sh, padding)

        self.lattice = Buffer(lattice)
        self.emb = Buffer(emb)
        self.sh = Buffer(sh)
        self.padding = padding

        # Precompute sphere voxel count (used for sphere_norm, must be concrete)
        if self.sphere_norm:
            r = self.diameter / 2
            norms = np.sqrt(np.sum(np.asarray(lattice) ** 2, axis=-1))
            self._n_sphere_voxels_val = int((norms <= r).sum())
        else:
            self._n_sphere_voxels_val = None

    def _build_lattice_buffers_cuex(self, steps, ls):
        """Fallback: build lattice buffers using cuequivariance (for l > 2)."""
        r = self.diameter / 2

        axes = []
        for step in steps:
            half = math.floor(r / step)
            half = _trim_half_extent(half, step, r,
                                     self.kernel_trim_cross_section,
                                     self.kernel_trim_threshold)
            axes.append(jnp.arange(-half, half + 1.0) * step)
        x, y, z = axes

        gx, gy, gz = jnp.meshgrid(x, y, z, indexing='ij')
        lattice = jnp.stack([gx, gy, gz], axis=-1)
        padding = tuple(s // 2 for s in lattice.shape[:3])

        norms = jnp.sqrt(jnp.sum(lattice ** 2, axis=-1))
        emb = soft_one_hot_linspace(
            x=norms, start=0.0, end=r,
            number=self.num_radial_basis,
            basis='smooth_finite', cutoff=self.cutoff,
        )

        lattice_flat = lattice.reshape(-1, 3)
        with cue.assume(cue.mul_ir):
            vec_rep = cuex.RepArray(
                cue.Irreps("O3", "1x1o"), lattice_flat
            )
            sh_rep = cuex.spherical_harmonics(ls, vec_rep, normalize=True)
        sh = sh_rep.array.reshape(*lattice.shape[:3], -1)

        return lattice, emb, sh, padding

    def _kernel_pure(self, weight, emb, sh, consts, n_sphere_voxels=None):
        """Pure-function kernel computation (no self access to Variables).

        All JAX array inputs are passed explicitly so this can be wrapped
        in jax.checkpoint.
        """
        # Cast f32 buffers to weight dtype (bf16) to keep kernel computation
        # in low precision — avoids f32 intermediates that double memory.
        emb = emb.astype(weight.dtype)
        sh = sh.astype(weight.dtype)

        w = emb @ weight

        if self.sphere_norm and n_sphere_voxels is not None:
            w = w / n_sphere_voxels
        elif self.normalize_by_lattice:
            w = w / (sh.shape[0] * sh.shape[1] * sh.shape[2])

        X, Y, Z = sh.shape[:3]
        N = X * Y * Z

        sh_flat = sh.reshape(N, -1)
        w_flat = w.reshape(N, -1)

        kernel = self._tp_right_fn(sh_flat, w_flat, consts)

        kernel = kernel.reshape(X, Y, Z, self.in_dim, self.out_dim)
        kernel = jnp.einsum('xyzio->oixyz', kernel)
        return kernel

    def kernel(self):
        """Compute the convolution kernel."""
        weight = self.weight[...]
        emb = self.emb[...]
        sh = self.sh[...]
        consts = {k: getattr(self, f'_tpc_{k}')[...] for k in self._tp_const_keys}

        n_sv = self._n_sphere_voxels_val

        if self.remat_kernel:
            return jax.checkpoint(self._kernel_pure)(weight, emb, sh, consts, n_sv)
        return self._kernel_pure(weight, emb, sh, consts, n_sv)

    def __call__(self, x):
        """Forward pass.

        Parameters
        ----------
        x : jax.Array
            Shape (batch, in_dim, D, H, W)

        Returns
        -------
        jax.Array
            Shape (batch, out_dim, D, H, W)
        """
        if self._has_sc_scatter:
            sc_w = _build_sc_weight_matrix_fast(
                self.sc_weight[...],
                self._sc_src[...], self._sc_dst[...], self._sc_alpha[...],
                self.out_dim, self.in_dim,
            )
        else:
            sc_w = jnp.zeros((self.out_dim, self.in_dim), dtype=self.sc_weight[...].dtype)

        is_1x1x1 = all(s == 1 for s in self.lattice[...].shape[:3])
        if is_1x1x1:
            k = sc_w[:, :, None, None, None].astype(x.dtype)
            return jax.lax.conv_general_dilated(
                x, k,
                window_strides=(1, 1, 1),
                padding=[(p, p) for p in self.padding],
                dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW'),
            )

        kernel = self.kernel()
        cx = kernel.shape[2] // 2
        cy = kernel.shape[3] // 2
        cz = kernel.shape[4] // 2

        # Fuse SC into kernel center.
        # When sphere_norm is active, the TP kernel weights are already divided
        # by n_sphere_voxels in _kernel_pure().  Apply the same normalization
        # to the SC weight matrix so both contributions are on the same scale.
        if self.sphere_norm and self._n_sphere_voxels_val:
            sc_w = sc_w / self._n_sphere_voxels_val
        kernel = kernel.at[:, :, cx, cy, cz].add(sc_w)

        return jax.lax.conv_general_dilated(
            x, kernel.astype(x.dtype),
            window_strides=(1, 1, 1),
            padding=[(p, p) for p in self.padding],
            dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW'),
        )


# ---------------------------------------------------------------------------
# EquivariantGate
# ---------------------------------------------------------------------------

_NORMALIZE2MOM_CACHE = {}

def _normalize2mom_factor(fn):
    """Compute normalization factor so that E[f(x)^2] = 1 for x ~ N(0,1).

    Returns scale such that scale * f(x) has second moment equal to 1.
    Matches e3nn.math.normalize2mom by using the same random sampling approach
    with the same seed (torch.Generator.manual_seed(0), 1M samples).
    """
    # Cache by function identity to avoid recomputation
    fn_key = id(fn)
    if fn_key in _NORMALIZE2MOM_CACHE:
        return _NORMALIZE2MOM_CACHE[fn_key]

    # Use torch to match e3nn's exact random samples and computation
    import torch
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64)
    # Evaluate function in JAX, then compute moment in numpy for precision
    z_jax = jnp.array(z.numpy(), dtype=jnp.float32)
    fz = np.array(fn(z_jax), dtype=np.float64)
    second_moment = float(np.mean(fz ** 2))
    factor = float(1.0 / np.sqrt(second_moment))
    _NORMALIZE2MOM_CACHE[fn_key] = factor
    return factor


class EquivariantGate(nnx.Module):
    """Equivariant gating activation.

    Splits input into (scalars, gate_values, gated_features).
    Applies scalar_act to scalars, sigmoid to gates, multiplies gates
    into gated features. Activations are normalized using normalize2mom
    convention (matching e3nn) so that the second moment is preserved.

    Parameters
    ----------
    irreps_scalars : cue.Irreps
        Scalar irreps to pass through activation.
    scalar_activation : callable
        Activation for scalars (e.g., jax.nn.relu).
    irreps_gates : cue.Irreps
        Gate scalar irreps (same count as gated irreps).
    gate_activation : callable or None
        Activation for gates (e.g., jax.nn.sigmoid). None if no gated features.
    irreps_gated : cue.Irreps
        Higher-order irreps to be gated.
    """

    def __init__(
        self,
        irreps_scalars,
        scalar_activation,
        irreps_gates,
        gate_activation,
        irreps_gated,
    ):
        self.irreps_scalars = _parse_irreps(irreps_scalars)
        self.irreps_gates = _parse_irreps(irreps_gates)
        self.irreps_gated = _parse_irreps(irreps_gated)

        self.scalar_activation = scalar_activation
        self.gate_activation = gate_activation

        # Compute normalize2mom factors (matching e3nn convention)
        self._scalar_act_factor = _normalize2mom_factor(scalar_activation)
        self._gate_act_factor = _normalize2mom_factor(gate_activation) if gate_activation is not None else 1.0

        self.scalars_dim = _irreps_dim(self.irreps_scalars)
        self.gates_dim = _irreps_dim(self.irreps_gates)
        self.gated_dim = _irreps_dim(self.irreps_gated)

        # Input: scalars + gates + gated (skip zero-dim components)
        # Output: scalars + gated (after gating)
        in_parts = [self.irreps_scalars]
        if self.gates_dim > 0:
            in_parts.append(self.irreps_gates)
        if self.gated_dim > 0:
            in_parts.append(self.irreps_gated)
        self.irreps_in = in_parts[0]
        for p in in_parts[1:]:
            self.irreps_in = self.irreps_in + p
        self.irreps_in = self.irreps_in.simplify()

        out_parts = [self.irreps_scalars]
        if self.gated_dim > 0:
            out_parts.append(self.irreps_gated)
        self.irreps_out = out_parts[0]
        for p in out_parts[1:]:
            self.irreps_out = self.irreps_out + p
        self.irreps_out = self.irreps_out.simplify()

        self.in_dim = _irreps_dim(self.irreps_in)
        self.out_dim = _irreps_dim(self.irreps_out)

        # Build gate-expand index: maps each gated feature to its gate scalar
        # e.g. for 2x1e + 1x2e: [0,0,0, 1,1,1, 2,2,2,2,2] (3 comps per l=1, 5 per l=2)
        expand_idx = []
        gate_idx = 0
        for mul, ir in self.irreps_gated:
            for _ in range(mul):
                expand_idx.extend([gate_idx] * ir.dim)
                gate_idx += 1
        self._gate_expand_idx = np.array(expand_idx, dtype=np.int32) if expand_idx else None

    def __call__(self, x):
        """Apply gating.

        Parameters
        ----------
        x : jax.Array
            Shape (..., in_dim) with features in last dimension.

        Returns
        -------
        jax.Array
            Shape (..., out_dim)
        """
        # Split input
        scalars = x[..., :self.scalars_dim]
        gates = x[..., self.scalars_dim:self.scalars_dim + self.gates_dim]
        gated = x[..., self.scalars_dim + self.gates_dim:]

        # Apply activations to scalars (with normalize2mom)
        scalars = self.scalar_activation(scalars) * jnp.array(self._scalar_act_factor, dtype=scalars.dtype)

        if self._gate_expand_idx is None:
            return scalars

        # Apply gate activation (sigmoid, with normalize2mom)
        gates = self.gate_activation(gates) * jnp.array(self._gate_act_factor, dtype=gates.dtype)

        # Expand gates to match gated features via indexing (single op, no loop)
        gate_expanded = gates[..., self._gate_expand_idx]
        gated = gated * gate_expanded

        return jnp.concatenate([scalars, gated], axis=-1)


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------

class EquivariantBatchNorm(nnx.Module):
    """Equivariant batch normalization.

    Per-irrep-type normalization by field norm. Training mode updates
    running stats; eval mode uses frozen stats.

    Parameters
    ----------
    irreps : str or cue.Irreps
    eps : float
    momentum : float
    affine : bool
    instance : bool
        If True, use instance normalization (no running stats).
    """

    def __init__(self, irreps, eps=1e-5, momentum=0.1, affine=True, instance=False):
        self.irreps = _parse_irreps(irreps)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.instance = instance
        self.dim = _irreps_dim(self.irreps)

        num_fields = sum(mul for mul, _ in self.irreps)
        num_scalar_fields = sum(mul for mul, ir in self.irreps if ir.l == 0)

        if affine:
            self.weight = nnx.Param(jnp.ones(num_fields))
            self.bias = nnx.Param(jnp.zeros(num_scalar_fields))

        if not instance:
            self.running_mean = nnx.BatchStat(jnp.zeros(num_scalar_fields))
            self.running_var = nnx.BatchStat(jnp.ones(num_fields))

        self._num_fields = num_fields
        self._num_scalar_fields = num_scalar_fields

        # Precompute groups: one group per irrep term in the (simplified) irreps.
        # Each group processes all mul fields of that irrep type as a single
        # vectorized operation, replacing the per-field loop.
        self._groups = []
        feat_offset = 0
        field_idx = 0
        scalar_idx = 0
        for mul, ir in self.irreps:
            d = ir.dim
            is_scalar = ir.l == 0
            feat_start = feat_offset
            feat_end = feat_offset + mul * d
            field_indices = list(range(field_idx, field_idx + mul))
            scalar_indices = list(range(scalar_idx, scalar_idx + mul)) if is_scalar else None
            self._groups.append({
                'start': feat_start,
                'end': feat_end,
                'ir_dim': d,
                'count': mul,
                'is_scalar': is_scalar,
                'field_idx': field_indices,
                'scalar_idx': scalar_indices,
            })
            feat_offset = feat_end
            field_idx += mul
            if is_scalar:
                scalar_idx += mul

    def __call__(self, x, *, use_running_average: bool = False):
        """Apply batch normalization.

        Parameters
        ----------
        x : jax.Array
            Shape (..., dim) with features in last dimension.
            For BN, leading dims are batch + spatial.
        use_running_average : bool
            If True, use running stats (eval mode).
        """
        # reduce_axes: all axes except the last (feature) dimension
        reduce_axes = tuple(range(x.ndim - 1))
        output_parts = []

        for g in self._groups:
            feat = x[..., g['start']:g['end']]
            fi = np.array(g['field_idx'], dtype=np.int32)

            if g['is_scalar']:
                # Scalar group: each field is 1D, process all at once
                # feat shape: (..., count)
                si = np.array(g['scalar_idx'], dtype=np.int32)
                if use_running_average and not self.instance:
                    means = self.running_mean[...][si]
                    vars_ = self.running_var[...][fi]
                else:
                    means = jnp.mean(feat, axis=reduce_axes)  # (count,)
                    vars_ = jnp.var(feat, axis=reduce_axes)   # (count,)
                    if not self.instance and not use_running_average:
                        old_means = self.running_mean[...][si]
                        old_vars = self.running_var[...][fi]
                        self.running_mean[...] = self.running_mean[...].at[si].set(
                            (1 - self.momentum) * old_means + self.momentum * means
                        )
                        self.running_var[...] = self.running_var[...].at[fi].set(
                            (1 - self.momentum) * old_vars + self.momentum * vars_
                        )

                feat = (feat - means) / jnp.sqrt(vars_ + self.eps)

                if self.affine:
                    w = self.weight[...][fi].astype(feat.dtype)
                    b = self.bias[...][si].astype(feat.dtype)
                    feat = feat * w + b

            else:
                # Non-scalar group: reshape to (..., count, ir_dim) for vectorized processing
                batch_shape = feat.shape[:-1]
                ir_dim = g['ir_dim']
                count = g['count']
                feat_r = feat.reshape(*batch_shape, count, ir_dim)

                if use_running_average and not self.instance:
                    vars_ = self.running_var[...][fi]  # (count,)
                else:
                    # Per-field var: mean of squared values over components, then over batch+spatial
                    sq_mean = jnp.mean(feat_r ** 2, axis=-1)  # (..., count)
                    ns_reduce = tuple(range(sq_mean.ndim - 1))
                    vars_ = jnp.mean(sq_mean, axis=ns_reduce)  # (count,)
                    if not self.instance and not use_running_average:
                        old_vars = self.running_var[...][fi]
                        self.running_var[...] = self.running_var[...].at[fi].set(
                            (1 - self.momentum) * old_vars + self.momentum * vars_
                        )

                inv_std = 1.0 / jnp.sqrt(vars_ + self.eps)  # (count,)
                feat_r = feat_r * inv_std[..., None]  # broadcast (count,) -> (count, 1)

                if self.affine:
                    w = self.weight[...][fi].astype(feat_r.dtype)
                    feat_r = feat_r * w[..., None]

                feat = feat_r.reshape(*batch_shape, count * ir_dim)

            output_parts.append(feat)

        return jnp.concatenate(output_parts, axis=-1)


class EquivariantLayerNorm(nnx.Module):
    """Equivariant LayerNorm respecting irrep structure.

    Mean-subtract scalars only, shared RMS across all features, affine.

    Parameters
    ----------
    irreps : str or cue.Irreps
    eps : float
    affine : bool
    """

    def __init__(self, irreps, eps=1e-5, affine=True):
        self.irreps = _parse_irreps(irreps)
        self.eps = eps
        self.affine = affine
        self.dim = _irreps_dim(self.irreps)

        self.num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0)

        if affine:
            num_groups = len(list(self.irreps))
            self.weight = nnx.Param(jnp.ones(num_groups))
            self.bias = nnx.Param(jnp.zeros(self.num_scalar))

        # Precompute scalar indices for vectorized mean subtraction
        scalar_indices = []
        offset = 0
        for mul, ir in self.irreps:
            d = ir.dim
            if ir.l == 0:
                for m in range(mul):
                    scalar_indices.append(offset + m * d)
            offset += mul * d
        if scalar_indices:
            self._scalar_indices_buf = Buffer(jnp.array(scalar_indices, dtype=jnp.int32))
            self._has_scalar_indices = True
        else:
            self._has_scalar_indices = False

        # Precompute weight-expand map: maps each feature position to its group index
        # for vectorized affine application (single multiply instead of per-block loop)
        if affine:
            weight_expand = []
            for gi, (mul, ir) in enumerate(self.irreps):
                size = mul * ir.dim
                weight_expand.extend([gi] * size)
            self._weight_expand_buf = Buffer(jnp.array(weight_expand, dtype=jnp.int32))

        # Scalar bias positions are the same as scalar indices
        self._has_scalar_bias = affine and self.num_scalar > 0 and scalar_indices

    def __call__(self, x, **kwargs):
        """Apply layer normalization. x: (..., channels)."""
        # Step 1: Subtract mean from scalar channels
        if self._has_scalar_indices:
            si = self._scalar_indices_buf[...]
            scalar_vals = x[..., si]  # (..., num_scalars)
            scalar_mean = jnp.mean(scalar_vals, axis=-1, keepdims=True)  # (..., 1)
            out = x.at[..., si].add(-scalar_mean)
        else:
            out = x

        # Step 2: Global RMS + normalize
        rms = jnp.sqrt(jnp.mean(out ** 2, axis=-1, keepdims=True) + self.eps)
        out = out / rms

        # Step 3: Affine (vectorized weight expand + per-scalar bias)
        if self.affine:
            weight = self.weight[...]
            we = self._weight_expand_buf[...]
            out = out * weight[we].astype(out.dtype)

            if self._has_scalar_bias:
                si = self._scalar_indices_buf[...]
                bias = self.bias[...].astype(out.dtype)
                out = out.at[..., si].add(bias)

        return out


class Identity(nnx.Module):
    """Identity module for optional normalization."""

    def __call__(self, x, **kwargs):
        return x


# ---------------------------------------------------------------------------
def _trilinear_upsample_align_corners(x, scale_factor):
    """Trilinear 3D upsample matching PyTorch's align_corners=True.

    Accumulates corner contributions one at a time to reduce peak intermediate
    memory (2 output-sized arrays live at once instead of 8).

    Parameters
    ----------
    x : jax.Array
        Shape (B, C, D, H, W).
    scale_factor : tuple of int
        Scale factors for (D, H, W).

    Returns
    -------
    jax.Array
        Shape (B, C, D*sf[0], H*sf[1], W*sf[2]).
    """
    B, C, D, H, W = x.shape
    oD, oH, oW = D * scale_factor[0], H * scale_factor[1], W * scale_factor[2]

    # Compute source coordinates with align_corners=True mapping
    def _coord(out_size, in_size):
        if out_size == 1:
            return jnp.zeros(out_size)
        return jnp.linspace(0, in_size - 1, out_size)

    d_coords = _coord(oD, D)
    h_coords = _coord(oH, H)
    w_coords = _coord(oW, W)

    # Floor and ceil indices
    d0 = jnp.floor(d_coords).astype(jnp.int32)
    d1 = jnp.minimum(d0 + 1, D - 1)
    h0 = jnp.floor(h_coords).astype(jnp.int32)
    h1 = jnp.minimum(h0 + 1, H - 1)
    w0 = jnp.floor(w_coords).astype(jnp.int32)
    w1 = jnp.minimum(w0 + 1, W - 1)

    # Fractional parts, reshaped for broadcasting: df(oD,1,1), hf(1,oH,1), wf(1,1,oW)
    # Cast to input dtype to avoid bf16 * f32 promotion
    df = (d_coords - jnp.floor(d_coords)).astype(x.dtype)[:, None, None]
    hf = (h_coords - jnp.floor(h_coords)).astype(x.dtype)[None, :, None]
    wf = (w_coords - jnp.floor(w_coords)).astype(x.dtype)[None, None, :]

    # Accumulate 8 trilinear corners one at a time.
    # Each iteration materializes one output-sized corner and adds it
    # to the accumulator. This lets XLA free each corner after the add,
    # reducing peak intermediate memory from 8x to ~2x output size.
    result = jnp.zeros((B, C, oD, oH, oW), dtype=x.dtype)
    for di, dw in [(d0, 1 - df), (d1, df)]:
        xd = x[:, :, di]  # (B, C, oD, H, W)
        for hi, hw in [(h0, 1 - hf), (h1, hf)]:
            xdh = xd[:, :, :, hi]  # (B, C, oD, oH, W)
            for wi, ww in [(w0, 1 - wf), (w1, wf)]:
                corner = xdh[:, :, :, :, wi]  # (B, C, oD, oH, oW)
                result = result + corner * (dw * hw * ww)

    return result


# Pooling
# ---------------------------------------------------------------------------

class EquivariantPool3d(nnx.Module):
    """Equivariant 3D pooling.

    Scalars: max pool. Higher-order: pool by norm then gather components.
    """

    def __init__(self, scale: float, steps: tuple, mode: str, irreps):
        self.scale = scale
        self.steps = steps
        self.mode = mode
        self.irreps = _parse_irreps(irreps)
        self.kernel_size = tuple(
            _pool_factor(self.scale, step) for step in self.steps
        )

    def update_spacing(self, steps: tuple, override_kernel=None):
        """Update kernel size for new spacing.

        Parameters
        ----------
        steps : tuple
            Effective spacing at this level.
        override_kernel : tuple of int or None
            If provided, use this kernel size instead of computing from
            scale/step. Used by targeted pooling.
        """
        self.steps = steps
        if override_kernel is not None:
            self.kernel_size = tuple(override_kernel)
        else:
            self.kernel_size = tuple(
                _pool_factor(self.scale, step) for step in self.steps
            )

    def __call__(self, x):
        """Pool x of shape (B, C, D, H, W)."""
        if self.mode == 'maxpool3d':
            return self._equivariant_max_pool(x)
        elif self.mode == 'average':
            return self._avg_pool(x)
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")

    def _avg_pool(self, x):
        """Average pooling via reduce_window."""
        kd, kh, kw = self.kernel_size
        # Use lax.reduce_window for avg pool
        # x: (B, C, D, H, W)
        init = 0.0
        window_shape = (1, 1, kd, kh, kw)
        strides = window_shape
        out = jax.lax.reduce_window(
            x, init, jax.lax.add, window_shape, strides, 'VALID'
        )
        return out / (kd * kh * kw)

    def _equivariant_max_pool(self, x):
        """Equivariant max pooling respecting irrep structure."""
        B = x.shape[0]
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        kd, kh, kw = self.kernel_size

        cat_list = []
        start = 0

        for l in _irreps_ls(self.irreps):
            end = start + 2 * l + 1
            temp = x[:, start:end, ...]  # (B, 2l+1, D, H, W)

            if l == 0:
                # Scalar: standard max pool using reduce_window
                ch = temp[:, 0, ...]  # (B, D, H, W)
                pooled = jax.lax.reduce_window(
                    ch, -jnp.inf, jax.lax.max,
                    (1, kd, kh, kw), (1, kd, kh, kw), 'VALID'
                )
                cat_list.append(pooled[:, None, ...])
            else:
                # Higher-order: pool based on norm, gather by argmax
                norms = jnp.sqrt(jnp.sum(temp ** 2, axis=1))  # (B, D, H, W)

                # Compute output spatial dims
                oD = D // kd
                oH = H // kh
                oW = W // kw

                # Reshape into windows: (B, oD, kd, oH, kh, oW, kw)
                norms_win = norms.reshape(B, oD, kd, oH, kh, oW, kw)

                # Transpose to group output and kernel dims: (B, oD, oH, oW, kd, kh, kw)
                norms_win = norms_win.transpose(0, 1, 3, 5, 2, 4, 6)

                # Flatten window dims and argmax
                norms_flat = norms_win.reshape(B, oD, oH, oW, kd * kh * kw)
                max_idx = jnp.argmax(norms_flat, axis=-1)  # (B, oD, oH, oW)

                # Convert flat index to 3D offset within window
                idx_d = max_idx // (kh * kw)
                idx_h = (max_idx % (kh * kw)) // kw
                idx_w = max_idx % kw

                # Compute absolute indices
                od_grid = jnp.arange(oD)[None, :, None, None]
                oh_grid = jnp.arange(oH)[None, None, :, None]
                ow_grid = jnp.arange(oW)[None, None, None, :]

                abs_d = od_grid * kd + idx_d
                abs_h = oh_grid * kh + idx_h
                abs_w = ow_grid * kw + idx_w

                # Gather ALL components at selected spatial location (single indexing op)
                b_idx = jnp.arange(B)[:, None, None, None]
                # Advanced indices at 0,2,3,4 + slice at 1 → (B, oD, oH, oW, 2l+1)
                gathered = temp[b_idx, :, abs_d, abs_h, abs_w]
                gathered = jnp.moveaxis(gathered, -1, 1)  # (B, 2l+1, oD, oH, oW)
                cat_list.append(gathered)

            start = end

        return jnp.concatenate(cat_list, axis=1)


# ---------------------------------------------------------------------------
# ConvolutionBlock
# ---------------------------------------------------------------------------

class ConvolutionBlock(nnx.Module):
    """Double convolution block with gating and normalization.

    Parameters
    ----------
    checkpoint : bool or str
        Gradient checkpointing mode:
        - False: no remat
        - True: block-level remat (saves nothing, recomputes everything)
        - 'dots': fine-grained remat using dots_with_no_batch_dims_saveable
          policy — saves conv outputs (expensive to recompute) while
          recomputing gate activations and norms (cheap)
        - 'dots_all': saves all dot-product outputs
    """

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
        checkpoint=False,
        remat_kernel: bool = False,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sphere_norm: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        irreps_in = _parse_irreps(irreps_in)
        irreps_hidden = _parse_irreps(irreps_hidden)
        irreps_sh = _parse_irreps(irreps_sh)

        # Setup normalization
        if normalization == 'None':
            make_norm = lambda irr: Identity()
        elif normalization == 'batch':
            make_norm = lambda irr: EquivariantBatchNorm(irr)
        elif normalization == 'instance':
            make_norm = lambda irr: EquivariantBatchNorm(irr, instance=True)
        elif normalization == 'layer':
            make_norm = lambda irr: EquivariantLayerNorm(irr)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        # Setup gate irreps
        irreps_scalars_parts = [(mul, ir) for mul, ir in irreps_hidden if ir.l == 0]
        irreps_gated_parts = [(mul, ir) for mul, ir in irreps_hidden if ir.l > 0]

        if irreps_scalars_parts:
            irreps_scalars = cue.Irreps("O3", " + ".join(f"{m}x{ir.l}{'e' if ir.p == 1 else 'o'}" for m, ir in irreps_scalars_parts))
        else:
            irreps_scalars = cue.Irreps("O3", "0x0e")

        if irreps_gated_parts:
            irreps_gated = cue.Irreps("O3", " + ".join(f"{m}x{ir.l}{'e' if ir.p == 1 else 'o'}" for m, ir in irreps_gated_parts))
            num_gates = sum(m for m, _ in irreps_gated_parts)
            irreps_gates = cue.Irreps("O3", f"{num_gates}x0e")
            gate_act = jax.nn.sigmoid
        else:
            irreps_gated = cue.Irreps("O3", "0x0e")
            irreps_gates = cue.Irreps("O3", "0x0e")
            gate_act = None

        scalar_act = activation[0] if isinstance(activation, (list, tuple)) else activation

        # Create gates
        self.gate1 = EquivariantGate(
            irreps_scalars, scalar_act,
            irreps_gates if gate_act else cue.Irreps("O3", "0x0e"),
            gate_act,
            irreps_gated if gate_act else cue.Irreps("O3", "0x0e"),
        )

        # First conv
        self.conv1 = VoxelConvolution(
            irreps_in, self.gate1.irreps_in, irreps_sh,
            diameter, num_radial_basis, steps, cutoff,
            remat_kernel=remat_kernel,
            kernel_trim_threshold=kernel_trim_threshold,
            kernel_trim_cross_section=kernel_trim_cross_section,
            sphere_norm=sphere_norm,
            rngs=rngs,
        )
        self.batchnorm1 = make_norm(self.gate1.irreps_in)

        # Second gate (same structure)
        self.gate2 = EquivariantGate(
            irreps_scalars, scalar_act,
            irreps_gates if gate_act else cue.Irreps("O3", "0x0e"),
            gate_act,
            irreps_gated if gate_act else cue.Irreps("O3", "0x0e"),
        )

        self.conv2 = VoxelConvolution(
            self.gate1.irreps_out, self.gate2.irreps_in, irreps_sh,
            diameter, num_radial_basis, steps, cutoff,
            remat_kernel=remat_kernel,
            kernel_trim_threshold=kernel_trim_threshold,
            kernel_trim_cross_section=kernel_trim_cross_section,
            sphere_norm=sphere_norm,
            rngs=rngs,
        )
        self.batchnorm2 = make_norm(self.gate2.irreps_in)

        self.dropout_prob = dropout_prob
        self.checkpoint = checkpoint
        self.irreps_out = self.gate2.irreps_out
        self.out_dim = _irreps_dim(self.irreps_out)

    def _forward(self, x, deterministic: bool = True, use_running_average: bool = False):
        """Core forward pass. x: (B, C, D, H, W)."""
        # First conv block
        x = self.conv1(x)
        # Transpose to channels-last for norm/gate, then back
        x = jnp.moveaxis(x, 1, -1)  # (B, D, H, W, C)
        x = self.batchnorm1(x, use_running_average=use_running_average)
        x = self.gate1(x)
        if not deterministic and self.dropout_prob > 0:
            # Simple dropout (not irrep-aware for simplicity)
            x = nnx.Dropout(rate=self.dropout_prob)(x, deterministic=False)
        x = jnp.moveaxis(x, -1, 1)  # (B, C, D, H, W)

        # Second conv block
        x = self.conv2(x)
        x = jnp.moveaxis(x, 1, -1)
        x = self.batchnorm2(x, use_running_average=use_running_average)
        x = self.gate2(x)
        if not deterministic and self.dropout_prob > 0:
            x = nnx.Dropout(rate=self.dropout_prob)(x, deterministic=False)
        x = jnp.moveaxis(x, -1, 1)

        return x

    def __call__(self, x, *, deterministic: bool = True, use_running_average: bool = False):
        """Forward pass with optional gradient checkpointing. x: (B, C, D, H, W)."""
        if not self.checkpoint:
            return self._forward(x, deterministic, use_running_average)

        # Select remat policy based on checkpoint mode
        policy = None
        if isinstance(self.checkpoint, str):
            if self.checkpoint == 'dots':
                policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable
            elif self.checkpoint == 'dots_all':
                policy = jax.checkpoint_policies.dots_saveable
            else:
                raise ValueError(f"Unknown checkpoint policy: {self.checkpoint}")

        if policy is not None:
            @functools.partial(nnx.remat, policy=policy)
            def _remat_forward(module, x):
                return module._forward(x, deterministic, use_running_average)
        else:
            @nnx.remat
            def _remat_forward(module, x):
                return module._forward(x, deterministic, use_running_average)

        return _remat_forward(self, x)

    def update_spacing(self, steps: tuple, _cache=None):
        """Update convolution lattices for new spacing."""
        self.conv1.update_spacing(steps, _cache=_cache)
        self.conv2.update_spacing(steps, _cache=_cache)


# ---------------------------------------------------------------------------
# Irreps construction helpers (mirrors layers.py _build_irreps etc.)
# ---------------------------------------------------------------------------

def _parse_ratios(ratios):
    """Parse ratios into (even_ratios, odd_ratios) tuples."""
    if isinstance(ratios, dict):
        return tuple(ratios.get('e', ())), tuple(ratios.get('o', ()))
    return tuple(ratios), tuple(ratios)


def _build_irreps(ne: int, no: int, ratios, fill_to: int = 0) -> cue.Irreps:
    """Build irreps from ne, no and ratios for each l."""
    even_ratios, odd_ratios = _parse_ratios(ratios)
    parts = []
    for l, r in enumerate(even_ratios):
        if r * ne > 0:
            parts.append(f"{r * ne}x{l}e")
    for l, r in enumerate(odd_ratios):
        if r * no > 0:
            parts.append(f"{r * no}x{l}o")

    if parts:
        irreps = cue.Irreps("O3", " + ".join(parts))
    else:
        irreps = cue.Irreps("O3", "0x0e")

    # Simplify: combine like terms
    irreps = irreps.simplify()

    # Fill to max with scalars
    if fill_to > 0 and _irreps_dim(irreps) < fill_to:
        extra = fill_to - _irreps_dim(irreps)
        irreps = (irreps + cue.Irreps("O3", f"{extra}x0e")).sort().simplify()

    return irreps


def _features_per_ne(ratios, has_odd: bool) -> int:
    """Compute features per ne unit from ratios."""
    even_ratios, odd_ratios = _parse_ratios(ratios)
    total = sum(r * (2 * l + 1) for l, r in enumerate(even_ratios))
    if has_odd:
        total += sum(r * (2 * l + 1) for l, r in enumerate(odd_ratios))
    return total


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nnx.Module):
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
        checkpoint=False,
        remat_kernel: bool = False,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sphere_norm: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        irreps_sh = _parse_irreps(irreps_sh)
        irreps_in = _parse_irreps(irreps_in)

        features_per_ne = _features_per_ne(irrep_ratios, no > 0)
        ne_max = max_features // features_per_ne

        down_blocks = []
        down_pool = []
        self.down_irreps_out = []
        self.ne_per_level = []
        self.no_per_level = []

        for n in range(n_downsample + 1):
            ne_capped = min(ne, ne_max)
            no_capped = min(no, ne_max) if no > 0 else 0

            self.ne_per_level.append(ne_capped)
            self.no_per_level.append(no_capped)

            fill_to = max_features if (fill_to_max and ne >= ne_max) else 0
            irreps_hidden = _build_irreps(ne_capped, no_capped, irrep_ratios, fill_to)

            # Per-level checkpoint: int means checkpoint first N levels only
            if isinstance(checkpoint, int):
                level_ckpt = n < checkpoint
            else:
                level_ckpt = checkpoint

            block = ConvolutionBlock(
                irreps_in, irreps_hidden, activation, irreps_sh, normalization,
                diameters[n], num_radial_basis, steps_array[n], dropout_prob, cutoff,
                checkpoint=level_ckpt, remat_kernel=remat_kernel,
                kernel_trim_threshold=kernel_trim_threshold,
                kernel_trim_cross_section=kernel_trim_cross_section,
                sphere_norm=sphere_norm, rngs=rngs,
            )
            down_blocks.append(block)
            self.down_irreps_out.append(block.irreps_out)
            irreps_in = block.irreps_out
            ne *= 2
            no *= 2

        # Pooling layers
        for n in range(n_downsample):
            down_pool.append(EquivariantPool3d(
                scales[n], steps_array[n], down_op, self.down_irreps_out[n]
            ))

        self.down_blocks = nnx.List(down_blocks)
        self.down_pool = nnx.List(down_pool)

    def update_spacing(self, steps_array: list, _cache=None, override_pool_kernels=None):
        """Update all blocks and pooling layers for new spacing.

        Parameters
        ----------
        steps_array : list of tuple
            Effective spacing at each level.
        _cache : dict or None
            Ephemeral dedup cache for lattice buffer computation.
        override_pool_kernels : dict or None
            Maps level index -> tuple of int (pool kernel per dim).
        """
        for i, block in enumerate(self.down_blocks):
            block.update_spacing(steps_array[i], _cache=_cache)
        for i, pool in enumerate(self.down_pool):
            override_k = None
            if override_pool_kernels and i in override_pool_kernels:
                override_k = override_pool_kernels[i]
            pool.update_spacing(steps_array[i], override_kernel=override_k)

    def __call__(self, x, *, deterministic=True, use_running_average=False):
        """Returns list of features at each level."""
        features = []
        for i, block in enumerate(self.down_blocks):
            x = block(x, deterministic=deterministic, use_running_average=use_running_average)
            features.append(x)
            if i < len(self.down_blocks) - 1:
                x = self.down_pool[i](x)
        return features


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class _DSHead(nnx.Module):
    """Deep supervision linear head."""

    def __init__(self, irreps_in, irreps_out, *, rngs: nnx.Rngs):
        self.layout = _precompute_sc_layout(irreps_in, irreps_out)
        self.in_dim = _irreps_dim(irreps_in)
        self.out_dim = _irreps_dim(irreps_out)
        sc_scatter = _precompute_sc_scatter(
            self.layout, self.out_dim, self.in_dim
        )
        if sc_scatter is not None:
            self._sc_src = Buffer(sc_scatter[0])
            self._sc_dst = Buffer(sc_scatter[1])
            self._sc_alpha = Buffer(sc_scatter[2])
            self._has_sc_scatter = True
        else:
            self._has_sc_scatter = False
        numel = _sc_weight_numel(self.layout)
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (numel,)))

    def __call__(self, x):
        """x: (B, C, D, H, W) -> (B, n_classes, D, H, W)."""
        w_mat = _build_sc_weight_matrix_fast(
            self.weight[...],
            self._sc_src[...], self._sc_dst[...], self._sc_alpha[...],
            self.out_dim, self.in_dim,
        )
        x_cl = jnp.moveaxis(x, 1, -1)  # (B, D, H, W, C)
        out = x_cl @ w_mat.astype(x_cl.dtype).T
        return jnp.moveaxis(out, -1, 1)


class Decoder(nnx.Module):
    """UNet decoder (upsampling path) with optional deep supervision."""

    def __init__(
        self,
        n_blocks: int,
        activation,
        irreps_sh,
        ne: int,
        no: int,
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
        checkpoint=False,
        remat_kernel: bool = False,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sphere_norm: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        irreps_sh = _parse_irreps(irreps_sh)

        self.n_blocks = n_blocks
        self.deep_supervision = deep_supervision
        self.scales = scales

        features_per_ne = _features_per_ne(irrep_ratios, no > 0)
        ne_max = max_features // features_per_ne

        even_ratios, odd_ratios = _parse_ratios(irrep_ratios)
        scalar_ratios = {'e': (sum(even_ratios) * 2,), 'o': (sum(odd_ratios) * 2,)}

        irreps_in = encoder_irreps[-1]

        up_blocks = []
        self.upsample_scale_factors = []
        block_irreps_out = []

        for n in range(n_blocks):
            ne_capped = min(ne, ne_max)
            no_capped = min(no, ne_max) if no > 0 else 0

            fill_to = max_features if (fill_to_max and ne >= ne_max) else 0

            if scalar_upsampling:
                irreps_hidden = _build_irreps(ne_capped, no_capped, scalar_ratios, fill_to)
            else:
                irreps_hidden = _build_irreps(ne_capped, no_capped, irrep_ratios, fill_to)

            skip_irreps = encoder_irreps[::-1][n + 1]

            # Per-level checkpoint: decoder level n maps to spatial level
            # n_blocks - 1 - n (decoder goes deepest-first)
            if isinstance(checkpoint, int):
                spatial_level = n_blocks - 1 - n
                level_ckpt = spatial_level < checkpoint
            else:
                level_ckpt = checkpoint

            block = ConvolutionBlock(
                irreps_in + skip_irreps, irreps_hidden, activation, irreps_sh,
                normalization, diameters[n], num_radial_basis, steps_array[n],
                dropout_prob, cutoff, checkpoint=level_ckpt,
                remat_kernel=remat_kernel,
                kernel_trim_threshold=kernel_trim_threshold,
                kernel_trim_cross_section=kernel_trim_cross_section,
                sphere_norm=sphere_norm, rngs=rngs,
            )
            up_blocks.append(block)
            block_irreps_out.append(block.irreps_out)
            irreps_in = block.irreps_out
            ne //= 2
            no //= 2

            # Upsample scale factor
            scale_factor = tuple(
                _pool_factor(scales[n], step) for step in steps_array[n]
            )
            self.upsample_scale_factors.append(scale_factor)

        self.up_blocks = nnx.List(up_blocks)

        # Deep supervision heads
        if deep_supervision and n_classes is not None:
            output_irreps = cue.Irreps("O3", f"{n_classes}x0e")
            ds_heads = []
            for i in range(n_blocks - 1):
                ds_heads.append(_DSHead(block_irreps_out[i], output_irreps, rngs=rngs))
            self.ds_heads = nnx.List(ds_heads)
        else:
            self.ds_heads = nnx.List([])

    def update_spacing(self, steps_array: list, _cache=None,
                       override_upsample_factors=None):
        """Update all blocks and upsample scale factors for new spacing.

        Parameters
        ----------
        steps_array : list of tuple
            Effective spacing at each decoder level (reversed from encoder).
        _cache : dict or None
            Ephemeral dedup cache for lattice buffer computation.
        override_upsample_factors : dict or None
            Maps decoder block index -> tuple of int (upsample factor per dim).
            Used by targeted pooling to match overridden pool kernels.
        """
        for i, block in enumerate(self.up_blocks):
            block.update_spacing(steps_array[i], _cache=_cache)
        self.upsample_scale_factors = []
        for i in range(self.n_blocks):
            if override_upsample_factors and i in override_upsample_factors:
                scale_factor = tuple(override_upsample_factors[i])
            else:
                scale_factor = tuple(
                    _pool_factor(self.scales[i], step) for step in steps_array[i]
                )
            self.upsample_scale_factors.append(scale_factor)

    def __call__(self, x, encoder_features, *, deterministic=True, use_running_average=False):
        """Forward pass with skip connections from encoder."""
        ds_outputs = []

        for i in range(self.n_blocks):
            # Upsample (align_corners=True to match PyTorch)
            sf = self.upsample_scale_factors[i]
            x = _trilinear_upsample_align_corners(x, sf)

            # Skip connection
            skip = encoder_features[::-1][i + 1]
            x = jnp.concatenate([x, skip], axis=1)

            # Conv block
            x = self.up_blocks[i](x, deterministic=deterministic, use_running_average=use_running_average)

            # Deep supervision
            if self.deep_supervision and len(self.ds_heads) > 0 and i < self.n_blocks - 1:
                ds_out = self.ds_heads[i](x)
                ds_outputs.append(ds_out)

        if self.deep_supervision and len(self.ds_heads) > 0:
            return x, ds_outputs
        return x
