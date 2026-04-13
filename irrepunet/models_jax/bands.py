"""Architecture band computation and targeted pooling for L2+ JIT trace reuse.

Computes ranges of L2 effective spacings that produce identical integer kernel
and pool sizes at all L2+ levels.  Given a set of spacing groups, chooses L1
pool kernels to consolidate L2+ families by moving groups into larger existing
clusters.

Key concepts:
- Architecture signature: tuple of all conv kernel half-extents and pool kernel
  sizes at L2+ levels.  Two spacings with the same signature produce identical
  graphdefs and can share a compiled XLA program.
- Targeted pooling: instead of the standard k1 = floor(scale1/step1), adjust k1
  by ±1 when it would move a spacing into a larger existing L2+ family.
"""

import math
from collections import defaultdict

from irrepunet.models_jax.layers import _pool_factor


def _conv_half_extent(diameter, step):
    """Half-extent of convolution kernel in one dimension (lattice points)."""
    return math.floor((diameter / 2) / step)


def _pool_kernel(scale, step):
    """Pool kernel size in one dimension.

    Delegates to _pool_factor for rounding-tolerance-aware computation.
    """
    return _pool_factor(scale, step)


def _architecture_signature(l2_step, model_scale, n_downsample, diameters):
    """Compute the full architecture signature from L2 onward.

    Propagates l2_step through levels 2..n_downsample, computing conv kernel
    half-extents and pool kernel sizes at each level.

    Parameters
    ----------
    l2_step : tuple of float
        Effective spacing at level 2 (3D).
    model_scale : float
        Base pooling scale (typically 2.0).
    n_downsample : int
        Total number of pooling levels.
    diameters : list of float
        Convolution diameters at each level (length n_downsample + 1).

    Returns
    -------
    tuple
        Hashable signature of (conv_half_extents, pool_kernels) at all L2+ levels.
    """
    sig = []
    step = list(l2_step)

    for level in range(2, n_downsample + 1):
        # Conv kernel half-extent at this level
        for d in range(3):
            sig.append(_conv_half_extent(diameters[level], step[d]))

        # Pool kernel (only for levels 2..n_downsample-1, not the bottleneck)
        if level < n_downsample:
            scale = model_scale * (2 ** level)
            pk = []
            new_step = []
            for d in range(3):
                k = _pool_kernel(scale, step[d])
                pk.append(k)
                new_step.append(k * step[d] if k > 1 else step[d])
            sig.extend(pk)
            step = new_step

    return tuple(sig)


def _compute_l2_step(spacing, model_scale, k1_override=None):
    """Compute L2 effective spacing from input spacing and optional k1 override.

    Parameters
    ----------
    spacing : tuple of float
    model_scale : float
    k1_override : tuple of int or None

    Returns
    -------
    tuple of float
    """
    scales = [model_scale * (2 ** i) for i in range(2)]
    step = list(spacing)

    # L0 pool
    for d in range(3):
        k0 = _pool_kernel(scales[0], step[d])
        step[d] = k0 * step[d] if k0 > 1 else step[d]

    # L1 pool
    if k1_override is None:
        for d in range(3):
            k1 = _pool_kernel(scales[1], step[d])
            step[d] = k1 * step[d] if k1 > 1 else step[d]
    else:
        for d in range(3):
            k1 = k1_override[d]
            step[d] = k1 * step[d] if k1 > 1 else step[d]

    return tuple(step)


def _get_standard_k1(spacing, model_scale):
    """Compute the standard L1 pool kernel for a spacing."""
    scales = [model_scale * (2 ** i) for i in range(2)]
    step = list(spacing)

    # L0 pool
    for d in range(3):
        k0 = _pool_kernel(scales[0], step[d])
        step[d] = k0 * step[d] if k0 > 1 else step[d]

    # Standard L1 pool
    k1 = tuple(
        _pool_kernel(scales[1], step[d])
        for d in range(3)
    )
    return k1


def compute_targeted_k1_for_groups(spacing_groups, model_scale, n_downsample, diameter):
    """Compute targeted L1 pool kernels to consolidate L2+ families.

    Strategy:
    1. Compute standard L2 steps and architecture signatures for all groups
    2. Build a map of signature -> group count (cluster sizes)
    3. For each group, try k1 ±1 per dimension
    4. Only override if the candidate produces a signature that matches an
       existing LARGER cluster (net family reduction)
    5. Never reduce k1 below 2 (k1=1 eliminates pooling, causing huge spatial dims)

    Parameters
    ----------
    spacing_groups : dict
        Maps spacing tuple -> group info dict.
    model_scale : float
    n_downsample : int
    diameter : float

    Returns
    -------
    dict
        Maps spacing tuple -> tuple of int (k1 override per dimension).
        Only includes spacings where the override differs from standard.
    """
    diameters = [diameter * (2 ** i) for i in range(n_downsample + 1)]

    # Phase 1: compute standard L2 steps and signatures
    std_info = {}  # spacing -> (k1_std, l2_step, signature)
    sig_counts = defaultdict(int)  # signature -> count

    for spacing in spacing_groups:
        k1_std = _get_standard_k1(spacing, model_scale)
        l2_step = _compute_l2_step(spacing, model_scale)
        sig = _architecture_signature(l2_step, model_scale, n_downsample, diameters)
        std_info[spacing] = (k1_std, l2_step, sig)
        sig_counts[sig] += 1

    n_std_families = len(sig_counts)

    # Phase 2: try k1 ±1 for each group, see if it merges into a larger cluster
    overrides = {}
    # Track updated signature counts as we apply overrides
    new_sig_counts = defaultdict(int, sig_counts)

    # Sort groups by cluster size ascending (smallest clusters first = most
    # likely to benefit from merging into larger ones)
    sorted_spacings = sorted(
        spacing_groups.keys(),
        key=lambda s: sig_counts[std_info[s][2]],
    )

    for spacing in sorted_spacings:
        k1_std, l2_step_std, sig_std = std_info[spacing]
        best_k1 = k1_std
        best_sig = sig_std
        best_score = new_sig_counts[sig_std]  # current cluster size

        # Generate candidates: adjust each dimension by ±1 independently
        # For simplicity, try all 3^3 - 1 combinations? No, too many.
        # Just try each dimension independently (3 * 2 = 6 candidates max).
        for d in range(3):
            for delta in [-1, +1]:
                k1_cand = list(k1_std)
                k1_cand[d] = k1_std[d] + delta
                # Never go below 2 (k1=1 removes pooling, huge spatial dims)
                if k1_cand[d] < 2:
                    continue
                k1_cand = tuple(k1_cand)

                l2_cand = _compute_l2_step(spacing, model_scale, k1_cand)
                sig_cand = _architecture_signature(
                    l2_cand, model_scale, n_downsample, diameters
                )

                # Only adopt if this signature already exists with more members
                cand_count = new_sig_counts[sig_cand]
                if cand_count > best_score:
                    best_k1 = k1_cand
                    best_sig = sig_cand
                    best_score = cand_count

        if best_k1 != k1_std:
            overrides[spacing] = best_k1
            # Update counts: leave old cluster, join new cluster
            new_sig_counts[sig_std] -= 1
            new_sig_counts[best_sig] += 1

    n_new_families = len([c for c in new_sig_counts.values() if c > 0])

    if overrides:
        print(f"Targeted pooling: {len(overrides)} overrides, "
              f"L2+ families {n_std_families} -> {n_new_families}", flush=True)
    else:
        print(f"Targeted pooling: no beneficial overrides found "
              f"({n_std_families} L2+ families)", flush=True)

    return overrides


def compute_l2_step_with_override(spacing, model_scale, k1_override):
    """Compute the L2 effective spacing given an L1 pool kernel override."""
    return _compute_l2_step(spacing, model_scale, k1_override)


def compute_voxel_patch_for_family(family_l2_shape, k0, k1):
    """Compute the voxel patch size needed to produce a given L2 input shape.

    After L0 pool (kernel k0) and L1 pool (kernel k1), the feature map spatial
    dims are: patch_voxels[d] / (k0[d] * k1[d]).  For JIT reuse, all members
    of a family need the same L2 input shape.

    Parameters
    ----------
    family_l2_shape : tuple of int
        Required spatial shape at L2 input (after L0 + L1 pooling).
    k0 : tuple of int
        L0 pool kernel per dimension.
    k1 : tuple of int
        L1 pool kernel per dimension.

    Returns
    -------
    tuple of int
        Required voxel patch size (before any padding).
    """
    return tuple(family_l2_shape[d] * k0[d] * k1[d] for d in range(3))
