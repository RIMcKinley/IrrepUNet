"""Spacing utilities for multi-resolution training.

Provides canonical spacing grids, rounding, grouping, and axis permutation
functions used by dataloaders and training scripts.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


# Allowed spacing values (mm) for rounding/grouping
SPACING_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0)


def round_to_grid(value: float, grid: tuple = SPACING_GRID) -> float:
    """Round a value to the nearest value in the grid.

    Parameters
    ----------
    value : float
        Value to round
    grid : tuple
        Allowed values to round to

    Returns
    -------
    float
        Nearest grid value
    """
    return min(grid, key=lambda x: abs(x - value))


def round_spacing_to_tolerance(spacing: tuple) -> tuple:
    """Round spacing values to the nearest values in the spacing grid.

    Grid values: 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0 mm

    Parameters
    ----------
    spacing : tuple
        Spacing values (D, H, W)

    Returns
    -------
    tuple
        Rounded spacing values
    """
    return tuple(round_to_grid(s) for s in spacing)


def group_cases_by_spacing(
    properties: Dict[str, dict],
    min_spacing: float = 0.0,
    max_inplane_spacing: float = 0.0,
    min_slice_thickness: float = 0.0,
    max_slice_thickness: float = 0.0,
    inplane_tolerance: float = 0.1,
) -> Dict[tuple, List[str]]:
    """Group cases by their spacing using tiered tolerance rounding.

    Uses tiered tolerances based on spacing magnitude:
    - Below 1.5mm: 0.25mm tolerance
    - 1.5mm to 4mm: 0.5mm tolerance
    - Above 4mm: 1.0mm tolerance

    Axis permutations are treated as equivalent (permutation invariant).
    E.g., (0.5, 0.5, 4.0) and (4.0, 0.5, 0.5) are in the same group.
    The canonical form is sorted ascending (finest spacing first).

    Parameters
    ----------
    properties : dict
        Mapping from case name to properties dict (must contain 'spacing')
    min_spacing : float
        Exclude cases with finest spacing below this value (default: 0.0, include all)
    max_inplane_spacing : float
        Exclude cases with in-plane spacing above this value (default: 0.0, include all)
    min_slice_thickness : float
        Exclude cases with slice thickness below this value (default: 0.0, include all)
    max_slice_thickness : float
        Exclude cases with slice thickness above this value (default: 0.0, include all)

    Returns
    -------
    dict
        Mapping from (rounded, canonical) spacing to list of case names
    """
    groups: Dict[tuple, List[str]] = {}

    dropped_anisotropic = 0
    for case_name, props in properties.items():
        spacing = tuple(props.get('spacing', (1.0, 1.0, 1.0)))

        # Use sorted spacing as canonical form (ascending = fine to coarse)
        canonical_spacing = tuple(sorted(spacing))
        inplane_spacing = canonical_spacing[0]  # Finest = in-plane
        slice_thickness = canonical_spacing[2]  # Coarsest = slice thickness

        # Drop cases whose in-plane axes (the two finest) are too anisotropic.
        # Uses the same isotropy definition as the preprocessing step.
        if inplane_tolerance > 0:
            ip_mean = (canonical_spacing[0] + canonical_spacing[1]) / 2
            if ip_mean > 0:
                ip_rel_diff = abs(canonical_spacing[1] - canonical_spacing[0]) / ip_mean
                if ip_rel_diff > inplane_tolerance:
                    dropped_anisotropic += 1
                    continue

        # Skip cases with spacing below min_spacing threshold
        if min_spacing > 0 and inplane_spacing < min_spacing:
            continue

        # Skip cases with in-plane spacing above max_inplane_spacing threshold
        if max_inplane_spacing > 0 and inplane_spacing > max_inplane_spacing:
            continue

        # Skip cases with slice thickness below min_slice_thickness threshold
        if min_slice_thickness > 0 and slice_thickness < min_slice_thickness:
            continue

        # Skip cases with slice thickness above max_slice_thickness threshold
        if max_slice_thickness > 0 and slice_thickness > max_slice_thickness:
            continue

        # Snap the in-plane pair to a shared value before grid-rounding so
        # that float noise at grid midpoints (e.g. (2.75, 2.75, ...)) cannot
        # split the pair across adjacent grid cells.
        ip_mean = (canonical_spacing[0] + canonical_spacing[1]) / 2
        ip_rounded = round_to_grid(ip_mean)
        sl_rounded = round_to_grid(canonical_spacing[2])
        rounded_spacing = (ip_rounded, ip_rounded, sl_rounded)

        # Find matching group (using rounded spacing as key)
        if rounded_spacing in groups:
            groups[rounded_spacing].append(case_name)
        else:
            groups[rounded_spacing] = [case_name]

    if dropped_anisotropic > 0:
        print(f"  Dropped {dropped_anisotropic} cases with in-plane anisotropy > "
              f"{inplane_tolerance:.0%}")

    return groups


def get_canonical_permutation(spacing: tuple) -> tuple:
    """Get the axis permutation to convert spacing to canonical (sorted) form.

    Parameters
    ----------
    spacing : tuple
        Original spacing (D, H, W)

    Returns
    -------
    tuple
        Permutation indices such that spacing[perm] is sorted ascending
    """
    return tuple(np.argsort(spacing))


def apply_axis_permutation(arr: np.ndarray, perm: tuple, has_channel: bool = True) -> np.ndarray:
    """Apply axis permutation to array.

    Parameters
    ----------
    arr : np.ndarray
        Array to permute. Shape (C, D, H, W) if has_channel else (D, H, W)
    perm : tuple
        Permutation indices for spatial dimensions
    has_channel : bool
        Whether array has channel dimension

    Returns
    -------
    np.ndarray
        Permuted array
    """
    if has_channel:
        # Adjust permutation for channel dimension
        full_perm = (0,) + tuple(p + 1 for p in perm)
    else:
        full_perm = perm
    return np.transpose(arr, full_perm)


