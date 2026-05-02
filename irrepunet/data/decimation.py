"""On-the-fly decimation for multi-resolution training.

Replaces pre-computed skip-subsampled variants (skip0_Nx, skipxy_Nx, combined)
with synthetic loader cases that apply np.take at patch-extraction time.

Semantics mirror ``preprocess.py::create_subsampled_case`` exactly so that any
case_id ``enumerate_decimation_variants`` emits corresponds to a decimation
configuration the legacy on-disk pipeline would have produced.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def compute_class_locations(
    label: np.ndarray, max_samples_per_class: int = 10000
) -> Dict[int, np.ndarray]:
    """Foreground voxel coordinates per class, capped per class (mirrors preprocess.py).

    ``label`` is (D, H, W); returns ``{class_id: (N, 3) int64}``.
    """
    out: Dict[int, np.ndarray] = {}
    for c in np.unique(label):
        if c == 0:
            continue
        coords = np.array(np.where(label == c)).T
        if len(coords) > max_samples_per_class:
            idx = np.random.choice(len(coords), max_samples_per_class, replace=False)
            coords = coords[idx]
        out[int(c)] = coords
    return out


def find_slice_axis(spacing):
    """Axis whose spacing differs most from the other two (the slice axis)."""
    s = np.array(spacing)
    differences = []
    for i in range(3):
        others = [s[j] for j in range(3) if j != i]
        mean_other = np.mean(others)
        diff = abs(s[i] - mean_other) / max(mean_other, 1e-6)
        differences.append(diff)
    return int(np.argmax(differences))


def decimate_array(
    arr: np.ndarray,
    strides: Tuple[int, int, int],
    has_channel: bool = True,
    offsets: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Apply ``np.take`` along each spatial axis with stride + start offset.

    ``strides[i] == 1`` → no-op for axis i.  ``offsets[i]`` must satisfy
    ``0 <= offsets[i] < strides[i]``.  When ``offsets == (0, 0, 0)`` the result
    is the deterministic "start at voxel 0" view used during verification.
    """
    spatial_offset = 1 if has_channel else 0
    result = arr
    for ax, stride in enumerate(strides):
        if stride <= 1:
            continue
        n = result.shape[ax + spatial_offset]
        start = int(offsets[ax]) % stride
        indices = list(range(start, n, stride))
        result = np.take(result, indices, axis=ax + spatial_offset)
    return result


def random_offsets(strides: Tuple[int, int, int], rng=None) -> Tuple[int, int, int]:
    """Draw random start offsets in ``[0, stride)`` per axis.

    ``rng`` may be ``None`` (uses ``np.random``) or a ``np.random.Generator``
    / ``RandomState`` for seeded determinism.
    """
    sample = (np.random.randint if rng is None else rng.randint)
    return tuple(
        int(sample(0, s)) if s > 1 else 0 for s in strides
    )


def decimate_class_locations(
    class_locations: Dict[int, np.ndarray],
    strides: Tuple[int, int, int],
) -> Dict[int, np.ndarray]:
    """Filter + remap foreground coordinates for a decimated view.

    A voxel at coord ``c`` survives iff ``c[ax] % strides[ax] == 0`` for every
    axis; its new coord is ``c[ax] // strides[ax]``.
    """
    out: Dict[int, np.ndarray] = {}
    strides_arr = np.asarray(strides, dtype=np.int64)
    for cls, coords in class_locations.items():
        if coords is None or len(coords) == 0:
            out[cls] = coords
            continue
        c = np.asarray(coords, dtype=np.int64)
        mask = np.ones(len(c), dtype=bool)
        for ax in range(3):
            if strides_arr[ax] > 1:
                mask &= (c[:, ax] % strides_arr[ax] == 0)
        if mask.any():
            kept = c[mask] // strides_arr[np.newaxis, :]
            out[cls] = kept.astype(coords.dtype, copy=False)
        else:
            out[cls] = np.empty((0, 3), dtype=coords.dtype)
    return out


def _decimated_shape(base_shape: Tuple[int, int, int], strides: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(int(np.ceil(base_shape[ax] / strides[ax])) for ax in range(3))


def enumerate_decimation_variants(
    base_case_id: str,
    base_props: dict,
    max_thickness: float,
    max_inplane: float,
    inplane_ratio_limit: float = 1.1,
) -> Dict[str, dict]:
    """Enumerate all synthetic (case_id → properties) pairs for a base source.

    Mirrors the subsample logic in ``preprocess.py``: slice-axis ladder first,
    then in-plane ladder applied to each slice-axis variant (and the original).

    Parameters
    ----------
    base_case_id : str
        Identifier of the source case (original or ``_upxy_*`` variant).
    base_props : dict
        pkl-like properties for the source, with keys ``spacing`` (tuple[3]),
        ``shape`` (tuple[3]) and optional ``class_locations``.
    max_thickness : float
        Maximum allowed slice thickness (mm) for the skip-axis ladder.
        Variants stop when ``spacing[skip_axis] * level > max_thickness + 1``.
    max_inplane : float
        Maximum allowed in-plane spacing (mm) for the skipxy ladder.
    inplane_ratio_limit : float
        Cap ``new_inplane <= slice_thickness * ratio_limit`` (default 1.1).

    Returns
    -------
    dict[str, dict]
        Maps synthetic case_id → properties dict. Each dict carries
        ``decimation_base_id``, ``decimation_strides``, derived ``spacing``,
        derived ``shape``, filtered ``class_locations`` and the usual
        ``is_subsampled``/``subsample_type``/``parent`` fields.
    """
    spacing = tuple(float(s) for s in base_props['spacing'])
    shape = tuple(int(s) for s in base_props['shape'])
    class_locations = base_props.get('class_locations', {}) or {}

    is_isotropic = max(spacing) / (min(spacing) + 1e-9) < 1.5
    if is_isotropic:
        skip_axes = [0, 1, 2]
    else:
        skip_axes = [find_slice_axis(spacing)]

    variants: Dict[str, dict] = {}
    original_inplane_done = False

    for skip_axis in skip_axes:
        inplane_axes = tuple(i for i in range(3) if i != skip_axis)
        ax0, ax1 = inplane_axes

        # Stage 1: slice-axis ladder.  Each entry: (suffix, strides, spacing, shape).
        slice_versions = [("", (1, 1, 1), spacing, shape)]

        current_thickness = spacing[skip_axis]
        level = 2
        while True:
            new_thickness = current_thickness * level
            if new_thickness > max_thickness + 1.0:
                break
            if shape[skip_axis] < level * 2:
                break

            strides = [1, 1, 1]
            strides[skip_axis] = level
            strides_t = tuple(strides)

            new_spacing = list(spacing)
            new_spacing[skip_axis] = new_thickness
            new_spacing_t = tuple(new_spacing)
            new_shape_t = _decimated_shape(shape, strides_t)

            suffix = f"_skip{skip_axis}_{level}x"
            case_id = f"{base_case_id}{suffix}"
            variants[case_id] = {
                'spacing': new_spacing_t,
                'shape': new_shape_t,
                'original_shape': base_props.get('original_shape'),
                'bbox': base_props.get('bbox'),
                # class_locations filtered from base as a placeholder; the
                # loader recomputes from the actual decimated seg (matches the
                # on-disk 10K-sample-per-class cap exactly).
                'class_locations': decimate_class_locations(class_locations, strides_t),
                'is_subsampled': True,
                'subsample_type': 'slice',
                'skip_axis': skip_axis,
                'skip_level': level,
                'parent': base_case_id,
                'decimation_base_id': base_case_id,
                'decimation_strides': strides_t,
            }

            slice_versions.append((suffix, strides_t, new_spacing_t, new_shape_t))
            level += 1

        # Stage 2: in-plane ladder on each slice-axis version (+ original, once).
        for base_suffix, base_strides, base_spacing, base_shape in slice_versions:
            if base_suffix == "" and original_inplane_done:
                continue

            current_inplane = (base_spacing[ax0] + base_spacing[ax1]) / 2
            slice_thickness = base_spacing[skip_axis]

            level = 2
            while True:
                new_inplane = current_inplane * level
                if new_inplane > max_inplane:
                    break
                if new_inplane > slice_thickness * inplane_ratio_limit:
                    break
                if base_shape[ax0] < level * 2 or base_shape[ax1] < level * 2:
                    break

                strides = list(base_strides)
                strides[ax0] *= level
                strides[ax1] *= level
                strides_t = tuple(strides)

                new_spacing = list(base_spacing)
                new_spacing[ax0] = base_spacing[ax0] * level
                new_spacing[ax1] = base_spacing[ax1] * level
                new_spacing_t = tuple(new_spacing)
                new_shape_t = _decimated_shape(shape, strides_t)

                inplane_suffix = f"_skipxy_{level}x"
                full_suffix = base_suffix + inplane_suffix
                case_id = f"{base_case_id}{full_suffix}"
                variants[case_id] = {
                    'spacing': new_spacing_t,
                    'shape': new_shape_t,
                    'original_shape': base_props.get('original_shape'),
                    'bbox': base_props.get('bbox'),
                    'class_locations': decimate_class_locations(class_locations, strides_t),
                    'is_subsampled': True,
                    'subsample_type': 'inplane' if base_suffix == "" else 'combined',
                    'inplane_axes': inplane_axes,
                    'skip_level': level,
                    'parent': base_case_id if base_suffix == "" else f"{base_case_id}{base_suffix}",
                    'decimation_base_id': base_case_id,
                    'decimation_strides': strides_t,
                }
                level += 1

            if base_suffix == "":
                original_inplane_done = True

    return variants
