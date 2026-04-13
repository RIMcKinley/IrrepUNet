"""Canonical inference functions for e3nnUNet.

Provides sliding_window_inference() with Gaussian weighting, batched patch
processing, automatic model projection, and regular-grid padding.

Also provides hierarchical_inference() for cascaded multi-resolution
inference: coarse-to-fine detection that avoids full-resolution processing
of empty regions.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .models import E3nnUNet


# =============================================================================
# Gaussian importance map
# =============================================================================

def make_gaussian_importance_map(patch_size: tuple, sigma_scale: float = 0.125) -> torch.Tensor:
    """3D Gaussian importance map (sigma = 1/8 patch size, matching nnUNet).

    Parameters
    ----------
    patch_size : tuple
        Patch dimensions (D, H, W)
    sigma_scale : float
        Sigma as fraction of patch size per dimension

    Returns
    -------
    torch.Tensor
        Importance map of shape (1, 1, D, H, W)
    """
    coords = []
    for s in patch_size:
        sigma = s * sigma_scale
        c = torch.arange(s, dtype=torch.float32) - (s - 1) / 2.0
        g = torch.exp(-0.5 * (c / max(sigma, 1e-6)) ** 2)
        coords.append(g)
    importance = coords[0][:, None, None] * coords[1][None, :, None] * coords[2][None, None, :]
    importance = importance / importance.max()
    importance = torch.clamp(importance, min=1e-4)
    return importance.unsqueeze(0).unsqueeze(0)


# =============================================================================
# Helpers
# =============================================================================

def _mm_to_voxels(patch_size_mm, spacing, img_shape):
    """Convert mm patch size to voxels, clamped to image size."""
    if isinstance(patch_size_mm, (int, float)):
        patch_size_mm = (float(patch_size_mm),) * 3
    spacing_arr = np.array(spacing)
    patch_voxels = tuple(
        max(8, int(round(mm / sp)))
        for mm, sp in zip(patch_size_mm, spacing_arr)
    )
    return tuple(min(pv, s) for pv, s in zip(patch_voxels, img_shape))


def _needs_projection(model):
    """Check if model contains native e3nn layers that need projection."""
    try:
        from irrepunet.models.layers import VoxelConvolution
    except ImportError:
        return False
    return any(isinstance(m, VoxelConvolution) for m in model.modules())


def _compute_padding(dim, patch, step):
    """Compute reflect-padding for a perfectly regular patch grid.

    Without padding, the last patch snaps back to (dim - patch), creating
    irregular overlap and position-dependent Gaussian weighting. This
    computes the minimum padding so every position is exactly step apart.
    """
    if dim <= patch:
        return 0
    n_steps = -(-(dim - patch) // step)  # ceil division
    return patch + n_steps * step - dim


# =============================================================================
# Sliding window inference
# =============================================================================

def sliding_window_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    spacing: tuple,
    patch_size_mm: Union[float, Tuple[float, ...]] = 80.0,
    overlap: float = 0.69,
    device: str = 'cuda',
    use_fp16: bool = True,
    mirror_axes: Optional[Tuple[int, ...]] = None,
    sw_batch_size: int = 4,
    native_e3nn: bool = False,
    model_spacing: Optional[tuple] = None,
) -> np.ndarray:
    """Batched sliding window inference with Gaussian weighting.

    Automatically projects native e3nn models to conventional Conv3d for
    fast inference. Reflect-pads the image for a regular patch grid to
    ensure equivariant coverage.

    Parameters
    ----------
    model : torch.nn.Module
        Model (native e3nn or pre-projected). Native models are
        automatically projected to the image spacing.
    image : np.ndarray
        (C, D, H, W) preprocessed image.
    spacing : tuple
        Voxel spacing in mm (used for patch voxel sizing).
    patch_size_mm : float or tuple
        Patch size in mm (isotropic if float).
    overlap : float
        Overlap fraction between patches (default 0.69, ~4x patch density).
    device : str
        CUDA device.
    use_fp16 : bool
        Use FP16 autocast.
    mirror_axes : tuple, optional
        Axes to mirror for test-time augmentation (e.g. (0, 1, 2)).
    sw_batch_size : int
        Number of patches per forward pass.
    native_e3nn : bool
        If True, skip projection and pass spacing to model forward call.
        Use this for native e3nn inference (e.g. pyramid kernel models).
    model_spacing : tuple, optional
        Override spacing passed to model forward call in native_e3nn mode.
        If None, uses ``spacing``. Useful for spacing-jitter TTA where
        patch sizing uses true spacing but the model sees jittered spacing.

    Returns
    -------
    np.ndarray
        Softmax probabilities (C, D, H, W).
    """
    from .models import project_to_spacing

    device = torch.device(device)

    # Project native e3nn models to Conv3d for this spacing, or keep native
    if native_e3nn:
        _spacing = model_spacing if model_spacing is not None else spacing
    elif _needs_projection(model):
        model = project_to_spacing(model, spacing, use_2d=True)
        _spacing = None
    else:
        _spacing = None
    model = model.to(device).eval()

    img_shape = image.shape[1:]  # (D, H, W)
    patch_voxels = _mm_to_voxels(patch_size_mm, spacing, img_shape)
    pd, ph, pw = patch_voxels
    D, H, W = img_shape

    # Compute step sizes and padding for regular grid
    step_d = max(1, int(pd * (1 - overlap)))
    step_h = max(1, int(ph * (1 - overlap)))
    step_w = max(1, int(pw * (1 - overlap)))

    total_pad_d = _compute_padding(D, pd, step_d)
    total_pad_h = _compute_padding(H, ph, step_h)
    total_pad_w = _compute_padding(W, pw, step_w)

    # Symmetric reflect-pad
    pad_d0, pad_d1 = total_pad_d // 2, total_pad_d - total_pad_d // 2
    pad_h0, pad_h1 = total_pad_h // 2, total_pad_h - total_pad_h // 2
    pad_w0, pad_w1 = total_pad_w // 2, total_pad_w - total_pad_w // 2

    image_tensor = torch.from_numpy(image).float()  # (C, D, H, W)
    if total_pad_d > 0 or total_pad_h > 0 or total_pad_w > 0:
        image_tensor = F.pad(
            image_tensor.unsqueeze(0),
            (pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1),
            mode='reflect',
        ).squeeze(0)

    Dp, Hp, Wp = image_tensor.shape[1:]

    # Regular grid patch positions (no boundary snapping)
    d_positions = list(range(0, Dp - pd + 1, step_d))
    h_positions = list(range(0, Hp - ph + 1, step_h))
    w_positions = list(range(0, Wp - pw + 1, step_w))

    patch_locs = [(d, h, w)
                  for d in d_positions for h in h_positions for w in w_positions]
    n_patches = len(patch_locs)

    # Gaussian importance map on GPU
    importance = make_gaussian_importance_map(patch_voxels).to(device)

    # TTA flip combos
    if mirror_axes and len(mirror_axes) > 0:
        spatial_dims = [a + 2 for a in mirror_axes]
        n_combos = 1 << len(spatial_dims)
        flip_combos = []
        for combo in range(n_combos):
            dims = [spatial_dims[i] for i in range(len(spatial_dims))
                    if combo & (1 << i)]
            flip_combos.append(dims)
    else:
        flip_combos = [[]]
        n_combos = 1

    # GPU output buffers (lazily allocated after first forward)
    output = None
    weight_sum = torch.zeros(1, 1, Dp, Hp, Wp, device=device)

    def _forward(batch):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_fp16):
            out = model(batch, _spacing) if _spacing is not None else model(batch)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return F.softmax(out.float(), dim=1)

    print(f"Processing {n_patches} patches (patch_voxels={patch_voxels}, "
          f"sw_batch_size={sw_batch_size}, tta={n_combos}x)...")

    with torch.no_grad():
        for batch_start in range(0, n_patches, sw_batch_size):
            batch_locs = patch_locs[batch_start:batch_start + sw_batch_size]

            # Extract batch of patches → (B, C, pd, ph, pw)
            patches = torch.stack([
                image_tensor[:, d:d+pd, h:h+ph, w:w+pw]
                for d, h, w in batch_locs
            ]).to(device)

            # TTA: average predictions over flip combos
            pred_sum = None
            for flip_dims in flip_combos:
                inp = torch.flip(patches, flip_dims) if flip_dims else patches
                pred = _forward(inp)
                if flip_dims:
                    pred = torch.flip(pred, flip_dims)
                pred_sum = pred if pred_sum is None else pred_sum + pred

            pred_avg = pred_sum if n_combos == 1 else pred_sum / n_combos

            if output is None:
                n_classes = pred_avg.shape[1]
                output = torch.zeros(1, n_classes, Dp, Hp, Wp, device=device)

            # Scatter-add weighted predictions into GPU buffer
            imp = importance[0]  # (1, pd, ph, pw)
            for i, (d, h, w) in enumerate(batch_locs):
                output[0, :, d:d+pd, h:h+ph, w:w+pw] += pred_avg[i] * imp
                weight_sum[0, :, d:d+pd, h:h+ph, w:w+pw] += imp

    # Normalize and crop back to original size
    output = output / weight_sum.clamp(min=1e-8)
    output = output[:, :, pad_d0:pad_d0+D, pad_h0:pad_h0+H, pad_w0:pad_w0+W]
    return output[0].cpu().numpy()  # (C, D, H, W)


# =============================================================================
# Scale-jitter TTA
# =============================================================================

def jitter_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    spacing: tuple,
    patch_size_mm: Union[float, Tuple[float, ...]] = 100.0,
    overlap: float = 0.69,
    device: str = 'cuda',
    use_fp16: bool = True,
    mirror_axes: Optional[Tuple[int, ...]] = None,
    sw_batch_size: int = 4,
    jitter_scale: float = 0.2,
    num_jitter: int = 9,
    native_e3nn: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-scale jitter TTA via majority voting.

    Runs sliding_window_inference at num_jitter patch sizes uniformly
    spaced from (1 - jitter_scale) * patch_size_mm to
    (1 + jitter_scale) * patch_size_mm. Each scale produces a binary
    prediction; the final output is the vote fraction and majority mask.

    The model is projected once and reused across all scales (unless
    native_e3nn=True, in which case the native model is used directly).

    Parameters
    ----------
    model : torch.nn.Module
        Model (native e3nn or pre-projected).
    image : np.ndarray
        (C, D, H, W) preprocessed image.
    spacing : tuple
        Voxel spacing in mm.
    patch_size_mm : float or tuple
        Base patch size in mm.
    overlap : float
        Overlap fraction between patches.
    device : str
        CUDA device.
    use_fp16 : bool
        Use FP16 autocast.
    mirror_axes : tuple, optional
        Axes to mirror for flip TTA (applied per scale).
    sw_batch_size : int
        Number of patches per forward pass.
    jitter_scale : float
        Jitter range as fraction of base patch size (default 0.2 = ±20%).
    num_jitter : int
        Number of scales to evaluate (default 9).
    native_e3nn : bool
        If True, skip projection and pass spacing to model forward call.

    Returns
    -------
    probability : np.ndarray
        (D, H, W) float32, fraction of scales predicting lesion (class > 0).
    mask : np.ndarray
        (D, H, W) int16, majority vote (probability > 0.5).
    """
    from .models import project_to_spacing

    # Project model once if needed (skip if native mode)
    if not native_e3nn and _needs_projection(model):
        model = project_to_spacing(model, spacing, use_2d=True)
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()

    # Parse base patch size
    if isinstance(patch_size_mm, (int, float)):
        base_mm = np.array([float(patch_size_mm)] * 3)
    else:
        base_mm = np.array([float(x) for x in patch_size_mm])

    # Generate uniform scale factors
    scales = np.linspace(1.0 - jitter_scale, 1.0 + jitter_scale, num_jitter)

    print(f"Jitter TTA: {num_jitter} scales from {scales[0]:.3f}x to {scales[-1]:.3f}x "
          f"(base={base_mm.tolist()}, range={jitter_scale})")

    img_shape = image.shape[1:]  # (D, H, W)
    vote_sum = np.zeros(img_shape, dtype=np.float32)

    for i, scale in enumerate(scales):
        scaled_mm = tuple(base_mm * scale)
        print(f"  [{i+1}/{num_jitter}] scale={scale:.3f}, patch_mm="
              f"({scaled_mm[0]:.0f}, {scaled_mm[1]:.0f}, {scaled_mm[2]:.0f})")

        probs = sliding_window_inference(
            model, image, spacing,
            patch_size_mm=scaled_mm,
            overlap=overlap,
            device=device,
            use_fp16=use_fp16,
            mirror_axes=mirror_axes,
            sw_batch_size=sw_batch_size,
            native_e3nn=native_e3nn,
        )

        # Binary vote: any class > 0 counts as lesion
        pred = probs.argmax(axis=0)
        vote_sum += (pred > 0).astype(np.float32)

    probability = vote_sum / num_jitter
    mask = (probability > 0.5).astype(np.int16)

    print(f"  Jitter TTA complete: {(mask > 0).sum()} lesion voxels "
          f"(mean prob in mask: {probability[mask > 0].mean():.3f})" if (mask > 0).any()
          else "  Jitter TTA complete: 0 lesion voxels")

    return probability, mask


def spacing_jitter_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    spacing: tuple,
    patch_size_mm: Union[float, Tuple[float, ...]] = 100.0,
    overlap: float = 0.69,
    device: str = 'cuda',
    use_fp16: bool = True,
    mirror_axes: Optional[Tuple[int, ...]] = None,
    sw_batch_size: int = 4,
    jitter_scale: float = 0.2,
    num_jitter: int = 9,
    native_e3nn: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spacing-jitter TTA via majority voting.

    Re-projects the native e3nn model at num_jitter different spacings
    uniformly scaled from (1 - jitter_scale) to (1 + jitter_scale) of the
    true spacing. Each projection produces different Conv3d kernels, giving
    the model a different "view" of the same voxel data. Each projection's
    prediction is binarized and the final output is the vote fraction.

    When native_e3nn=True, skips projection and passes the jittered spacing
    directly to the model's forward call.

    Requires a native (unprojected) e3nn model — will raise if given a
    pre-projected model since it cannot be re-projected.

    Parameters
    ----------
    model : torch.nn.Module
        Native e3nn model (must contain VoxelConvolution layers).
    image : np.ndarray
        (C, D, H, W) preprocessed image.
    spacing : tuple
        True voxel spacing in mm.
    patch_size_mm : float or tuple
        Patch size in mm (stays constant across jitter scales).
    overlap : float
        Overlap fraction between patches.
    device : str
        CUDA device.
    use_fp16 : bool
        Use FP16 autocast.
    mirror_axes : tuple, optional
        Axes to mirror for flip TTA (applied per scale).
    sw_batch_size : int
        Number of patches per forward pass.
    jitter_scale : float
        Jitter range as fraction of spacing (default 0.2 = +/-20%).
    num_jitter : int
        Number of spacings to evaluate (default 9).
    native_e3nn : bool
        If True, skip projection and pass jittered spacing to model forward call.

    Returns
    -------
    probability : np.ndarray
        (D, H, W) float32, fraction of spacings predicting lesion (class > 0).
    mask : np.ndarray
        (D, H, W) int16, majority vote (probability > 0.5).
    """
    from .models import project_to_spacing

    if not native_e3nn and not _needs_projection(model):
        raise ValueError(
            "spacing_jitter_inference requires a native e3nn model "
            "(with VoxelConvolution layers) so it can re-project at each "
            "jittered spacing. Got an already-projected model."
        )

    base_spacing = np.array([float(s) for s in spacing])
    scales = np.linspace(1.0 - jitter_scale, 1.0 + jitter_scale, num_jitter)

    print(f"Spacing jitter TTA: {num_jitter} spacings from {scales[0]:.3f}x to "
          f"{scales[-1]:.3f}x (base={[round(s, 3) for s in spacing]})")

    img_shape = image.shape[1:]
    vote_sum = np.zeros(img_shape, dtype=np.float32)

    for i, scale in enumerate(scales):
        jittered_spacing = tuple(base_spacing * scale)
        print(f"  [{i+1}/{num_jitter}] scale={scale:.3f}, spacing="
              f"({jittered_spacing[0]:.3f}, {jittered_spacing[1]:.3f}, {jittered_spacing[2]:.3f})")

        if native_e3nn:
            # Native mode: pass jittered spacing to model, true spacing for voxel sizing
            probs = sliding_window_inference(
                model, image, spacing,
                patch_size_mm=patch_size_mm,
                overlap=overlap,
                device=device,
                use_fp16=use_fp16,
                mirror_axes=mirror_axes,
                sw_batch_size=sw_batch_size,
                native_e3nn=True,
                model_spacing=jittered_spacing,
            )
        else:
            # Re-project model at jittered spacing
            projected = project_to_spacing(model, jittered_spacing, use_2d=True)
            projected = projected.to(torch.device(device)).eval()

            # Run sliding window (pass projected model + true spacing for voxel sizing)
            probs = sliding_window_inference(
                projected, image, spacing,
                patch_size_mm=patch_size_mm,
                overlap=overlap,
                device=device,
                use_fp16=use_fp16,
                mirror_axes=mirror_axes,
                sw_batch_size=sw_batch_size,
            )

            del projected
            torch.cuda.empty_cache()

        pred = probs.argmax(axis=0)
        vote_sum += (pred > 0).astype(np.float32)

    probability = vote_sum / num_jitter
    mask = (probability > 0.5).astype(np.int16)

    print(f"  Spacing jitter TTA complete: {(mask > 0).sum()} lesion voxels "
          f"(mean prob in mask: {probability[mask > 0].mean():.3f})" if (mask > 0).any()
          else "  Spacing jitter TTA complete: 0 lesion voxels")

    return probability, mask


# =============================================================================
# Hierarchical multi-resolution inference
# =============================================================================

def _resample_volume(volume, old_spacing, new_spacing, order=1):
    """Resample a 3D volume from old_spacing to new_spacing.

    Parameters
    ----------
    volume : np.ndarray
        3D array (D, H, W).
    old_spacing : tuple
        Current voxel spacing in mm.
    new_spacing : tuple
        Target voxel spacing in mm.
    order : int
        Interpolation order (1=linear for images, 0=nearest for labels).

    Returns
    -------
    np.ndarray
        Resampled volume.
    """
    from scipy.ndimage import zoom
    zoom_factors = tuple(o / n for o, n in zip(old_spacing, new_spacing))
    return zoom(volume, zoom_factors, order=order)


def _find_regions(mask, spacing, padding_mm=10.0):
    """Find bounding boxes around connected components with padding.

    Parameters
    ----------
    mask : np.ndarray
        Binary 3D mask (D, H, W).
    spacing : tuple
        Voxel spacing in mm.
    padding_mm : float
        Padding in mm to add around each region.

    Returns
    -------
    list of tuple
        Each element is ((d0, d1), (h0, h1), (w0, w1)) in voxel coords.
    """
    from scipy.ndimage import label, find_objects
    labeled, n_components = label(mask)
    if n_components == 0:
        return []

    padding_vox = tuple(max(1, int(round(padding_mm / s))) for s in spacing)
    shape = mask.shape

    regions = []
    for slices in find_objects(labeled):
        if slices is None:
            continue
        bbox = []
        for i, sl in enumerate(slices):
            lo = max(0, sl.start - padding_vox[i])
            hi = min(shape[i], sl.stop + padding_vox[i])
            bbox.append((lo, hi))
        regions.append(tuple(bbox))

    return regions


def _merge_overlapping_regions(regions):
    """Merge overlapping or adjacent bounding boxes.

    Uses iterative pairwise merging until no more overlaps exist.

    Parameters
    ----------
    regions : list of tuple
        Each element is ((d0, d1), (h0, h1), (w0, w1)).

    Returns
    -------
    list of tuple
        Merged regions.
    """
    if len(regions) <= 1:
        return regions

    def overlaps(a, b):
        for (a0, a1), (b0, b1) in zip(a, b):
            if a1 <= b0 or b1 <= a0:
                return False
        return True

    def merge(a, b):
        return tuple((min(a0, b0), max(a1, b1)) for (a0, a1), (b0, b1) in zip(a, b))

    merged = list(regions)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = set()
        for i in range(len(merged)):
            if i in used:
                continue
            current = merged[i]
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                if overlaps(current, merged[j]):
                    current = merge(current, merged[j])
                    used.add(j)
                    changed = True
            new_merged.append(current)
        merged = new_merged

    return merged


def hierarchical_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    spacing: tuple,
    resolution_levels: List[Tuple[float, float, float]],
    patch_size_mm: Union[float, Tuple[float, ...]] = 80.0,
    overlap: float = 0.69,
    device: str = 'cuda',
    use_fp16: bool = True,
    mirror_axes: Optional[Tuple[int, ...]] = None,
    sw_batch_size: int = 4,
    coarse_threshold: float = 0.3,
    padding_mm: float = 10.0,
    checkpoint: Optional[dict] = None,
) -> np.ndarray:
    """Cascaded multi-resolution inference: coarse detection, fine refinement.

    Runs inference at progressively finer resolutions, only processing
    regions where lesions were detected at the previous (coarser) level.
    This avoids running expensive full-resolution inference on empty regions.

    Algorithm:
    1. Resample image to the coarsest resolution level
    2. Run sliding_window_inference on the full coarse volume
    3. Threshold softmax output at coarse_threshold to detect candidate regions
    4. Find connected components and extract bounding boxes with padding
    5. For each finer level: extract crops from the image at that resolution,
       run inference on each crop, keep only regions that persist
    6. At native resolution: compose all refined predictions into the
       native-resolution output

    Parameters
    ----------
    model : torch.nn.Module
        Native e3nn model (will be projected at each resolution level).
    image : np.ndarray
        (C, D, H, W) preprocessed image at native spacing.
    spacing : tuple
        Native voxel spacing in mm.
    resolution_levels : list of (float, float, float)
        Target spacings from coarsest to finest. The last level should
        be close to the native spacing. Example:
        [(5.0, 5.0, 5.0), (2.0, 2.0, 2.0), (1.0, 1.0, 1.0)]
    patch_size_mm : float or tuple
        Patch size in mm for sliding window inference.
    overlap : float
        Overlap fraction between patches.
    device : str
        CUDA device.
    use_fp16 : bool
        Use FP16 autocast.
    mirror_axes : tuple, optional
        Axes for mirror TTA.
    sw_batch_size : int
        Patches per forward pass.
    coarse_threshold : float
        Softmax threshold at coarse levels (lower = more sensitive, fewer
        missed lesions). Default 0.3.
    padding_mm : float
        Padding in mm around detected regions. Default 10mm.
    checkpoint : dict, optional
        Checkpoint dict that may contain 'projected_levels' from
        export_hierarchical_bundle(). If present, pre-projected weights
        are loaded directly instead of projecting at runtime.

    Returns
    -------
    np.ndarray
        Softmax probabilities (C, D, H, W) at native resolution.
    """
    from .models import project_to_spacing
    from scipy.ndimage import zoom

    native_shape = image.shape[1:]  # (D, H, W)
    native_spacing = tuple(float(s) for s in spacing)

    # Validate resolution levels are coarse-to-fine
    for i in range(len(resolution_levels) - 1):
        cur = np.mean(resolution_levels[i])
        nxt = np.mean(resolution_levels[i + 1])
        if cur < nxt:
            raise ValueError(
                f"Resolution levels must be coarse-to-fine. Level {i} "
                f"({resolution_levels[i]}) is finer than level {i+1} "
                f"({resolution_levels[i+1]})."
            )

    print(f"Hierarchical inference: {len(resolution_levels)} levels, "
          f"native_shape={native_shape}, native_spacing={native_spacing}")

    # ---- Pre-project model for all resolution levels + native ----
    # Each unique spacing needs one projection; cache and reuse across crops.
    # If the checkpoint contains pre-projected weights (from
    # export_hierarchical_bundle), load them directly — no e3nn needed.
    import time as _time
    all_spacings = [tuple(float(s) for s in lvl) for lvl in resolution_levels]
    all_spacings.append(native_spacing)

    # Check for pre-projected models in checkpoint (coarse levels, full models)
    cached_levels = {}
    if checkpoint is not None and 'projected_levels' in checkpoint:
        cached_levels = checkpoint['projected_levels']

    # Check for per-level dedup fragments (new format)
    level_fragments = {}
    n_downsample_bundle = None
    if checkpoint is not None and 'level_fragments' in checkpoint:
        level_fragments = checkpoint['level_fragments']
        n_downsample_bundle = checkpoint.get('n_downsample', model.n_downsample)

    # Check for native architecture projections (legacy full-model format)
    native_projections = {}
    if checkpoint is not None and 'native_projections' in checkpoint:
        native_projections = checkpoint['native_projections']

    from .models.distill import compute_architecture_key, _assemble_state_dict

    projected_models = {}
    arch_cache = {}  # {arch_key: projected_model} — runtime dedup across spacings
    n_from_bundle = 0
    print(f"\n  Preparing models for {len(all_spacings)} spacing(s)...")
    for sp in all_spacings:
        if sp in projected_models:
            continue

        # 1. Coarse-level cache (full models from bundle)
        if sp in cached_levels:
            t0 = _time.time()
            cached = cached_levels[sp]
            if isinstance(cached, torch.nn.Module):
                proj = cached.to(torch.device(device)).eval()
            else:
                proj = project_to_spacing(model, sp, use_2d=True)
                proj.load_state_dict(cached)
                proj = proj.to(torch.device(device)).eval()
            dt = _time.time() - t0
            print(f"    {sp} -> loaded from bundle in {dt:.1f}s")
            projected_models[sp] = proj
            n_from_bundle += 1
            continue

        arch = compute_architecture_key(model, sp)

        # 2. Runtime architecture cache (reuse across spacings with same arch)
        if arch in arch_cache:
            projected_models[sp] = arch_cache[arch]
            print(f"    {sp} -> reused cached architecture")
            continue

        # 3. Per-level fragment assembly (new dedup format)
        if level_fragments:
            assembled = _assemble_state_dict(
                level_fragments, arch, n_downsample_bundle)
            if assembled is not None:
                t0 = _time.time()
                proj = project_to_spacing(
                    model, sp, use_2d=True, skip_kernels=True)
                proj.load_state_dict(assembled)
                proj = proj.to(torch.device(device)).eval()
                dt = _time.time() - t0
                n_frags = n_downsample_bundle * 2 + 2  # enc + dec + other
                print(f"    {sp} -> assembled from {n_frags} fragments in {dt:.1f}s")
                projected_models[sp] = proj
                arch_cache[arch] = proj
                n_from_bundle += 1
                continue

        # 4. Legacy native projections (full models by architecture key)
        if native_projections and arch in native_projections:
            t0 = _time.time()
            proj = native_projections[arch].to(torch.device(device)).eval()
            dt = _time.time() - t0
            print(f"    {sp} -> loaded native arch from bundle in {dt:.1f}s")
            projected_models[sp] = proj
            arch_cache[arch] = proj
            n_from_bundle += 1
            continue

        # 5. Full projection from e3nn model (fallback)
        t0 = _time.time()
        proj = project_to_spacing(model, sp, use_2d=True)
        proj = proj.to(torch.device(device)).eval()
        dt = _time.time() - t0
        print(f"    {sp} -> projected in {dt:.1f}s")
        projected_models[sp] = proj
        arch_cache[arch] = proj

    print(f"  {len(projected_models)} unique projection(s) ready "
          f"({n_from_bundle} from bundle)")

    # ---- Level 0: Coarsest resolution, full volume ----
    coarse_spacing = all_spacings[0]

    # Resample native image to coarse resolution
    zoom_to_coarse = tuple(ns / cs for ns, cs in zip(native_spacing, coarse_spacing))
    coarse_image = np.stack([
        zoom(image[c], zoom_to_coarse, order=1) for c in range(image.shape[0])
    ])
    coarse_shape = coarse_image.shape[1:]

    print(f"\n  Level 0: spacing={coarse_spacing}, shape={coarse_shape}")
    print(f"    Running full-volume inference...")

    # Run at coarse spacing using pre-projected model
    # Coarse levels only need detection, not precise segmentation:
    # use minimal overlap and no TTA for speed.
    coarse_probs = sliding_window_inference(
        projected_models[coarse_spacing], coarse_image, coarse_spacing,
        patch_size_mm=patch_size_mm,
        overlap=0.33,
        device=device,
        use_fp16=use_fp16,
        mirror_axes=None,
        sw_batch_size=sw_batch_size,
    )

    # Detect candidate regions at coarse level
    # Use max non-background probability
    coarse_fg_prob = 1.0 - coarse_probs[0]  # prob of any foreground class
    coarse_mask = (coarse_fg_prob > coarse_threshold).astype(np.int8)
    n_lesion_voxels = coarse_mask.sum()
    print(f"    Coarse detection: {n_lesion_voxels} voxels above "
          f"threshold {coarse_threshold}")

    if n_lesion_voxels == 0:
        # No lesions detected at coarse level — return background everywhere
        print("    No lesions detected at coarse level. Returning background.")
        n_classes = coarse_probs.shape[0]
        result = np.zeros((n_classes,) + native_shape, dtype=np.float32)
        result[0] = 1.0  # All background
        return result

    # Find bounding boxes around coarse detections
    coarse_regions = _find_regions(coarse_mask, coarse_spacing, padding_mm)
    coarse_regions = _merge_overlapping_regions(coarse_regions)
    print(f"    Found {len(coarse_regions)} region(s) after merging")

    # ---- Levels 1..N-1: Progressive refinement ----
    # Track regions in native voxel coordinates throughout
    # Convert coarse regions to native coordinates
    active_regions = []
    for region in coarse_regions:
        native_region = tuple(
            (int(round(lo / zoom_to_coarse[d])),
             min(native_shape[d], int(round(hi / zoom_to_coarse[d]))))
            for d, (lo, hi) in enumerate(region)
        )
        active_regions.append(native_region)

    n_classes = coarse_probs.shape[0]

    for level_idx in range(1, len(resolution_levels)):
        level_spacing = all_spacings[level_idx]

        # Determine zoom from native to this level's spacing
        zoom_native_to_level = tuple(
            ns / ls for ns, ls in zip(native_spacing, level_spacing)
        )

        total_crop_voxels = 0
        new_regions = []

        print(f"\n  Level {level_idx}: spacing={level_spacing}, "
              f"{len(active_regions)} region(s) to process")

        for r_idx, region in enumerate(active_regions):
            # Extract crop from native image
            slices = tuple(slice(lo, hi) for lo, hi in region)
            native_crop = image[(slice(None),) + slices]
            crop_native_shape = native_crop.shape[1:]

            # Resample crop to this level's spacing
            level_crop = np.stack([
                zoom(native_crop[c], zoom_native_to_level, order=1)
                for c in range(native_crop.shape[0])
            ])
            crop_shape = level_crop.shape[1:]
            total_crop_voxels += np.prod(crop_shape)

            print(f"    Region {r_idx}: native_crop={crop_native_shape} "
                  f"-> level_crop={crop_shape}")

            # Run inference on this crop (pre-projected model)
            # Intermediate levels only refine detection: minimal overlap, no TTA.
            crop_probs = sliding_window_inference(
                projected_models[level_spacing], level_crop, level_spacing,
                patch_size_mm=patch_size_mm,
                overlap=0.33,
                device=device,
                use_fp16=use_fp16,
                mirror_axes=None,
                sw_batch_size=sw_batch_size,
            )

            # Threshold to refine regions
            crop_fg_prob = 1.0 - crop_probs[0]
            crop_mask = (crop_fg_prob > coarse_threshold).astype(np.int8)

            if crop_mask.sum() == 0:
                print(f"    Region {r_idx}: no lesions at this resolution, dropping")
                continue

            # Find sub-regions within this crop (in level-spacing coords)
            sub_regions = _find_regions(crop_mask, level_spacing, padding_mm)
            sub_regions = _merge_overlapping_regions(sub_regions)

            # Map sub-regions back to native coordinates
            for sub_region in sub_regions:
                native_sub = tuple(
                    (max(0, region[d][0] + int(round(lo / zoom_native_to_level[d]))),
                     min(native_shape[d],
                         region[d][0] + int(round(hi / zoom_native_to_level[d]))))
                    for d, (lo, hi) in enumerate(sub_region)
                )
                new_regions.append(native_sub)

        # Merge refined regions and use them for the next level
        active_regions = _merge_overlapping_regions(new_regions)
        full_vol_voxels = np.prod(
            [int(round(s / z)) for s, z in zip(native_shape, zoom_native_to_level)]
        )
        print(f"    Total crop voxels: {total_crop_voxels} / {full_vol_voxels} "
              f"({100 * total_crop_voxels / max(1, full_vol_voxels):.1f}% of full)")
        print(f"    Refined to {len(active_regions)} region(s)")

        if not active_regions:
            print("    All regions dropped. Returning background.")
            result = np.zeros((n_classes,) + native_shape, dtype=np.float32)
            result[0] = 1.0
            return result

    # ---- Final level: compose at native resolution ----
    # Run inference at native spacing on the remaining active regions
    print(f"\n  Final composition at native spacing={native_spacing}")
    print(f"    {len(active_regions)} region(s) to process at native resolution")

    result = np.zeros((n_classes,) + native_shape, dtype=np.float32)
    result[0] = 1.0  # Background by default
    result_mask = np.zeros(native_shape, dtype=bool)

    for r_idx, region in enumerate(active_regions):
        slices = tuple(slice(lo, hi) for lo, hi in region)
        native_crop = image[(slice(None),) + slices]
        crop_shape = native_crop.shape[1:]

        print(f"    Region {r_idx}: shape={crop_shape}, "
              f"voxels={np.prod(crop_shape)}")

        crop_probs = sliding_window_inference(
            projected_models[native_spacing], native_crop, native_spacing,
            patch_size_mm=patch_size_mm,
            overlap=overlap,
            device=device,
            use_fp16=use_fp16,
            mirror_axes=mirror_axes,
            sw_batch_size=sw_batch_size,
        )

        # Place into result (overwrite background)
        result[(slice(None),) + slices] = crop_probs
        result_mask[slices] = True

    total_native_voxels = sum(
        np.prod([hi - lo for lo, hi in region]) for region in active_regions
    )
    # Free projected models
    del projected_models

    print(f"\n  Hierarchical inference complete:")
    print(f"    Native voxels processed: {total_native_voxels} / {np.prod(native_shape)} "
          f"({100 * total_native_voxels / np.prod(native_shape):.1f}%)")
    n_lesion = (result.argmax(axis=0) > 0).sum()
    print(f"    Final lesion voxels: {n_lesion}")

    return result


# =============================================================================
# End-to-end NIfTI prediction
# =============================================================================

def predict_nifti(
    checkpoint_path: str,
    input_path: str,
    output_path: str,
    device: str = 'cuda:0',
    patch_size_mm: float = 80.0,
    overlap: float = 0.69,
    use_fp16: bool = True,
    mirror_axes: Optional[Tuple[int, ...]] = None,
    save_probabilities: bool = False,
    sw_batch_size: int = 4,
    jitter_scale: Optional[float] = None,
    num_jitter: int = 9,
    spacing_jitter_scale: Optional[float] = None,
    hierarchical: bool = False,
    resolution_levels: Optional[List[Tuple[float, float, float]]] = None,
    coarse_threshold: float = 0.3,
    padding_mm: float = 10.0,
    native_e3nn: bool = False,
):
    """End-to-end: load model, preprocess NIfTI, infer, save NIfTI.

    Preprocessing: reorient to RAS, crop to nonzero, z-score normalize.
    Postprocessing: uncrop, reorient back to original orientation.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint
    input_path : str
        Path to input NIfTI
    output_path : str
        Path to output NIfTI
    device : str
        CUDA device
    patch_size_mm : float
        Patch size in mm
    overlap : float
        Overlap fraction
    use_fp16 : bool
        Use FP16 autocast
    mirror_axes : tuple, optional
        Axes for TTA
    save_probabilities : bool
        Save softmax probabilities instead of argmax
    sw_batch_size : int
        Number of patches per forward pass
    jitter_scale : float, optional
        If set, run patch-size jitter TTA (e.g. 0.2 = ±20%).
        Saves both mask and probability map.
    num_jitter : int
        Number of jitter scales (default 9). Used by both jitter modes.
    spacing_jitter_scale : float, optional
        If set, run spacing-jitter TTA (e.g. 0.2 = ±20%).
        Re-projects the model at each jittered spacing.
        Saves both mask and probability map.
    hierarchical : bool
        If True, use hierarchical multi-resolution inference.
    resolution_levels : list of tuple, optional
        Resolution levels for hierarchical inference, coarse to fine.
        Each tuple is (d_mm, h_mm, w_mm). If None and hierarchical=True,
        defaults to [(4.0, 4.0, 4.0), (2.0, 2.0, 2.0)].
    coarse_threshold : float
        Softmax threshold for coarse detection in hierarchical mode.
    padding_mm : float
        Padding in mm around detected regions in hierarchical mode.
    native_e3nn : bool
        If True, skip projection and pass spacing to model forward call.
    """
    import nibabel as nib
    from nibabel import orientations as nib_orient

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, checkpoint = E3nnUNet.load_checkpoint(checkpoint_path, device='cpu')

    if native_e3nn:
        print("  Native e3nn mode: skipping projection, spacing passed at forward time")

    # Load NIfTI
    print(f"Loading image from {input_path}")
    nii = nib.load(str(input_path))
    original_affine = nii.affine.copy()
    original_header = nii.header.copy()
    original_shape = nii.shape[:3]

    # Reorient to RAS
    original_ornt = nib.io_orientation(nii.affine)
    ras_ornt = nib_orient.axcodes2ornt(('R', 'A', 'S'))
    ornt_transform = nib_orient.ornt_transform(original_ornt, ras_ornt)
    ras_data = nib_orient.apply_orientation(nii.get_fdata(), ornt_transform).astype(np.float32)
    new_affine = nii.affine @ nib_orient.inv_ornt_aff(ornt_transform, nii.shape)
    ras_nii = nib.Nifti1Image(ras_data, new_affine, nii.header)

    spacing = tuple(float(s) for s in ras_nii.header.get_zooms()[:3])
    ras_shape = ras_data.shape
    print(f"  RAS shape: {ras_shape}, spacing: {spacing}")

    # Crop to nonzero
    ras_data_4d = ras_data[np.newaxis, ...]  # (1, D, H, W)
    from scipy.ndimage import binary_fill_holes
    nonzero_mask = ras_data_4d[0] != 0
    nonzero_mask = binary_fill_holes(nonzero_mask)
    coords = np.argwhere(nonzero_mask)
    if len(coords) > 0:
        bbox_lb = coords.min(axis=0)
        bbox_ub = coords.max(axis=0) + 1
        bbox = tuple((int(lb), int(ub)) for lb, ub in zip(bbox_lb, bbox_ub))
    else:
        bbox = tuple((0, s) for s in ras_shape)
    slices = tuple(slice(b[0], b[1]) for b in bbox)
    cropped = ras_data_4d[(slice(None),) + slices]
    print(f"  Cropped shape: {cropped.shape[1:]}")

    # Z-score normalize (foreground mask only, matching training preprocessing)
    fg_mask = cropped[0] != 0
    fg_values = cropped[0][fg_mask]
    mean_val = float(fg_values.mean())
    std_val = float(fg_values.std())
    normalized = np.zeros_like(cropped)
    normalized[0][fg_mask] = (cropped[0][fg_mask] - mean_val) / max(std_val, 1e-8)

    # Run inference (model projection handled automatically)
    reverse_transform = nib_orient.ornt_transform(ras_ornt, original_ornt)

    if hierarchical:
        # Use bundled resolution levels if available, else default
        if resolution_levels is None:
            if 'hierarchical_resolution_levels' in checkpoint:
                resolution_levels = [
                    tuple(s) for s in checkpoint['hierarchical_resolution_levels']
                ]
                print(f"  Using bundled resolution levels: {resolution_levels}")
            else:
                resolution_levels = [(4.0, 4.0, 4.0), (2.0, 2.0, 2.0)]

        probs = hierarchical_inference(
            model, normalized, spacing,
            resolution_levels=resolution_levels,
            patch_size_mm=patch_size_mm,
            overlap=overlap,
            device=device,
            use_fp16=use_fp16,
            mirror_axes=mirror_axes,
            sw_batch_size=sw_batch_size,
            coarse_threshold=coarse_threshold,
            padding_mm=padding_mm,
            checkpoint=checkpoint,
        )

        # Uncrop
        if save_probabilities:
            full_probs = np.zeros((probs.shape[0],) + ras_shape, dtype=np.float32)
            full_probs[(slice(None),) + slices] = probs
            output_data = np.stack([
                nib_orient.apply_orientation(full_probs[c], reverse_transform)
                for c in range(full_probs.shape[0])
            ], axis=-1)  # (D, H, W, C)
        else:
            pred_cropped = probs.argmax(axis=0).astype(np.int16)
            full_pred = np.zeros(ras_shape, dtype=np.int16)
            full_pred[slices] = pred_cropped
            output_data = nib_orient.apply_orientation(full_pred, reverse_transform).astype(np.int16)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_nii = nib.Nifti1Image(output_data, original_affine, original_header)
        if not save_probabilities:
            out_nii.header.set_data_dtype(np.int16)
        nib.save(out_nii, str(output_path))
        print(f"Saved: {output_path}")
        return

    _use_jitter = jitter_scale is not None or spacing_jitter_scale is not None

    if _use_jitter:
        if spacing_jitter_scale is not None:
            jitter_fn = spacing_jitter_inference
            jitter_kw = dict(jitter_scale=spacing_jitter_scale, num_jitter=num_jitter)
        else:
            jitter_fn = jitter_inference
            jitter_kw = dict(jitter_scale=jitter_scale, num_jitter=num_jitter)

        prob_cropped, mask_cropped = jitter_fn(
            model, normalized, spacing,
            patch_size_mm=patch_size_mm,
            overlap=overlap,
            device=device,
            use_fp16=use_fp16,
            mirror_axes=mirror_axes,
            sw_batch_size=sw_batch_size,
            native_e3nn=native_e3nn,
            **jitter_kw,
        )

        # Uncrop + reorient mask
        full_mask = np.zeros(ras_shape, dtype=np.int16)
        full_mask[slices] = mask_cropped
        output_data = nib_orient.apply_orientation(full_mask, reverse_transform).astype(np.int16)

        # Uncrop + reorient probability
        full_prob = np.zeros(ras_shape, dtype=np.float32)
        full_prob[slices] = prob_cropped
        prob_data = nib_orient.apply_orientation(full_prob, reverse_transform).astype(np.float32)

        # Save mask
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_nii = nib.Nifti1Image(output_data, original_affine, original_header)
        out_nii.header.set_data_dtype(np.int16)
        nib.save(out_nii, str(output_path))
        print(f"Saved mask: {output_path}")

        # Save probability map alongside
        prob_path = output_path.parent / output_path.name.replace('.nii', '_prob.nii')
        prob_nii = nib.Nifti1Image(prob_data, original_affine, original_header)
        nib.save(prob_nii, str(prob_path))
        print(f"Saved probability: {prob_path}")

    else:
        probs = sliding_window_inference(
            model, normalized, spacing,
            patch_size_mm=patch_size_mm,
            overlap=overlap,
            device=device,
            use_fp16=use_fp16,
            mirror_axes=mirror_axes,
            sw_batch_size=sw_batch_size,
            native_e3nn=native_e3nn,
        )

        # Uncrop
        if save_probabilities:
            full_probs = np.zeros((probs.shape[0],) + ras_shape, dtype=np.float32)
            full_probs[(slice(None),) + slices] = probs
            output_data = np.stack([
                nib_orient.apply_orientation(full_probs[c], reverse_transform)
                for c in range(full_probs.shape[0])
            ], axis=-1)  # (D, H, W, C)
        else:
            pred_cropped = probs.argmax(axis=0).astype(np.int16)
            full_pred = np.zeros(ras_shape, dtype=np.int16)
            full_pred[slices] = pred_cropped
            output_data = nib_orient.apply_orientation(full_pred, reverse_transform).astype(np.int16)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_nii = nib.Nifti1Image(output_data, original_affine, original_header)
        if not save_probabilities:
            out_nii.header.set_data_dtype(np.int16)
        nib.save(out_nii, str(output_path))
        print(f"Saved: {output_path}")
