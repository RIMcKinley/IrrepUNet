"""Training infrastructure for JAX E3nnUNet.

Standard Flax NNX training with Dice + CE loss for medical image segmentation.
"""

import functools
import threading
import time
from collections import deque

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax


def configure_memory_optimizations(disable_xla_remat=False):
    """Configure JAX/XLA flags for memory optimization.

    Call this before any JIT compilation (before model creation or first
    train_step call).

    Parameters
    ----------
    disable_xla_remat : bool
        If True, disable XLA's automatic rematerialization pass. Use this
        when applying manual remat via checkpoint='dots' on ConvolutionBlock
        to prevent XLA from making conflicting remat decisions.
    """
    if disable_xla_remat:
        jax.config.update('jax_compiler_enable_remat_pass', False)


def dice_loss(logits, labels, n_classes, no_background=False):
    """Soft Dice loss for segmentation.

    Parameters
    ----------
    logits : jax.Array
        Shape (B, n_classes, D, H, W).
    labels : jax.Array
        Shape (B, D, H, W) with integer class labels, or (B, n_classes, D, H, W) one-hot.
    n_classes : int
    no_background : bool
        If True, skip class 0 (background) in dice computation.

    Returns
    -------
    jax.Array
        Scalar loss.
    """
    # Convert to probabilities
    probs = jax.nn.softmax(logits, axis=1)

    # One-hot encode labels if needed
    if labels.ndim == logits.ndim - 1:
        one_hot = jax.nn.one_hot(labels, n_classes)  # (B, D, H, W, n_classes)
        one_hot = jnp.moveaxis(one_hot, -1, 1)  # (B, n_classes, D, H, W)
    else:
        one_hot = labels

    # Optionally skip background class
    if no_background and n_classes > 1:
        probs = probs[:, 1:]
        one_hot = one_hot[:, 1:]

    # Compute per-class Dice
    smooth = 1e-5
    dims = tuple(range(2, logits.ndim))  # spatial dims

    intersection = jnp.sum(probs * one_hot, axis=dims)  # (B, C)
    cardinality = jnp.sum(probs + one_hot, axis=dims)  # (B, C)

    dice = (2 * intersection + smooth) / (cardinality + smooth)

    return 1.0 - jnp.mean(dice)


def cross_entropy_loss(logits, labels):
    """Standard cross-entropy loss for segmentation.

    Parameters
    ----------
    logits : jax.Array
        Shape (B, n_classes, D, H, W).
    labels : jax.Array
        Shape (B, D, H, W) with integer class labels.

    Returns
    -------
    jax.Array
        Scalar loss.
    """
    # Move class dim to last: (B, D, H, W, C)
    logits = jnp.moveaxis(logits, 1, -1)
    # Flatten spatial
    B = logits.shape[0]
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1).astype(jnp.int32)

    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat))


def dice_ce_loss(logits, labels, n_classes, dice_weight=1.0, ce_weight=1.0,
                 no_background=False):
    """Combined Dice + CE loss.

    Parameters
    ----------
    logits : jax.Array
        Shape (B, n_classes, D, H, W).
    labels : jax.Array
        Shape (B, D, H, W) with integer labels.
    n_classes : int
    dice_weight : float
    ce_weight : float
    no_background : bool
        If True, skip class 0 in dice computation.

    Returns
    -------
    jax.Array
        Scalar loss.
    """
    d = dice_loss(logits, labels, n_classes, no_background=no_background)
    ce = cross_entropy_loss(logits, labels)
    return dice_weight * d + ce_weight * ce


def deep_supervision_loss(outputs_list, labels, weights=None, n_classes=2,
                          no_background=False):
    """Weighted multi-scale loss for deep supervision.

    Parameters
    ----------
    outputs_list : list of jax.Array
        From coarsest to finest resolution.
    labels : jax.Array
        Ground truth at finest resolution. Shape (B, D, H, W).
    weights : list of float or None
        Per-scale weights. If None, exponentially decaying.
    n_classes : int
    no_background : bool
        If True, skip class 0 in dice computation.

    Returns
    -------
    jax.Array
        Scalar loss.
    """
    n_scales = len(outputs_list)
    if weights is None:
        weights = [0.5 ** i for i in range(n_scales - 1, -1, -1)]
        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

    loss = jnp.zeros(())
    for i, (out, w) in enumerate(zip(outputs_list, weights)):
        # Downsample labels to match output resolution if needed
        out_shape = out.shape[2:]  # spatial
        label_shape = labels.shape[1:]  # spatial

        if out_shape != label_shape:
            # Nearest-neighbor downsample for labels
            # labels: (B, D, H, W) -> resize spatial dims
            B = labels.shape[0]
            labels_float = labels.astype(jnp.float32)
            labels_resized = jax.image.resize(
                labels_float, (B, *out_shape), method='nearest'
            ).astype(jnp.int32)
        else:
            labels_resized = labels

        loss = loss + w * dice_ce_loss(out, labels_resized, n_classes,
                                       no_background=no_background)

    return loss


def create_train_state(model, learning_rate=1e-3):
    """Create optimizer for the model.

    Parameters
    ----------
    model : E3nnUNet
        The model (Flax NNX module).
    learning_rate : float

    Returns
    -------
    nnx.Optimizer
    """
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    return optimizer


def train_step(model, optimizer, batch, n_classes):
    """Single training step.

    Parameters
    ----------
    model : E3nnUNet
    optimizer : nnx.Optimizer
    batch : dict with 'image' and 'label'
    n_classes : int

    Returns
    -------
    loss : jax.Array
    """
    def loss_fn(model):
        logits = model(batch['image'], deterministic=False, use_running_average=False)
        if isinstance(logits, list):
            return deep_supervision_loss(logits, batch['label'], n_classes=n_classes)
        return dice_ce_loss(logits, batch['label'], n_classes)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def eval_step(model, batch, n_classes):
    """Single evaluation step.

    Parameters
    ----------
    model : E3nnUNet
    batch : dict with 'image' and 'label'
    n_classes : int

    Returns
    -------
    loss : jax.Array
    """
    logits = model(batch['image'], deterministic=True, use_running_average=True)
    if isinstance(logits, list):
        return deep_supervision_loss(logits, batch['label'], n_classes=n_classes)
    return dice_ce_loss(logits, batch['label'], n_classes)


def create_jitted_train_step(model, optimizer, n_classes, donate=True,
                             no_background=False):
    """Create a JIT-compiled training step with optional buffer donation.

    Uses functional JAX JIT with donate_argnums to allow XLA to reuse
    input buffers (model params, optimizer state) for outputs. This avoids
    allocating duplicate buffers during the parameter update, reducing
    peak memory by ~param_size.

    Parameters
    ----------
    model : E3nnUNet
        The model (Flax NNX module). Will be mutated in-place each step.
    optimizer : nnx.Optimizer
        The optimizer. Will be mutated in-place each step.
    n_classes : int
        Number of output classes for loss computation.
    donate : bool
        If True, donate input state buffers to XLA for reuse.

    Returns
    -------
    callable
        A function ``step(batch) -> loss`` that runs a training step
        and updates model/optimizer state in-place.
    """
    graphdef, state = nnx.split((model, optimizer))

    donate_argnums = (0,) if donate else ()

    @functools.partial(jax.jit, donate_argnums=donate_argnums)
    def _functional_step(state, x, y):
        (model_inner, opt_inner) = nnx.merge(graphdef, state)

        def loss_fn(m):
            logits = m(x, deterministic=False, use_running_average=False)
            if isinstance(logits, list):
                return deep_supervision_loss(logits, y, n_classes=n_classes,
                                             no_background=no_background)
            return dice_ce_loss(logits, y, n_classes,
                                no_background=no_background)

        loss, grads = nnx.value_and_grad(loss_fn)(model_inner)
        opt_inner.update(model_inner, grads)

        _, new_state = nnx.split((model_inner, opt_inner))
        return loss, new_state

    # Mutable state holder (replaced each step)
    state_holder = [state]

    def step(batch):
        loss, new_state = _functional_step(
            state_holder[0], batch['image'], batch['label']
        )
        state_holder[0] = new_state
        # Update the original model/optimizer objects in-place
        nnx.update((model, optimizer), new_state)
        return loss

    return step


def create_jitted_train_step_dynamic(model, optimizer, n_classes, donate=True,
                                     no_background=False):
    """JIT-compiled training step that supports dynamic spacing.

    Re-splits model/optimizer every step and passes graphdef as a static
    argument. When spacing changes (via model.update_spacing()), the new
    graphdef triggers JIT retracing. Each unique spacing is cached separately.

    Usage::

        step = create_jitted_train_step_dynamic(model, optimizer, n_classes)
        for batch in dataloader:
            model.update_spacing(batch['spacing'])
            loss = step(batch)

    Parameters
    ----------
    model : E3nnUNet
        The model (Flax NNX module). Will be mutated in-place each step.
    optimizer : nnx.Optimizer
        The optimizer. Will be mutated in-place each step.
    n_classes : int
        Number of output classes for loss computation.
    donate : bool
        If True, donate input state buffers to XLA for reuse.

    Returns
    -------
    callable
        A function ``step(batch) -> loss`` that runs a training step
        and updates model/optimizer state in-place.
    """
    donate_argnums = (1,) if donate else ()

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=donate_argnums)
    def _functional_step(graphdef, state, x, y):
        (model_inner, opt_inner) = nnx.merge(graphdef, state)

        def loss_fn(m):
            logits = m(x, deterministic=False, use_running_average=False)
            if isinstance(logits, list):
                return deep_supervision_loss(logits, y, n_classes=n_classes,
                                             no_background=no_background)
            return dice_ce_loss(logits, y, n_classes,
                                no_background=no_background)

        loss, grads = nnx.value_and_grad(loss_fn)(model_inner)
        opt_inner.update(model_inner, grads)
        _, new_state = nnx.split((model_inner, opt_inner))
        return loss, new_state

    # Cache graphdef keyed by (spacing, override_pool_key) to avoid deep array
    # comparison every step. graphdef includes spacing-dependent buffers
    # (lattice, emb, sh) and pool kernel sizes; when neither spacing nor pool
    # overrides have changed, skip the full split and reuse the cached graphdef.
    cached = [None, None]  # [graphdef, cache_key]
    cached[0], _ = nnx.split((model, optimizer))
    cached[1] = (model.spacing, getattr(model, '_override_pool_key', None))

    def step(batch):
        graphdef, state = nnx.split((model, optimizer))
        cache_key = (model.spacing, getattr(model, '_override_pool_key', None))
        if cache_key != cached[1]:
            cached[0] = graphdef
            cached[1] = cache_key
        loss, new_state = _functional_step(
            cached[0], state, batch['image'], batch['label']
        )
        nnx.update((model, optimizer), new_state)
        return loss

    # Expose internal JIT function for AOT warmup compilation
    step.jitted_fn = _functional_step

    return step


# =============================================================================
# Nested JIT Training Step
# =============================================================================

class _L2PlusView(nnx.Module):
    """Lightweight view into model's L2+ submodules for nested JIT.

    References (not copies) the model's encoder blocks[2:], pools[2:],
    decoder blocks[:-2], and DS heads. Used to split/merge L2+ state
    independently inside the training step's loss_fn.
    """

    def __init__(self, model):
        n = model.n_downsample
        self.enc_blocks = nnx.List(list(model.encoder.down_blocks)[2:])
        self.enc_pools = nnx.List(list(model.encoder.down_pool)[2:])
        self.dec_blocks = nnx.List(list(model.decoder.up_blocks)[:-2])
        n_l2_ds = n - 2
        if model.decoder.deep_supervision and len(model.decoder.ds_heads) > 0:
            self.ds_heads = nnx.List(list(model.decoder.ds_heads)[:n_l2_ds])
        else:
            self.ds_heads = nnx.List([])
        self.deep_supervision = model.decoder.deep_supervision
        self.n_dec_blocks = n - 2
        self.upsample_sfs = model.decoder.upsample_scale_factors[:-2]


def _make_l2p_inner_jit(deterministic, use_running_average):
    """Create an inner JIT for L2+ forward with fixed mode flags.

    deterministic and use_running_average are used in Python `if` statements
    inside the model blocks, so they must be concrete (not traced). We bake
    them into the function definition so the inner JIT sees them as constants.

    Returns a @jit function(graphdef, state, x_l2) -> (x_out, ds_outputs, new_state).
    """
    from irrepunet.models_jax.layers import _trilinear_upsample_align_corners

    @functools.partial(jax.jit, static_argnums=(0,))
    def _l2p_inner(graphdef, state, x_l2):
        view = nnx.merge(graphdef, state)

        # Encoder: blocks[2..N] with pools[2..N-1]
        features = []
        x = x_l2
        for i, block in enumerate(view.enc_blocks):
            x = block(x, deterministic=deterministic,
                      use_running_average=use_running_average)
            features.append(x)
            if i < len(view.enc_blocks) - 1:
                x = view.enc_pools[i](x)

        # Decoder: blocks[0..N-3] with upsampling and skip connections
        x = features[-1]
        ds_outputs = []
        for i in range(view.n_dec_blocks):
            sf = view.upsample_sfs[i]
            x = _trilinear_upsample_align_corners(x, sf)
            skip = features[::-1][i + 1]
            x = jnp.concatenate([x, skip], axis=1)
            x = view.dec_blocks[i](x, deterministic=deterministic,
                                   use_running_average=use_running_average)
            if view.deep_supervision and len(view.ds_heads) > 0:
                if i < len(view.ds_heads):
                    ds_outputs.append(view.ds_heads[i](x))

        _, new_state = nnx.split(view)
        return x, ds_outputs, new_state

    return _l2p_inner


# Pre-create for the two modes used in practice
_l2p_inner_train = _make_l2p_inner_jit(deterministic=False, use_running_average=False)
_l2p_inner_eval = _make_l2p_inner_jit(deterministic=True, use_running_average=True)


def create_nested_jit_train_step(model, optimizer, n_classes, donate=True,
                                 no_background=False):
    """Training step with nested JIT for L2+ trace reuse across spacing families.

    Like create_jitted_train_step_dynamic, but wraps the L2+ portion of the
    forward pass in an inner @jit. When the outer step is retraced for a new
    spacing in the same L2+ family, JAX reuses the cached L2+ trace instead
    of re-tracing all L2+ encoder/decoder blocks from scratch.

    Empirically ~38% faster compilation across multiple spacings sharing
    L2+ families. No runtime performance difference (XLA flattens nested JIT).
    Gradients are exact (not approximate).

    Parameters
    ----------
    model : E3nnUNet
        The model (Flax NNX module). Must have n_downsample >= 3.
    optimizer : nnx.Optimizer
        The optimizer.
    n_classes : int
        Number of output classes.
    donate : bool
        If True, donate input state buffers to XLA for reuse.

    Returns
    -------
    callable
        A function ``step(batch) -> loss``.
    """
    from irrepunet.models_jax.layers import (
        _trilinear_upsample_align_corners,
        _build_sc_weight_matrix_fast,
    )

    n_downsample = model.n_downsample
    deep_supervision = model.deep_supervision
    donate_argnums = (1,) if donate else ()

    # Cache L2+ graphdef keyed by L2+ family.
    # Graphdef is concrete (Python structure, not arrays) so it's valid
    # as a static arg across trace levels.
    l2p_cached_gd = [None]

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=donate_argnums)
    def _functional_step(graphdef, state, x, y):
        (model_inner, opt_inner) = nnx.merge(graphdef, state)

        def loss_fn(m):
            # --- Padding ---
            pad = m._compute_padding(x.shape[-3:])
            x_padded = jnp.pad(x, (
                (0, 0), (0, 0),
                (pad[0], 0), (pad[1], 0), (pad[2], 0),
            ))

            # --- L0 encode ---
            h = m.encoder.down_blocks[0](
                x_padded, deterministic=False, use_running_average=False)
            skip0 = h
            h = m.encoder.down_pool[0](h)

            # --- L1 encode ---
            h = m.encoder.down_blocks[1](
                h, deterministic=False, use_running_average=False)
            skip1 = h
            h = m.encoder.down_pool[1](h)

            # --- L2+ via inner JIT (trace cached per L2+ family) ---
            # Create view from m so variables are at the correct trace level
            l2p = _L2PlusView(m)
            l2p_gd, l2p_st = nnx.split(l2p)
            # Use cached graphdef for trace reuse across L2+ family
            if l2p_cached_gd[0] is not None:
                l2p_gd = l2p_cached_gd[0]
            h_l2, ds_l2, l2p_new_st = _l2p_inner_train(
                l2p_gd, l2p_st, h)
            nnx.update(l2p, l2p_new_st)

            # --- L1 decode ---
            sf_l1 = m.decoder.upsample_scale_factors[-2]
            h = _trilinear_upsample_align_corners(h_l2, sf_l1)
            h = jnp.concatenate([h, skip1], axis=1)
            h = m.decoder.up_blocks[-2](
                h, deterministic=False, use_running_average=False)
            # L1 DS head
            ds_l1 = None
            if deep_supervision and len(m.decoder.ds_heads) > (n_downsample - 2):
                ds_l1 = m.decoder.ds_heads[n_downsample - 2](h)

            # --- L0 decode ---
            sf_l0 = m.decoder.upsample_scale_factors[-1]
            h = _trilinear_upsample_align_corners(h, sf_l0)
            h = jnp.concatenate([h, skip0], axis=1)
            h = m.decoder.up_blocks[-1](
                h, deterministic=False, use_running_average=False)

            # --- Output projection ---
            w_mat = _build_sc_weight_matrix_fast(
                m.out_weight[...],
                m._out_sc_src[...], m._out_sc_dst[...], m._out_sc_alpha[...],
                m._out_out_dim, m._out_in_dim,
            )
            h = jnp.moveaxis(h, 1, -1)
            h = h @ w_mat.astype(h.dtype).T
            h = jnp.moveaxis(h, -1, 1)

            bias = m.bias[...].astype(h.dtype).reshape(-1, 1, 1, 1)
            h = h + bias

            # Remove padding
            h = h[..., pad[0]:, pad[1]:, pad[2]:]

            if deep_supervision:
                # Assemble DS outputs: L2+ DS + L1 DS + finest
                final = []
                for i, ds_out in enumerate(ds_l2):
                    ds_out = ds_out + bias
                    sf = 2 ** (n_downsample - 1 - i)
                    sp = tuple(max(0, pad[d] // sf) for d in range(3))
                    ds_out = ds_out[..., sp[0]:, sp[1]:, sp[2]:]
                    final.append(ds_out)
                if ds_l1 is not None:
                    ds_l1 = ds_l1 + bias
                    sp = tuple(max(0, pad[d] // 2) for d in range(3))
                    ds_l1 = ds_l1[..., sp[0]:, sp[1]:, sp[2]:]
                    final.append(ds_l1)
                final.append(h)
                return deep_supervision_loss(final, y, n_classes=n_classes,
                                             no_background=no_background)
            return dice_ce_loss(h, y, n_classes,
                                no_background=no_background)

        loss, grads = nnx.value_and_grad(loss_fn)(model_inner)
        opt_inner.update(model_inner, grads)
        _, new_state = nnx.split((model_inner, opt_inner))
        return loss, new_state

    # Outer graphdef cache (same as monolithic)
    cached = [None, None]
    cached[0], _ = nnx.split((model, optimizer))
    cached[1] = (model.spacing, getattr(model, '_override_pool_key', None))

    def step(batch):
        # Update L2+ graphdef cache if family changed
        l2p_tmp = _L2PlusView(model)
        l2p_gd_now, _ = nnx.split(l2p_tmp)
        if l2p_gd_now != l2p_cached_gd[0]:
            l2p_cached_gd[0] = l2p_gd_now

        graphdef, state = nnx.split((model, optimizer))
        cache_key = (model.spacing, getattr(model, '_override_pool_key', None))
        if cache_key != cached[1]:
            cached[0] = graphdef
            cached[1] = cache_key
        loss, new_state = _functional_step(
            cached[0], state, batch['image'], batch['label']
        )
        nnx.update((model, optimizer), new_state)
        return loss

    step.jitted_fn = _functional_step

    return step


def create_jitted_val_step(model, n_classes, deep_supervision=False,
                           no_background=False):
    """JIT-compiled validation step that supports dynamic spacing.

    Mirrors create_jitted_train_step_dynamic but for inference only:
    no optimizer, no gradients, deterministic=True. Returns (loss, pred)
    where pred is argmax of the finest-resolution logits (compact int32).

    Parameters
    ----------
    model : E3nnUNet
        The model (Flax NNX module). Not mutated.
    n_classes : int
        Number of output classes for loss computation.
    deep_supervision : bool
        If True, model returns list of multi-scale outputs.
    no_background : bool
        If True, skip class 0 in dice computation.

    Returns
    -------
    callable
        A function ``val_step(image, label) -> (loss, pred)`` where
        pred is int32 argmax of finest logits.
    """
    @functools.partial(jax.jit, static_argnums=(0,))
    def _functional_val_step(graphdef, state, x, y):
        model_inner = nnx.merge(graphdef, state)
        logits = model_inner(x, deterministic=True, use_running_average=True)

        if deep_supervision and isinstance(logits, list):
            loss = deep_supervision_loss(logits, y, n_classes=n_classes,
                                         no_background=no_background)
            logits_finest = logits[-1]
        else:
            if isinstance(logits, list):
                logits_finest = logits[-1]
            else:
                logits_finest = logits
            loss = dice_ce_loss(logits_finest, y, n_classes,
                                no_background=no_background)

        pred = jnp.argmax(logits_finest, axis=1)
        return loss, pred

    # Cache graphdef keyed by (spacing, override_pool_key)
    cached = [None, None]  # [graphdef, cache_key]
    cached[0], _ = nnx.split(model)
    cached[1] = (model.spacing, getattr(model, '_override_pool_key', None))

    def val_step(image, label):
        graphdef, state = nnx.split(model)
        cache_key = (model.spacing, getattr(model, '_override_pool_key', None))
        if cache_key != cached[1]:
            cached[0] = graphdef
            cached[1] = cache_key
        return _functional_val_step(cached[0], state, image, label)

    val_step.jitted_fn = _functional_val_step
    return val_step


def _get_rss_mb():
    """Get current process RSS (Resident Set Size) in MB from /proc/self/status."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except (OSError, ValueError):
        return 0.0
    return 0.0


class BackgroundJITCompiler:
    """Compiles JIT traces for spacing groups in a background thread.

    Pre-snapshots graphdefs for all spacings at init (avoids model mutation
    races during training). Background thread compiles via .lower().compile()
    which populates the in-memory JIT dispatch cache shared with the main
    training thread.

    Usage::

        compiler = BackgroundJITCompiler(
            model, optimizer, step_fn, spacing_groups, use_fp16=True,
        )
        compiler.start()

        # In training loop:
        if compiler.is_ready(spacing):
            loss = step_fn(batch)
        else:
            compiler.prioritize(spacing)
            continue  # skip this batch

        compiler.shutdown()

    Parameters
    ----------
    model : E3nnUNet
        The model. Temporarily mutated via update_spacing() during init
        to snapshot graphdefs, then restored to original spacing.
    optimizer : nnx.Optimizer
        The optimizer (needed for graphdef/state splitting).
    step_fn : callable
        Return value of create_jitted_train_step_dynamic. Must have
        ``step_fn.jitted_fn`` attribute.
    spacing_groups : dict
        Maps spacing tuple -> dict with 'patch_size' (D,H,W) and
        'batch_size' (int).
    use_fp16 : bool
        Whether to use bfloat16 input dtype.
    cache_max : int
        Maximum number of compiled programs to keep in memory. If exceeded,
        stops compiling new ones (safety valve).
    """

    def __init__(self, model, optimizer, step_fn, spacing_groups,
                 use_fp16=True, cache_max=40, pool_kernel_overrides=None):
        self._functional_step = step_fn.jitted_fn
        self._spacing_groups = spacing_groups
        self._dtype = jnp.bfloat16 if use_fp16 else jnp.float32
        self._cache_max = cache_max
        self._pool_kernel_overrides = pool_kernel_overrides or {}

        self._ready = set()
        self._queue = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        self._compile_times = {}

        # Pre-snapshot graphdefs for all spacings (avoids model mutation races)
        self._graphdefs = {}
        self._state_abstracts = {}
        original = model.spacing

        for spacing in spacing_groups:
            overrides = self._pool_kernel_overrides.get(spacing)
            model.update_spacing(spacing, override_pool_kernels=overrides)
            gd, st = nnx.split((model, optimizer))
            st_abs = jax.tree.map(
                lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), st
            )
            del st
            self._graphdefs[spacing] = gd
            self._state_abstracts[spacing] = st_abs

        original_overrides = self._pool_kernel_overrides.get(original)
        model.update_spacing(original, override_pool_kernels=original_overrides)
        print(f"BackgroundJITCompiler: snapshotted {len(spacing_groups)} "
              f"graphdefs", flush=True)

    def start(self):
        """Pre-enqueue all spacings (most-sampled first) and start background thread."""
        # Sort by loader sampling weight (descending) so the spacings that
        # training encounters most often get compiled first.
        sorted_spacings = sorted(
            self._spacing_groups,
            key=lambda s: self._spacing_groups[s].get('weight', 0.0),
            reverse=True,
        )
        with self._lock:
            for spacing in sorted_spacings:
                self._queue.append(spacing)
        self._thread = threading.Thread(target=self._compile_loop, daemon=True)
        self._thread.start()
        self._wake.set()

    def is_ready(self, spacing):
        """Check if spacing is compiled (thread-safe, set membership is atomic)."""
        return spacing in self._ready

    def prioritize(self, spacing):
        """Move spacing to front of compile queue."""
        with self._lock:
            if spacing in self._queue:
                self._queue.remove(spacing)
            if spacing not in self._ready:
                self._queue.appendleft(spacing)
        self._wake.set()

    def shutdown(self):
        """Stop the background thread."""
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def n_ready(self):
        return len(self._ready)

    @property
    def n_queued(self):
        with self._lock:
            return len(self._queue)

    def _compile_loop(self):
        """Background thread: compile spacings from queue."""
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()

            while not self._stop.is_set():
                with self._lock:
                    if not self._queue:
                        break
                    spacing = self._queue.popleft()

                if spacing in self._ready:
                    continue

                # Safety valve: stop if too many compiled programs
                if len(self._ready) >= self._cache_max:
                    print(f"  [BG compile] Cache limit reached "
                          f"({self._cache_max}), pausing", flush=True)
                    break

                t0 = time.time()
                try:
                    self._compile_one(spacing)
                    dt = time.time() - t0
                    self._ready.add(spacing)
                    self._compile_times[spacing] = dt
                    rss = _get_rss_mb()
                    print(f"  [BG compile] {spacing} done in {dt:.1f}s "
                          f"({len(self._ready)}/{len(self._spacing_groups)} "
                          f"ready, RSS: {rss:.0f} MB)", flush=True)
                except Exception as e:
                    print(f"  [BG compile] {spacing} FAILED: {e}",
                          flush=True)

    def _compile_one(self, spacing):
        """AOT-compile _functional_step for one spacing group."""
        info = self._spacing_groups[spacing]
        patch = info['patch_size']
        bs = info['batch_size']

        graphdef = self._graphdefs[spacing]
        state_abs = self._state_abstracts[spacing]
        x_abs = jax.ShapeDtypeStruct((bs, 1, *patch), self._dtype)
        y_abs = jax.ShapeDtypeStruct((bs, *patch), jnp.int32)

        self._functional_step.lower(
            graphdef, state_abs, x_abs, y_abs
        ).compile()


def accumulated_train_step(model, optimizer, batches, n_classes,
                           no_background=False):
    """Accumulate gradients over multiple micro-batches.

    Parameters
    ----------
    model : E3nnUNet
    optimizer : nnx.Optimizer
    batches : list of dict
        Each dict has 'image' and 'label' keys.
    n_classes : int

    Returns
    -------
    jax.Array
        Mean loss across micro-batches.
    """
    def loss_fn(model, x, y):
        logits = model(x, deterministic=False, use_running_average=False)
        if isinstance(logits, list):
            return deep_supervision_loss(logits, y, n_classes=n_classes,
                                         no_background=no_background)
        return dice_ce_loss(logits, y, n_classes,
                            no_background=no_background)

    total_loss = jnp.zeros(())
    acc_grads = None
    for batch in batches:
        loss, grads = nnx.value_and_grad(loss_fn)(
            model, batch['image'], batch['label']
        )
        total_loss = total_loss + loss
        if acc_grads is None:
            acc_grads = grads
        else:
            acc_grads = jax.tree.map(jnp.add, acc_grads, grads)

    n = len(batches)
    acc_grads = jax.tree.map(lambda g: g / n, acc_grads)
    optimizer.update(model, acc_grads)
    return total_loss / n


def offload_optimizer_state_to_cpu(optimizer):
    """Move optimizer momentum/variance state to CPU RAM.

    Adam stores 2 extra copies of all parameters (mu and nu).
    Moving these to CPU between training steps frees GPU memory.
    They are automatically moved back to GPU when optimizer.update()
    is called (JAX handles cross-device transfers transparently).

    Parameters
    ----------
    optimizer : nnx.Optimizer
        The optimizer whose internal state to offload.

    Returns
    -------
    nnx.Optimizer
        The same optimizer with state on CPU.
    """
    cpu = jax.devices('cpu')[0]
    opt_state = optimizer.opt_state
    cpu_state = jax.tree.map(
        lambda x: jax.device_put(x, cpu) if isinstance(x, jax.Array) else x,
        opt_state,
    )
    optimizer.opt_state = cpu_state
    return optimizer


def restore_optimizer_state_to_gpu(optimizer, device=None):
    """Move optimizer state back to GPU for the update step.

    Parameters
    ----------
    optimizer : nnx.Optimizer
        The optimizer whose state to move to GPU.
    device : jax.Device or None
        Target GPU device. If None, uses default device.

    Returns
    -------
    nnx.Optimizer
        The same optimizer with state on GPU.
    """
    if device is None:
        device = jax.devices()[0]
    opt_state = optimizer.opt_state
    gpu_state = jax.tree.map(
        lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x,
        opt_state,
    )
    optimizer.opt_state = gpu_state
    return optimizer
