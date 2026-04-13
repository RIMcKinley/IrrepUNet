"""3-way split JIT compilation for E3nnUNet.

Splits the model into L0 | L1 | L2+ pieces for independent JIT compilation.
This enables sharing compiled XLA programs across spacing groups with identical
deeper architectures (same family key), reducing unique compilations.

Split structure:
  L0  = enc_block[0] + pool[0] + dec_block[-1] + output_head
  L1  = enc_block[1] + pool[1] + dec_block[-2]
  L2+ = enc_blocks[2..N] + pools[2..N-1] + dec_blocks[0..N-3] + DS heads

Reuse:
  L0:  one trace per spacing (spacing-dependent kernels at finest resolution)
  L1:  one trace per L1 family (shared across spacings with same L1 step)
  L2+: one trace per L2+ family (shared across spacings with same L2+ step)

Usage:
  Training: monolithic JIT (value_and_grad needs single XLA program)
  Validation: 5 sequential split JIT calls (no gradients)
  Background AOT: compile split pieces first (fewer unique programs),
                   then monolithic for training readiness
"""

import functools
import math
import threading
import time
from collections import defaultdict, deque

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

from irrepunet.models_jax import E3nnUNet
from irrepunet.models_jax.layers import (
    Buffer, _build_sc_weight_matrix_fast, _trilinear_upsample_align_corners,
)
from irrepunet.models_jax.train import (
    dice_ce_loss, deep_supervision_loss, _get_rss_mb,
)


# =============================================================================
# Family Registry
# =============================================================================

class FamilyRegistry:
    """Compute and store L1/L2+ family keys for all spacing groups.

    Family key = effective spacing tuple at that level (from _compute_steps_array).
    Same key -> identical pool kernels, conv kernels, feature shapes at that
    level and below -> same graphdef -> same XLA program.

    Parameters
    ----------
    spacing_groups : dict
        Maps spacing tuple -> group info dict (with 'patch_size', 'batch_size',
        'weight' keys).
    model_scale : float
        Model scale parameter (typically 2.0).
    n_downsample : int
        Number of pooling levels.
    pool_kernel_overrides : dict or None
        Maps spacing tuple -> dict of {level_index: kernel_tuple}.
        Used by targeted pooling to consolidate L2+ families.
    """

    def __init__(self, spacing_groups, model_scale, n_downsample,
                 pool_kernel_overrides=None):
        self._spacings = list(spacing_groups.keys())
        self._model_scale = model_scale
        self._n_downsample = n_downsample
        self._pool_kernel_overrides = pool_kernel_overrides or {}

        scales = [model_scale * (2 ** i) for i in range(n_downsample)]

        # Per-spacing keys
        self._l1_key = {}   # spacing -> l1 family key tuple
        self._l2_key = {}   # spacing -> l2+ family key tuple

        # Family -> member spacings
        self._l1_families = defaultdict(list)
        self._l2_families = defaultdict(list)

        for spacing in self._spacings:
            overrides = self._pool_kernel_overrides.get(spacing)
            steps = E3nnUNet._compute_steps_array(
                spacing, scales, n_downsample,
                override_pool_kernels=overrides,
            )

            # L1 key = effective spacing at level 1
            l1_key = tuple(float(s) for s in steps[1])
            self._l1_key[spacing] = l1_key
            self._l1_families[l1_key].append(spacing)

            # L2+ key = effective spacing at level 2
            if n_downsample >= 3:
                l2_key = tuple(float(s) for s in steps[2])
            else:
                # With only 2 levels, L2+ is the bottleneck = level 2 step
                l2_key = tuple(float(s) for s in steps[min(2, len(steps) - 1)])
            self._l2_key[spacing] = l2_key
            self._l2_families[l2_key].append(spacing)

    def l1_key(self, spacing):
        """Get L1 family key for a spacing."""
        return self._l1_key[spacing]

    def l2_key(self, spacing):
        """Get L2+ family key for a spacing."""
        return self._l2_key[spacing]

    @property
    def l1_families(self):
        """Dict mapping L1 family key -> list of spacings."""
        return dict(self._l1_families)

    @property
    def l2_families(self):
        """Dict mapping L2+ family key -> list of spacings."""
        return dict(self._l2_families)

    @property
    def n_l0(self):
        """Number of unique L0 programs (one per spacing)."""
        return len(self._spacings)

    @property
    def n_l1(self):
        """Number of unique L1 programs."""
        return len(self._l1_families)

    @property
    def n_l2(self):
        """Number of unique L2+ programs."""
        return len(self._l2_families)

    def l1_representative(self, l1_key):
        """Get first spacing in a L1 family (the one that cold-compiles)."""
        return self._l1_families[l1_key][0]

    def l2_representative(self, l2_key):
        """Get first spacing in a L2+ family (the one that cold-compiles)."""
        return self._l2_families[l2_key][0]

    def summary(self):
        """Print family summary."""
        print(f"FamilyRegistry: {self.n_l0} L0, {self.n_l1} L1, {self.n_l2} L2+ "
              f"unique programs (from {len(self._spacings)} spacing groups)")
        for l1_key, members in sorted(self._l1_families.items()):
            print(f"  L1 {l1_key}: {len(members)} spacings")
        for l2_key, members in sorted(self._l2_families.items()):
            print(f"  L2+ {l2_key}: {len(members)} spacings")


# =============================================================================
# Wrapper Modules
# =============================================================================

class L0Wrapper(nnx.Module):
    """L0: enc_block[0], pool[0], dec_block[-1], output head."""

    def __init__(self, model):
        self.enc_block = model.encoder.down_blocks[0]
        self.enc_pool = model.encoder.down_pool[0]
        self.dec_block = model.decoder.up_blocks[-1]
        self.upsample_sf = model.decoder.upsample_scale_factors[-1]

        self.out_weight = model.out_weight
        self.bias = model.bias
        self._out_in_dim = model._out_in_dim
        self._out_out_dim = model._out_out_dim
        self._has_out_sc = model._has_out_sc
        if model._has_out_sc:
            self._out_sc_src = model._out_sc_src
            self._out_sc_dst = model._out_sc_dst
            self._out_sc_alpha = model._out_sc_alpha

        self.n_downsample = model.n_downsample
        self.deep_supervision = model.deep_supervision

    def encode(self, x_padded):
        x = self.enc_block(x_padded, deterministic=True, use_running_average=True)
        skip0 = x
        x = self.enc_pool(x)
        return x, skip0

    def decode(self, x_from_l1, skip0, pad, ds_outputs):
        x = _trilinear_upsample_align_corners(x_from_l1, self.upsample_sf)
        x = jnp.concatenate([x, skip0], axis=1)
        x = self.dec_block(x, deterministic=True, use_running_average=True)

        w_mat = _build_sc_weight_matrix_fast(
            self.out_weight[...],
            self._out_sc_src[...], self._out_sc_dst[...],
            self._out_sc_alpha[...],
            self._out_out_dim, self._out_in_dim,
        )
        x = jnp.moveaxis(x, 1, -1)
        x = x @ w_mat.astype(x.dtype).T
        x = jnp.moveaxis(x, -1, 1)

        bias = self.bias[...].astype(x.dtype).reshape(-1, 1, 1, 1)
        x = x + bias
        x = x[..., pad[0]:, pad[1]:, pad[2]:]

        if self.deep_supervision and ds_outputs is not None:
            final = []
            for i, ds_out in enumerate(ds_outputs):
                ds_out = ds_out + bias
                sf = 2 ** (self.n_downsample - 1 - i)
                sp = tuple(max(0, p // sf) for p in pad)
                ds_out = ds_out[..., sp[0]:, sp[1]:, sp[2]:]
                final.append(ds_out)
            final.append(x)
            return final
        return x


class L1Wrapper(nnx.Module):
    """L1: enc_block[1], pool[1], dec_block[-2], ds_head for L1 level."""

    def __init__(self, model):
        self.enc_block = model.encoder.down_blocks[1]
        self.enc_pool = model.encoder.down_pool[1]
        self.dec_block = model.decoder.up_blocks[-2]
        self.upsample_sf = model.decoder.upsample_scale_factors[-2]

        # DS head for L1 decode level (decoder index n_downsample - 2)
        n = model.n_downsample
        self.deep_supervision = model.deep_supervision
        ds_head_idx = n - 2  # up_blocks[-2] corresponds to ds_heads[n-2]
        if (model.deep_supervision
                and len(model.decoder.ds_heads) > ds_head_idx):
            self.ds_head = model.decoder.ds_heads[ds_head_idx]
            self._has_ds_head = True
        else:
            self._has_ds_head = False

    def encode(self, x_l1):
        x = self.enc_block(x_l1, deterministic=True, use_running_average=True)
        skip1 = x
        x = self.enc_pool(x)
        return x, skip1

    def decode(self, x_from_inner, skip1):
        x = _trilinear_upsample_align_corners(x_from_inner, self.upsample_sf)
        x = jnp.concatenate([x, skip1], axis=1)
        x = self.dec_block(x, deterministic=True, use_running_average=True)
        if self.deep_supervision and self._has_ds_head:
            ds_out = self.ds_head(x)
            return x, ds_out
        return x, None


class L2PlusWrapper(nnx.Module):
    """L2+: enc_blocks[2..N], pools[2..N-1], dec_blocks[0..N-3], DS heads."""

    def __init__(self, model):
        n = model.n_downsample
        self.enc_blocks = nnx.List(list(model.encoder.down_blocks)[2:])
        self.enc_pools = nnx.List(list(model.encoder.down_pool)[2:])
        self.dec_blocks = nnx.List(list(model.decoder.up_blocks)[:-2])
        # Only take the DS heads for L2+ decoder levels (first n-2 of n-1 total)
        n_l2_ds = n - 2
        if model.decoder.deep_supervision and len(model.decoder.ds_heads) > 0:
            self.ds_heads = nnx.List(list(model.decoder.ds_heads)[:n_l2_ds])
        else:
            self.ds_heads = nnx.List([])
        self.deep_supervision = model.decoder.deep_supervision
        self.n_dec_blocks = n - 2
        self.upsample_sfs = model.decoder.upsample_scale_factors[:-2]

    def __call__(self, x_l2):
        features = []
        x = x_l2
        for i, block in enumerate(self.enc_blocks):
            x = block(x, deterministic=True, use_running_average=True)
            features.append(x)
            if i < len(self.enc_blocks) - 1:
                x = self.enc_pools[i](x)

        x = features[-1]
        ds_outputs = []
        for i in range(self.n_dec_blocks):
            sf = self.upsample_sfs[i]
            x = _trilinear_upsample_align_corners(x, sf)
            skip = features[::-1][i + 1]
            x = jnp.concatenate([x, skip], axis=1)
            x = self.dec_blocks[i](x, deterministic=True, use_running_average=True)

            if self.deep_supervision and len(self.ds_heads) > 0:
                if i < len(self.ds_heads):
                    ds_outputs.append(self.ds_heads[i](x))

        if self.deep_supervision and len(ds_outputs) > 0:
            return x, ds_outputs
        return x, None


def update_wrappers(model, l0, l1, l2p):
    """Sync spacing-dependent upsample scale factors from model to wrappers."""
    l0.upsample_sf = model.decoder.upsample_scale_factors[-1]
    l1.upsample_sf = model.decoder.upsample_scale_factors[-2]
    l2p.upsample_sfs = model.decoder.upsample_scale_factors[:-2]


# =============================================================================
# JIT Functions
# =============================================================================

@jax.jit
def _l0_enc_jit(graphdef, state, x_padded):
    """JIT-compiled L0 encode: enc_block[0] + pool[0]."""
    w = nnx.merge(graphdef, state)
    return w.encode(x_padded)


@jax.jit
def _l1_enc_jit(graphdef, state, x_l1):
    """JIT-compiled L1 encode: enc_block[1] + pool[1]."""
    w = nnx.merge(graphdef, state)
    return w.encode(x_l1)


@jax.jit
def _l2p_fwd_jit(graphdef, state, x_l2):
    """JIT-compiled L2+ forward: enc_blocks[2..N] + dec_blocks[0..N-3] + DS heads."""
    w = nnx.merge(graphdef, state)
    return w(x_l2)


@jax.jit
def _l1_dec_jit(graphdef, state, x_from_inner, skip1):
    """JIT-compiled L1 decode: dec_block[-2] + optional DS head."""
    w = nnx.merge(graphdef, state)
    return w.decode(x_from_inner, skip1)


@functools.partial(jax.jit, static_argnums=(4,))
def _l0_dec_jit(graphdef, state, x_from_l1, skip0, pad, ds_outputs):
    """JIT-compiled L0 decode: dec_block[-1] + output_head + unpad.

    pad is static_argnums=(4,) because slice operations need static integers.
    Must be a tuple (hashable).
    """
    w = nnx.merge(graphdef, state)
    return w.decode(x_from_l1, skip0, pad, ds_outputs)


# =============================================================================
# SplitJITManager
# =============================================================================

class SplitJITManager:
    """Manages split and monolithic JIT compilations for all spacing groups.

    Replaces BackgroundJITCompiler. Compiles split pieces (fewer unique programs)
    first for fast validation readiness, then monolithic programs for training.

    Parameters
    ----------
    model : E3nnUNet
        The model. Temporarily mutated via update_spacing() during init.
    optimizer : nnx.Optimizer
        The optimizer (needed for monolithic graphdef/state splitting).
    step_fn : callable
        Return value of create_jitted_train_step_dynamic. Must have
        ``step_fn.jitted_fn`` attribute.
    spacing_groups : dict
        Maps spacing tuple -> dict with 'patch_size', 'batch_size', 'weight'.
    use_fp16 : bool
        Whether to use bfloat16 input dtype.
    cache_max : int
        Maximum number of compiled monolithic programs in memory.
    """

    def __init__(self, model, optimizer, step_fn, spacing_groups,
                 use_fp16=True, cache_max=40, pool_kernel_overrides=None):
        self._functional_step = step_fn.jitted_fn
        self._spacing_groups = spacing_groups
        self._dtype = jnp.bfloat16 if use_fp16 else jnp.float32
        self._cache_max = cache_max
        self._pool_kernel_overrides = pool_kernel_overrides or {}

        # Build family registry
        self._registry = FamilyRegistry(
            spacing_groups, model.scale, model.n_downsample,
            pool_kernel_overrides=pool_kernel_overrides,
        )
        self._registry.summary()

        # Create wrappers (reference model submodules, not copies)
        self._l0 = L0Wrapper(model)
        self._l1 = L1Wrapper(model)
        self._l2p = L2PlusWrapper(model)

        # Read channel dimensions from model irreps (fixed by architecture)
        self._enc_channels = []
        for block in model.encoder.down_blocks:
            from irrepunet.models_jax.layers import _irreps_dim
            self._enc_channels.append(_irreps_dim(block.irreps_out))

        # Threading state
        self._ready_mono = set()       # monolithic programs ready
        self._ready_l0 = set()         # L0 programs ready (keyed by spacing)
        self._ready_l1 = set()         # L1 programs ready (keyed by l1_key)
        self._ready_l2 = set()         # L2+ programs ready (keyed by l2_key)
        self._queue = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        self._compile_times = {}

        # Pre-snapshot graphdefs for all spacings
        self._mono_graphdefs = {}
        self._mono_state_abstracts = {}
        self._l0_graphdefs = {}
        self._l0_state_abstracts = {}
        self._l1_graphdefs = {}      # keyed by l1_key
        self._l1_state_abstracts = {}
        self._l2_graphdefs = {}      # keyed by l2_key
        self._l2_state_abstracts = {}

        original = model.spacing

        # Track which family keys we've already snapshotted
        l1_snapshotted = set()
        l2_snapshotted = set()

        for spacing in spacing_groups:
            overrides = self._pool_kernel_overrides.get(spacing)
            model.update_spacing(spacing, override_pool_kernels=overrides)
            update_wrappers(model, self._l0, self._l1, self._l2p)

            # Monolithic graphdef
            gd, st = nnx.split((model, optimizer))
            st_abs = jax.tree.map(
                lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), st
            )
            del st
            self._mono_graphdefs[spacing] = gd
            self._mono_state_abstracts[spacing] = st_abs

            # L0 graphdef (one per spacing)
            gd_l0, st_l0 = nnx.split(self._l0)
            self._l0_graphdefs[spacing] = gd_l0
            self._l0_state_abstracts[spacing] = jax.tree.map(
                lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), st_l0
            )
            del st_l0

            # L1 graphdef (one per L1 family)
            l1_key = self._registry.l1_key(spacing)
            if l1_key not in l1_snapshotted:
                gd_l1, st_l1 = nnx.split(self._l1)
                self._l1_graphdefs[l1_key] = gd_l1
                self._l1_state_abstracts[l1_key] = jax.tree.map(
                    lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), st_l1
                )
                del st_l1
                l1_snapshotted.add(l1_key)

            # L2+ graphdef (one per L2+ family)
            l2_key = self._registry.l2_key(spacing)
            if l2_key not in l2_snapshotted:
                gd_l2, st_l2 = nnx.split(self._l2p)
                self._l2_graphdefs[l2_key] = gd_l2
                self._l2_state_abstracts[l2_key] = jax.tree.map(
                    lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), st_l2
                )
                del st_l2
                l2_snapshotted.add(l2_key)

        original_overrides = self._pool_kernel_overrides.get(original)
        model.update_spacing(original, override_pool_kernels=original_overrides)
        update_wrappers(model, self._l0, self._l1, self._l2p)

        print(f"SplitJITManager: snapshotted {len(spacing_groups)} monolithic, "
              f"{len(self._l0_graphdefs)} L0, "
              f"{len(self._l1_graphdefs)} L1, "
              f"{len(self._l2_graphdefs)} L2+ graphdefs", flush=True)

    @property
    def registry(self):
        return self._registry

    @property
    def l0_wrapper(self):
        return self._l0

    @property
    def l1_wrapper(self):
        return self._l1

    @property
    def l2p_wrapper(self):
        return self._l2p

    def warmup(self, n=None):
        """Synchronously compile the top-N most-sampled monolithic programs.

        Blocks until done. Call before start() so that the most common
        spacing groups are ready when training begins, avoiding skips.

        Parameters
        ----------
        n : int or None
            Number of top groups to compile. If None, compiles all.
        """
        sorted_mono = sorted(
            self._spacing_groups.keys(),
            key=lambda s: self._spacing_groups[s].get('weight', 0.0),
            reverse=True,
        )
        if n is not None:
            sorted_mono = sorted_mono[:n]

        print(f"Warmup: compiling {len(sorted_mono)} monolithic programs "
              f"(top by sampling weight)...", flush=True)
        t_total = time.time()
        for spacing in sorted_mono:
            if spacing in self._ready_mono:
                continue
            if len(self._ready_mono) >= self._cache_max:
                print(f"  Warmup: hit cache limit ({self._cache_max}), stopping", flush=True)
                break
            t0 = time.time()
            try:
                self._compile_mono(spacing)
                self._ready_mono.add(spacing)
                dt = time.time() - t0
                self._compile_times[('mono', spacing)] = dt
                rss = _get_rss_mb()
                print(f"  [warmup] mono {spacing} done in {dt:.1f}s "
                      f"({len(self._ready_mono)}/{len(sorted_mono)}, "
                      f"RSS: {rss:.0f} MB)", flush=True)
            except Exception as e:
                print(f"  [warmup] mono {spacing} FAILED: {e}", flush=True)

        dt_total = time.time() - t_total
        print(f"Warmup complete: {len(self._ready_mono)} programs in {dt_total:.1f}s",
              flush=True)

    def start(self):
        """Enqueue remaining monolithic compilations and start background thread.

        Only monolithic programs are compiled in the background (training
        needs them). Split programs (L0/L1/L2+) compile lazily on first
        call during validation — they're fast (1-10s each) and don't
        compete for GPU memory with the monolithic cache.

        Within the monolithic tier, sort by sampling weight descending.
        Programs already compiled by warmup() are skipped automatically.
        """
        sorted_mono = sorted(
            self._spacing_groups.keys(),
            key=lambda s: self._spacing_groups[s].get('weight', 0.0),
            reverse=True,
        )

        with self._lock:
            for spacing in sorted_mono:
                if spacing not in self._ready_mono:
                    self._queue.append(('mono', spacing))

        self._thread = threading.Thread(target=self._compile_loop, daemon=True)
        self._thread.start()
        self._wake.set()

    def is_ready(self, spacing):
        """Check if monolithic program is compiled (for training step)."""
        return spacing in self._ready_mono

    def is_split_ready(self, spacing):
        """Check if all split pieces are compiled (for validation)."""
        l1_key = self._registry.l1_key(spacing)
        l2_key = self._registry.l2_key(spacing)
        return (spacing in self._ready_l0
                and l1_key in self._ready_l1
                and l2_key in self._ready_l2)

    def prioritize(self, spacing):
        """Move monolithic program for this spacing to front of queue.

        Only promotes the monolithic program — split pieces stay in their
        natural queue position to avoid GPU memory pressure from accumulating
        too many cached programs during training warmup.
        """
        with self._lock:
            # Remove existing monolithic entry for this spacing
            new_queue = deque()
            for item in self._queue:
                kind, key = item
                if kind == 'mono' and key == spacing:
                    continue  # will re-add at front
                new_queue.append(item)

            # Re-add monolithic at front
            if spacing not in self._ready_mono:
                new_queue.appendleft(('mono', spacing))

            self._queue = new_queue

        self._wake.set()

    def shutdown(self):
        """Stop the background thread."""
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def n_ready(self):
        """Number of monolithic programs ready (for compatibility)."""
        return len(self._ready_mono)

    @property
    def n_queued(self):
        with self._lock:
            return len(self._queue)

    @property
    def n_split_ready(self):
        """Number of spacings with all split pieces ready."""
        count = 0
        for spacing in self._spacing_groups:
            if self.is_split_ready(spacing):
                count += 1
        return count

    def _compile_loop(self):
        """Background thread: compile programs from queue."""
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()

            while not self._stop.is_set():
                with self._lock:
                    if not self._queue:
                        break
                    kind, key = self._queue.popleft()

                # Check if already done
                if kind == 'mono' and key in self._ready_mono:
                    continue
                if kind == 'l0' and key in self._ready_l0:
                    continue
                if kind == 'l1' and key in self._ready_l1:
                    continue
                if kind == 'l2' and key in self._ready_l2:
                    continue

                # Safety valve for monolithic programs
                if kind == 'mono' and len(self._ready_mono) >= self._cache_max:
                    print(f"  [BG split] Monolithic cache limit ({self._cache_max}), "
                          f"pausing monolithic compiles", flush=True)
                    continue

                t0 = time.time()
                try:
                    if kind == 'l2':
                        self._compile_l2(key)
                        self._ready_l2.add(key)
                    elif kind == 'l1':
                        self._compile_l1(key)
                        self._ready_l1.add(key)
                    elif kind == 'l0':
                        self._compile_l0(key)
                        self._ready_l0.add(key)
                    elif kind == 'mono':
                        self._compile_mono(key)
                        self._ready_mono.add(key)

                    dt = time.time() - t0
                    self._compile_times[(kind, key)] = dt
                    rss = _get_rss_mb()
                    ready_str = (f"L0={len(self._ready_l0)} "
                                 f"L1={len(self._ready_l1)} "
                                 f"L2+={len(self._ready_l2)} "
                                 f"mono={len(self._ready_mono)}")
                    print(f"  [BG split] {kind} {key} done in {dt:.1f}s "
                          f"({ready_str}, RSS: {rss:.0f} MB)", flush=True)
                except Exception as e:
                    print(f"  [BG split] {kind} {key} FAILED: {e}", flush=True)

    def _compute_intermediate_shapes(self, spacing):
        """Compute spatial shapes at each level for a given spacing.

        Returns dict with keys: padded_shape, l1_shape, l2_shape, plus
        channel counts at each level.
        """
        info = self._spacing_groups[spacing]
        patch = info['patch_size']
        bs = info['batch_size']

        scales = [self._registry._model_scale * (2 ** i)
                  for i in range(self._registry._n_downsample)]
        overrides = self._pool_kernel_overrides.get(spacing)
        steps = E3nnUNet._compute_steps_array(
            spacing, scales, self._registry._n_downsample,
            override_pool_kernels=overrides,
        )

        # Compute pool kernels at each level
        pool_kernels = []
        for level in range(self._registry._n_downsample):
            if overrides and level in overrides:
                ks = tuple(overrides[level])
            else:
                step = steps[level]
                ks = tuple(
                    math.floor(scales[level] / s) if s < scales[level] else 1
                    for s in step
                )
            pool_kernels.append(ks)

        # Compute padding
        pooling_factor = np.ones(3, dtype='int')
        for ks in pool_kernels:
            pooling_factor *= np.array(ks)
        pad = []
        for f, s in zip(pooling_factor, patch):
            t = s % f
            p = (f - t) if t != 0 else 0
            pad.append(p)

        # Spatial shapes after padding and each pool
        padded_spatial = tuple(p + pad_v for p, pad_v in zip(patch, pad))
        l1_spatial = tuple(s // k for s, k in zip(padded_spatial, pool_kernels[0]))
        if self._registry._n_downsample >= 2:
            l2_spatial = tuple(s // k for s, k in zip(l1_spatial, pool_kernels[1]))
        else:
            l2_spatial = l1_spatial

        return {
            'batch_size': bs,
            'pad': tuple(pad),
            'padded_spatial': padded_spatial,
            'l1_spatial': l1_spatial,
            'l2_spatial': l2_spatial,
            'pool_kernels': pool_kernels,
        }

    def _compile_l0(self, spacing):
        """AOT-compile L0 encode + decode for one spacing."""
        shapes = self._compute_intermediate_shapes(spacing)
        bs = shapes['batch_size']
        padded = shapes['padded_spatial']

        graphdef = self._l0_graphdefs[spacing]
        state_abs = self._l0_state_abstracts[spacing]

        # L0 encode: input is (B, 1, *padded_spatial)
        x_abs = jax.ShapeDtypeStruct((bs, 1, *padded), self._dtype)
        _l0_enc_jit.lower(graphdef, state_abs, x_abs).compile()

        # L0 decode: x_from_l1 is at l1_spatial (upsampled inside decode),
        # skip0 is at padded_spatial
        l1_sp = shapes['l1_spatial']
        l0_ch = self._enc_channels[0]
        from irrepunet.models_jax.layers import _irreps_dim
        l1_dec_ch = _irreps_dim(self._l1.dec_block.irreps_out)

        x_from_l1_abs = jax.ShapeDtypeStruct(
            (bs, l1_dec_ch, *l1_sp), self._dtype
        )
        skip0_abs = jax.ShapeDtypeStruct(
            (bs, l0_ch, *padded), self._dtype
        )
        pad_tuple = shapes['pad']
        # ds_outputs=None for AOT compilation
        _l0_dec_jit.lower(
            graphdef, state_abs, x_from_l1_abs, skip0_abs, pad_tuple, None
        ).compile()

    def _compile_l1(self, l1_key):
        """AOT-compile L1 encode + decode for one L1 family."""
        rep = self._registry.l1_representative(l1_key)
        shapes = self._compute_intermediate_shapes(rep)
        bs = shapes['batch_size']
        l1_sp = shapes['l1_spatial']
        l2_sp = shapes['l2_spatial']

        graphdef = self._l1_graphdefs[l1_key]
        state_abs = self._l1_state_abstracts[l1_key]

        # L1 encode: input has L0 encoder output channels
        l0_ch = self._enc_channels[0]
        x_abs = jax.ShapeDtypeStruct((bs, l0_ch, *l1_sp), self._dtype)
        _l1_enc_jit.lower(graphdef, state_abs, x_abs).compile()

        # L1 decode: x_from_inner + skip1
        l1_ch = self._enc_channels[1]

        # x_from_inner channels: L2+ wrapper's last decoder block output
        # = dec_blocks[0..N-3][-1].irreps_out if n_dec_blocks > 0
        # else enc_blocks[2..N][-1].irreps_out (bottleneck)
        from irrepunet.models_jax.layers import _irreps_dim
        if self._l2p.n_dec_blocks > 0:
            l2p_out_ch = _irreps_dim(self._l2p.dec_blocks[-1].irreps_out)
        else:
            l2p_out_ch = _irreps_dim(self._l2p.enc_blocks[-1].irreps_out)

        x_inner_abs = jax.ShapeDtypeStruct(
            (bs, l2p_out_ch, *l2_sp), self._dtype
        )
        skip1_abs = jax.ShapeDtypeStruct(
            (bs, l1_ch, *l1_sp), self._dtype
        )
        _l1_dec_jit.lower(graphdef, state_abs, x_inner_abs, skip1_abs).compile()

    def _compile_l2(self, l2_key):
        """AOT-compile L2+ forward for one L2+ family."""
        rep = self._registry.l2_representative(l2_key)
        shapes = self._compute_intermediate_shapes(rep)
        bs = shapes['batch_size']
        l2_sp = shapes['l2_spatial']

        graphdef = self._l2_graphdefs[l2_key]
        state_abs = self._l2_state_abstracts[l2_key]

        # L2+ input has L1 encoder output channels
        l1_ch = self._enc_channels[1]
        x_abs = jax.ShapeDtypeStruct((bs, l1_ch, *l2_sp), self._dtype)
        _l2p_fwd_jit.lower(graphdef, state_abs, x_abs).compile()

    def _compile_mono(self, spacing):
        """AOT-compile monolithic train step for one spacing."""
        info = self._spacing_groups[spacing]
        patch = info['patch_size']
        bs = info['batch_size']

        graphdef = self._mono_graphdefs[spacing]
        state_abs = self._mono_state_abstracts[spacing]
        x_abs = jax.ShapeDtypeStruct((bs, 1, *patch), self._dtype)
        y_abs = jax.ShapeDtypeStruct((bs, *patch), jnp.int32)

        self._functional_step.lower(
            graphdef, state_abs, x_abs, y_abs
        ).compile()


# =============================================================================
# Split Validation Step
# =============================================================================

def create_split_val_step(model, manager, n_classes, deep_supervision=False,
                          no_background=False):
    """Create a split validation step with hot-swap caching.

    Returns a callable that runs 5 sequential JIT calls (L0 enc -> L1 enc ->
    L2+ fwd -> L1 dec -> L0 dec) and computes loss + argmax prediction.

    Hot-swap: tracks current spacing/family keys, only re-splits wrappers
    when the corresponding family key changes.

    Parameters
    ----------
    model : E3nnUNet
        The model (wrappers reference its submodules).
    manager : SplitJITManager
        The manager holding wrappers and registry.
    n_classes : int
    deep_supervision : bool
    no_background : bool

    Returns
    -------
    callable
        ``split_val_step(image, label) -> (loss, pred)``
    """
    registry = manager.registry
    l0 = manager.l0_wrapper
    l1 = manager.l1_wrapper
    l2p = manager.l2p_wrapper

    # Hot-swap cache: track current keys
    cached_spacing = [None]
    cached_l1_key = [None]
    cached_l2_key = [None]
    cached_l0_gd = [None]
    cached_l1_gd = [None]
    cached_l2_gd = [None]

    def split_val_step(image, label):
        spacing = model.spacing
        l1_key = registry.l1_key(spacing)
        l2_key = registry.l2_key(spacing)

        # Hot-swap: only re-split wrappers whose family key changed
        if spacing != cached_spacing[0]:
            update_wrappers(model, l0, l1, l2p)
            gd_l0, _ = nnx.split(l0)
            cached_l0_gd[0] = gd_l0
            cached_spacing[0] = spacing

        if l1_key != cached_l1_key[0]:
            gd_l1, _ = nnx.split(l1)
            cached_l1_gd[0] = gd_l1
            cached_l1_key[0] = l1_key

        if l2_key != cached_l2_key[0]:
            gd_l2, _ = nnx.split(l2p)
            cached_l2_gd[0] = gd_l2
            cached_l2_key[0] = l2_key

        # Always get fresh state (weights may have been updated by training)
        _, st_l0 = nnx.split(l0)
        _, st_l1 = nnx.split(l1)
        _, st_l2 = nnx.split(l2p)

        # Compute padding
        pad = model._compute_padding(image.shape[-3:])
        x_padded = jnp.pad(image, (
            (0, 0), (0, 0),
            (pad[0], 0), (pad[1], 0), (pad[2], 0),
        ))

        # Forward pass: 5 sequential JIT calls
        x_l1, skip0 = _l0_enc_jit(cached_l0_gd[0], st_l0, x_padded)
        x_l2, skip1 = _l1_enc_jit(cached_l1_gd[0], st_l1, x_l1)
        x_for_l1, l2_ds = _l2p_fwd_jit(cached_l2_gd[0], st_l2, x_l2)
        x_for_l0, l1_ds = _l1_dec_jit(cached_l1_gd[0], st_l1, x_for_l1, skip1)

        # Combine DS outputs: L2+ DS heads + L1 DS head
        if l2_ds is not None and l1_ds is not None:
            ds = list(l2_ds) + [l1_ds]
        elif l2_ds is not None:
            ds = l2_ds
        else:
            ds = None

        pad_tuple = tuple(int(p) for p in pad)
        output = _l0_dec_jit(cached_l0_gd[0], st_l0, x_for_l0, skip0, pad_tuple, ds)

        # Compute loss
        if deep_supervision and isinstance(output, list):
            logits_finest = output[-1]
            loss = deep_supervision_loss(output, label, n_classes=n_classes,
                                         no_background=no_background)
        else:
            logits_finest = output[-1] if isinstance(output, list) else output
            loss = dice_ce_loss(logits_finest, label, n_classes,
                                no_background=no_background)

        pred = jnp.argmax(logits_finest, axis=1)
        return loss, pred

    return split_val_step
