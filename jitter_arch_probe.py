#!/usr/bin/env python
"""Probe within-architecture kernel-sampling sensitivity.

Stays strictly inside the native spacing's architecture range (same Conv3d
kernel shapes, same pool factors) and varies the spacing to re-sample the
e3nn continuous basis at different points.  If dice is flat across the
arch range, unseen kernel sampling is not the source of native-spacing
underperformance; residual differences vs canonical must come from
crossing architecture boundaries.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from irrepunet.inference import sliding_window_inference
from irrepunet.models import (
    architecture_spacing_range,
    project_to_spacing,
    compute_architecture_key,
)
from irrepunet.models.unet import E3nnUNet


def _dice(pred, target):
    p = (pred == 1).astype(float)
    t = (target == 1).astype(float)
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    if union == 0 and t.sum() == 0:
        return 1.0
    return (2.0 * inter) / max(union, 1e-8)


def _within_arch_samples(model, native_sp, n_samples, margin=0.02, max_tries=800, seed=0):
    """Sample spacings strictly inside the native's architecture range.

    Uses ``compute_architecture_key`` as the authoritative "same architecture"
    oracle (it matches the projection machinery; ``architecture_spacing_range``
    can disagree at boundaries due to kernel-trim thresholds).  Candidates are
    drawn from the safe interior of the per-axis arch ranges and accepted only
    when their arch key equals native's.
    """
    import random
    rng = random.Random(seed)

    native_arch = compute_architecture_key(model, native_sp)
    ranges = architecture_spacing_range(model, native_sp)

    safe = []
    for d, (lo, hi) in enumerate(ranges):
        if not np.isfinite(hi):
            hi = native_sp[d] * 2.0
        gap = hi - lo
        safe.append((lo + margin * gap, hi - margin * gap))

    samples = [tuple(native_sp)]

    def _dup(sp):
        return any(
            all(abs(a - b) / max(b, 1e-6) < 0.005 for a, b in zip(sp, x))
            for x in samples
        )

    # Try deterministic grid first (linspace along all axes together)
    for frac in np.linspace(0.0, 1.0, n_samples):
        sp = tuple(float(lo + frac * (hi - lo)) for lo, hi in safe)
        if compute_architecture_key(model, sp) == native_arch and not _dup(sp):
            samples.append(sp)

    # Fill remainder with random interior points
    tries = 0
    while len(samples) < n_samples + 1 and tries < max_tries:
        tries += 1
        sp = tuple(rng.uniform(lo, hi) for lo, hi in safe)
        if compute_architecture_key(model, sp) == native_arch and not _dup(sp):
            samples.append(sp)

    return samples


def probe(experiment_dir, cases, n_samples, gpu, patch_mm):
    exp = Path(experiment_dir)
    device = torch.device(f'cuda:{gpu}')

    model, _ = E3nnUNet.load_checkpoint(str(exp / 'model_best.pt'),
                                        device=torch.device('cpu'))

    config = json.load(open(exp / 'config.json'))
    data = config.get('data', config)
    preproc = Path(data['preprocessed_dir'])
    model_cfg = config.get('model', config)
    n_downsample = model_cfg.get('n_downsample', 4)
    model_scale = float(model_cfg.get('scale', 2.0))

    all_rows = []

    for case in cases:
        props = pickle.load(open(preproc / f'{case}.pkl', 'rb'))
        native_sp = tuple(float(s) for s in props['spacing'])
        img = np.load(preproc / f'{case}.npy')
        seg = np.load(preproc / f'{case}_seg.npy')
        if seg.ndim == 4:
            seg = seg[0]

        native_ranges = architecture_spacing_range(model, native_sp)
        native_arch = compute_architecture_key(model, native_sp)

        print(f'\n=== {case} native=({native_sp[0]:.4f},{native_sp[1]:.4f},{native_sp[2]:.4f}) ===')
        print(f'Arch ranges (per-axis): ' +
              ', '.join(f'({lo:.4f},{hi:.4f}]' for lo, hi in native_ranges))

        ok_samples = _within_arch_samples(model, native_sp, n_samples)

        print(f'Probing {len(ok_samples)} within-arch samples...')

        # All samples share one architecture → one projection reused
        proj = None
        for i, sp in enumerate(ok_samples):
            t0 = time.time()
            if proj is None:
                proj = project_to_spacing(model, sp, use_2d=True)
                proj.to(device).eval()
            else:
                # Same arch — update sampled kernel weights in place
                from irrepunet.models import update_projected_weights
                update_projected_weights(proj, model, sp, verify=False)
            pt = time.time() - t0

            t0 = time.time()
            probs = sliding_window_inference(
                proj, img, sp,
                patch_size_mm=patch_mm, overlap=0.69,
                device=str(device), use_fp16=True, mirror_axes=None,
                sw_batch_size=2, n_downsample=n_downsample, model_scale=model_scale,
            )
            pred = probs.argmax(axis=0).astype(np.int16)
            d = _dice(pred, seg)
            it = time.time() - t0

            is_native = all(abs(a - b) / max(b, 1e-6) < 0.001
                            for a, b in zip(sp, native_sp))
            tag = 'NATIVE' if is_native else '      '
            print(f'  [{i+1:>2}/{len(ok_samples)}] {tag} '
                  f'({sp[0]:.4f},{sp[1]:.4f},{sp[2]:.4f})  dice={d:.4f}  '
                  f'(proj/update {pt:.1f}s, inf {it:.1f}s)')

            all_rows.append(dict(
                case=case, spacing=list(sp), native_spacing=list(native_sp),
                is_native=is_native, dice=float(d),
            ))

        if proj is not None:
            proj.cpu()
            del proj
            torch.cuda.empty_cache()

    out = exp / 'jitter_arch_probe.json'
    json.dump(all_rows, open(out, 'w'), indent=2)

    # Per-case summary
    print('\n\n=== Per-case within-arch dice summary ===')
    from collections import defaultdict
    by_case = defaultdict(list)
    for r in all_rows:
        by_case[r['case']].append(r['dice'])
    print(f'{"case":>12}  {"n":>3}  {"native":>7}  {"mean":>7}  {"range":>14}  {"std":>7}')
    for case, dices in by_case.items():
        # Find native dice
        nd = next((r['dice'] for r in all_rows if r['case'] == case and r['is_native']), None)
        a = np.array(dices)
        print(f'{case:>12}  {len(a):>3}  {nd if nd is not None else -1:>7.4f}  '
              f'{a.mean():>7.4f}  {a.min():.4f}-{a.max():.4f}  {a.std():>7.4f}')
    print(f'\nSaved {len(all_rows)} rows to {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment_dir')
    ap.add_argument('--cases', nargs='+', required=True)
    ap.add_argument('--n_samples', type=int, default=8,
                    help='Number of within-arch spacings to probe per case '
                         '(native is always included)')
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--patch_mm', type=float, default=80.0)
    args = ap.parse_args()
    probe(args.experiment_dir, args.cases, args.n_samples, args.gpu, args.patch_mm)
