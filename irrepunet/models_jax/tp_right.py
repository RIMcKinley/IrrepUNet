"""Tensor product right-evaluation in pure JAX.

Direct per-path evaluation using CG coefficients from cuequivariance.
Each path operates on small (mul, ir_dim) tensors, avoiding the O(in_dim^3)
backward pass of the identity trick approach.
"""

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import cuequivariance as cue


def build_tp_right(
    cue_irreps_in: cue.Irreps,
    cue_irreps_sh: cue.Irreps,
    cue_irreps_out: cue.Irreps,
) -> Tuple[Callable, int, Dict[str, jax.Array]]:
    """Build a tensor product right-evaluation function for JAX.

    Given: output[kw] = sum_{ijk} CG[ijk] * weight[uvw] * input[iu] * sh[jv]

    .right() fixes sh and weight, producing a matrix:
        result[..., in_dim, out_dim] such that output = input @ result

    Uses direct per-path evaluation: iterates over TP paths at trace time,
    computing per-path contributions with small einsums. This avoids the
    identity trick (which inflates the batch dimension by in_dim and causes
    O(in_dim^3) backward pass scaling).

    Parameters
    ----------
    cue_irreps_in, cue_irreps_sh, cue_irreps_out : cue.Irreps
        Input, SH, and output irreps in cuequivariance format.

    Returns
    -------
    tp_right_fn : callable
        (sh: (N, sh_dim), weight: (N, weight_numel), constants: dict)
        -> (N, in_dim, out_dim)
    weight_numel : int
        Number of weight parameters.
    constants : dict
        CG coefficient arrays to pass to tp_right_fn.
    """
    d = cue.descriptors.fully_connected_tensor_product(
        cue_irreps_in, cue_irreps_sh, cue_irreps_out
    )
    stp = d.polynomial.operations[0][1]

    # Operand layout: 0=weights(u,v,w), 1=input(i,u), 2=sh(j,v), 3=output(k,w)
    weight_seg_slices = stp.operands[0].segment_slices()
    sh_seg_slices = stp.operands[2].segment_slices()
    num_in_segs = stp.operands[1].num_segments
    num_out_segs = stp.operands[3].num_segments
    weight_numel = stp.operands[0].size

    in_seg_slices = stp.operands[1].segment_slices()
    out_seg_slices = stp.operands[3].segment_slices()
    in_seg_dims = [sl.stop - sl.start for sl in in_seg_slices]
    out_seg_dims = [sl.stop - sl.start for sl in out_seg_slices]

    # Precompute all CG coefficients and path metadata
    constants = {}
    path_meta = []

    for pid, path in enumerate(stp.paths):
        cg = jnp.array(path.coefficients, dtype=jnp.float32)
        i_dim, j_dim, k_dim = cg.shape

        w_seg_idx, in_seg_idx, sh_seg_idx, out_seg_idx = path.indices
        u, _v, w = stp.operands[0].segments[w_seg_idx]
        i = stp.operands[1].segments[in_seg_idx][0]
        j = stp.operands[2].segments[sh_seg_idx][0]
        k = stp.operands[3].segments[out_seg_idx][0]

        w_sl = weight_seg_slices[w_seg_idx]
        sh_sl = sh_seg_slices[sh_seg_idx]

        meta = {
            'pid': pid,
            'in_seg_idx': in_seg_idx,
            'out_seg_idx': out_seg_idx,
            'u': u, 'w': w, 'i': i, 'j': j, 'k': k,
            'w_start': w_sl.start, 'w_stop': w_sl.stop,
            'sh_start': sh_sl.start, 'sh_stop': sh_sl.stop,
        }

        if j_dim == 1:
            # l_sh=0: CG = scalar * eye(i), i == k
            path_weight = float(path.coefficients.flat[0])
            constants[f'eye_{pid}'] = jnp.eye(i_dim, dtype=jnp.float32)
            meta['type'] = 'sh0'
            meta['path_weight'] = path_weight
        elif i_dim == 1:
            # l_in=0: CG collapses to (j, k)
            constants[f'cg_{pid}'] = cg.squeeze(0)
            meta['type'] = 'in0'
        elif k_dim == 1:
            # l_out=0: CG collapses to (j, i), store transposed for matmul
            constants[f'cg_{pid}'] = cg.squeeze(2).T
            meta['type'] = 'out0'
        else:
            # General case: full CG
            constants[f'cg_{pid}'] = cg
            meta['type'] = 'general'

        path_meta.append(meta)

    def tp_right_fn(sh, weight, consts):
        """Compute tensor product right-evaluation.

        Parameters
        ----------
        sh : jax.Array, shape (N, sh_dim)
        weight : jax.Array, shape (N, weight_numel)
        consts : dict of CG arrays

        Returns
        -------
        jax.Array, shape (N, in_dim, out_dim)
        """
        blocks = {}

        for meta in path_meta:
            pid = meta['pid']
            u, w = meta['u'], meta['w']
            i, j, k = meta['i'], meta['j'], meta['k']

            w_data = weight[:, meta['w_start']:meta['w_stop']].reshape(-1, u, w)
            sh_data = sh[:, meta['sh_start']:meta['sh_stop']]

            ptype = meta['type']
            if ptype == 'sh0':
                eye = consts[f'eye_{pid}'].astype(sh.dtype)
                pw = meta['path_weight']
                scaled_sh = sh_data * pw
                scaled_w = (scaled_sh * w_data.reshape(-1, u * w)).reshape(-1, u, w)
                block = jnp.einsum('nuw,ik->nuiwk', scaled_w, eye)

            elif ptype == 'in0':
                cg_arr = consts[f'cg_{pid}'].astype(sh.dtype)
                sh_cg = sh_data @ cg_arr
                block = jnp.einsum('nk,nuw->nuwk', sh_cg, w_data)

            elif ptype == 'out0':
                cg_arr = consts[f'cg_{pid}'].astype(sh.dtype)
                sh_cg = sh_data @ cg_arr
                block = jnp.einsum('ni,nuw->nuiw', sh_cg, w_data)

            else:
                cg_arr = consts[f'cg_{pid}'].astype(sh.dtype)
                block = jnp.einsum('ijk,nuw,nj->nuiwk', cg_arr, w_data, sh_data)

            block = block.reshape(-1, u * i, w * k)
            key = (meta['in_seg_idx'], meta['out_seg_idx'])
            if key not in blocks:
                blocks[key] = block
            else:
                blocks[key] = blocks[key] + block

        # Assembly: concatenate along in/out dimensions
        N = sh.shape[0]
        rows = []
        for in_idx in range(num_in_segs):
            cols = []
            for out_idx in range(num_out_segs):
                if (in_idx, out_idx) in blocks:
                    cols.append(blocks[(in_idx, out_idx)])
                else:
                    cols.append(jnp.zeros(
                        (N, in_seg_dims[in_idx], out_seg_dims[out_idx]),
                        dtype=sh.dtype,
                    ))
            rows.append(jnp.concatenate(cols, axis=2))
        result = jnp.concatenate(rows, axis=1)

        return result

    return tp_right_fn, weight_numel, constants
