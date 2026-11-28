# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Output-only recurrent GDN verification with a compact ReplaySSM ring.

The checkpoint/ring contract follows SGLang's GDN ReplaySSM implementation:
FP32 [slot, value_head, V, K] checkpoints, activation-dtype D/K high parts,
optional low parts or exact raw inputs, and request-keyed circular cursors.
This entry only verifies one window. The caller owns cursor advancement and
checkpoint commit/flush, and must commit a flush row before verifying it.

Unlike conventional recurrent verification, this kernel never materializes
per-token full-state snapshots. It reconstructs a register tile from the
checkpoint and committed compact history, applies the short draft chain, and
writes outputs plus compact records. A linear draft chain is required.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_replayssm_recurrent_verify(
    q, k, v, a, b, A_log, dt_bias, out, checkpoint,
    d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
    cu_seqlens, state_indices, replay_indices, write_pos, cache_base, is_flush,
    scale,
    stride_q: tl.constexpr, stride_k: tl.constexpr, stride_v: tl.constexpr,
    stride_a: tl.constexpr, stride_b: tl.constexpr, stride_out: tl.constexpr,
    stride_state: tl.constexpr, stride_d: tl.constexpr, stride_kc: tl.constexpr,
    stride_gc: tl.constexpr, stride_rawv: tl.constexpr, stride_rawk: tl.constexpr,
    stride_beta: tl.constexpr, stride_cu: tl.constexpr,
    stride_si: tl.constexpr, stride_ri: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, L: tl.constexpr,
    NORMALIZE: tl.constexpr, FLUSH: tl.constexpr, NULL_ID: tl.constexpr,
    EXACT_RAW: tl.constexpr, RESIDUAL: tl.constexpr,
    HISTORY_DOT: tl.constexpr,
):
    iv, n, hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = hv // (HV // H)
    bos = tl.load(cu_seqlens + n * stride_cu).to(tl.int64)
    eos = tl.load(cu_seqlens + (n + 1) * stride_cu).to(tl.int64)
    state = tl.load(state_indices + n * stride_si).to(tl.int64)
    replay = tl.load(replay_indices + n * stride_ri).to(tl.int64)
    kk = tl.arange(0, K)
    vv = iv * BV + tl.arange(0, BV)
    mv = vv < V
    if state <= NULL_ID:
        if not FLUSH:
            for token in range(bos, eos):
                tl.store(out + token * stride_out + hv * V + vv, 0.0, mask=mv)
        return
    if eos <= bos:
        return
    flush = tl.load(is_flush + replay) != 0
    if flush != FLUSH:
        return
    pos = tl.load(write_pos + replay).to(tl.int32)
    base = tl.load(cache_base + replay).to(tl.int32)
    hp = checkpoint + state * stride_state + hv * V * K + vv[None, :] * K + kk[:, None]
    state_tile = tl.load(hp, mask=mv[None, :], other=0.0).to(tl.float32)

    # Match the compact output reconstruction contract: history uses D/K
    # high parts; low parts are consumed by the separate checkpoint commit.
    if not FLUSH and HISTORY_DOT:
        tt = tl.arange(0, L)
        history_physical = ((base + tt) & (L - 1)).to(tl.int64)
        valid = tt < pos
        gates = tl.load(g_cache + replay * stride_gc + hv * L + history_physical,
                        mask=valid, other=0.0).to(tl.float32)
        total = tl.sum(gates, 0)
        weights = tl.where(valid, tl.exp(total - tl.cumsum(gates, 0)), 0.0)
        keys = tl.load(k_cache + replay * stride_kc + (h * L + history_physical[None, :]) * K + kk[:, None],
                       mask=valid[None, :], other=0.0)
        updates = tl.load(d_cache + replay * stride_d + (hv * L + history_physical[:, None]) * V + vv[None, :],
                          mask=valid[:, None] & mv[None, :], other=0.0).to(tl.float32) * weights[:, None]
        # Two activation-dtype products retain the FP32 decay scaling while
        # letting the history reconstruction use matrix instructions.
        hi = updates.to(keys.dtype)
        lo = (updates - hi.to(tl.float32)).to(keys.dtype)
        delta = tl.dot(keys, hi) + tl.dot(keys, lo)
        state_tile = state_tile * tl.exp(total) + delta
    elif not FLUSH:
        for t in range(pos):
            physical = ((base + t) & (L - 1)).to(tl.int64)
            kh = tl.load(k_cache + replay * stride_kc +
                         (h * L + physical) * K + kk).to(tl.float32)
            dh = tl.load(d_cache + replay * stride_d + (hv * L + physical)
                         * V + vv, mask=mv, other=0.0).to(tl.float32)
            gh = tl.load(
                g_cache + replay * stride_gc + hv * L + physical
            ).to(tl.float32)
            state_tile = state_tile * tl.exp(gh) + kh[:, None] * dh[None, :]

    decay = tl.exp(tl.load(A_log + hv).to(tl.float32))
    bias = tl.load(dt_bias + hv).to(tl.float32)
    write_key = (iv == 0) & (hv % (HV // H) == 0)
    for token in range(bos, eos):
        raw_key = tl.load(k + token * stride_k + h * K + kk).to(tl.float32)
        query = tl.load(q + token * stride_q + h * K + kk).to(tl.float32)
        value = tl.load(
            v + token * stride_v + hv * V + vv, mask=mv, other=0.0
        ).to(tl.float32)
        raw_value = value
        x = tl.load(a + token * stride_a + hv).to(tl.float32) + bias
        gate = -decay * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        beta = 1.0 / (1.0 + tl.exp(-tl.load(b + token * stride_b + hv).to(tl.float32)))
        key = raw_key
        if NORMALIZE:
            key = key * (1.0 / tl.sqrt(tl.sum(key * key, 0) + 1e-6))
            query = query * (1.0 / tl.sqrt(tl.sum(query * query, 0) + 1e-6))
        query *= scale
        state_tile *= tl.exp(gate)
        update = (value - tl.sum(state_tile * key[:, None], 0)) * beta
        state_tile += key[:, None] * update[None, :]
        result = tl.sum(state_tile * query[:, None], 0)
        tl.store(out + token * stride_out + hv * V + vv, result, mask=mv)

        logical = pos + token - bos
        physical = ((base + logical) & (L - 1)).to(tl.int64)
        writable = logical < L
        dp = d_cache + replay * stride_d + (hv * L + physical) * V + vv
        high_d = update.to(d_cache.dtype.element_ty)
        tl.store(dp, high_d, mask=mv & writable)
        if EXACT_RAW or RESIDUAL:
            rp = rawv_cache + replay * stride_rawv + (hv * L + physical) * V + vv
            if EXACT_RAW:
                tl.store(rp, raw_value, mask=mv & writable)
            else:
                tl.store(rp, update - high_d.to(tl.float32), mask=mv & writable)
        if iv == 0:
            tl.store(
                g_cache + replay * stride_gc + hv * L + physical,
                gate, mask=writable,
            )
            if EXACT_RAW:
                tl.store(
                    beta_cache + replay * stride_beta + hv * L + physical,
                    beta, mask=writable,
                )
        if write_key:
            kp = k_cache + replay * stride_kc + (h * L + physical) * K + kk
            high_k = key.to(k_cache.dtype.element_ty)
            tl.store(kp, high_k, mask=writable)
            if EXACT_RAW or RESIDUAL:
                rkp = rawk_cache + replay * stride_rawk + (h * L + physical) * K + kk
                if EXACT_RAW:
                    tl.store(rkp, raw_key, mask=writable)
                else:
                    tl.store(rkp, key - high_k.to(tl.float32), mask=writable)


def gdn_replayssm_verify(
    q, k, v, a, b, A_log, dt_bias, out, checkpoint_state,
    d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache,
    query_start_loc, ssm_state_indices, replay_indices, write_pos, cache_base,
    is_flush, scale, max_cache_len, max_spec_len,
    use_qk_l2norm_in_kernel=True, is_flush_kernel=False,
    block_v=64, num_warps=2, null_block_id=0, history_dot=True,
):
    """Verify a linear GDN draft window without full-state snapshots.

    K=V=128 and an FP32 checkpoint are required. All inner tensor axes must
    be dense; token and pool slot strides may be padded. Ring positions must
    satisfy write_pos + sequence_length <= max_cache_len. The caller must
    ensure nonoverlapping live state/ring indices and valid index ranges.
    No tensor values are copied to the host, so the operation is Graph safe.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or q.shape != k.shape:
        raise ValueError('Expected packed [tokens, heads, width] Q/K/V tensors')
    H, K = k.shape[-2:]
    HV, V = v.shape[-2:]
    if K != 128 or V != 128 or checkpoint_state.dtype != torch.float32:
        raise ValueError('GDN ReplaySSM verify requires K=V=128 and FP32 state')
    if HV % H or max_cache_len < 1 or max_cache_len & (max_cache_len - 1):
        raise ValueError('Invalid grouped-head ratio or circular cache length')
    if block_v not in (16, 32, 64) or num_warps not in (1, 2, 4, 8):
        raise ValueError('Unsupported GDN ReplaySSM tile configuration')
    if max_spec_len < 1 or max_spec_len > max_cache_len:
        raise ValueError('The draft window must fit the ring')
    if history_dot and (max_cache_len < 16 or k_cache.dtype not in (
            torch.float16, torch.bfloat16)):
        raise ValueError(
            'Matrix history reconstruction requires a >=16-entry FP16/BF16 ring')
    if q.shape[0] != v.shape[0] or tuple(a.shape) != (
            q.shape[0], HV) or b.shape != a.shape:
        raise ValueError('Token counts or gate shapes do not match')
    if any(x.stride(-1) != 1 for x in (q, k, v, a, b)) or any(
        x.stride(-2) != x.shape[-1] for x in (q, k, v)
    ):
        raise ValueError('Head and width axes must be dense')
    if (rawv_cache is None) != (rawk_cache is None) or (
            beta_cache is not None and rawv_cache is None):
        raise ValueError('Raw/low buffers must be supplied as a pair')
    exact = rawv_cache is not None and rawk_cache is not None and beta_cache is not None
    residual = rawv_cache is not None and rawk_cache is not None and beta_cache is None
    rv = rawv_cache if rawv_cache is not None else d_cache
    rk = rawk_cache if rawk_cache is not None else k_cache
    bc = beta_cache if beta_cache is not None else g_cache
    _gdn_replayssm_recurrent_verify[(triton.cdiv(V, block_v), query_start_loc.numel()-1, HV)](
        q, k, v, a, b, A_log, dt_bias, out, checkpoint_state, d_cache, k_cache, g_cache, rv, rk, bc,
        query_start_loc, ssm_state_indices, replay_indices, write_pos, cache_base, is_flush, scale,
        q.stride(0), k.stride(0), v.stride(0), a.stride(0), b.stride(0), out.stride(0),
        checkpoint_state.stride(0), d_cache.stride(0), k_cache.stride(0), g_cache.stride(0),
        rv.stride(0), rk.stride(0), bc.stride(0), query_start_loc.stride(0),
        ssm_state_indices.stride(0), replay_indices.stride(0), H=H, HV=HV, K=K, V=V,
        BV=block_v, L=max_cache_len, NORMALIZE=use_qk_l2norm_in_kernel,
        FLUSH=is_flush_kernel, NULL_ID=null_block_id, EXACT_RAW=exact, RESIDUAL=residual,
        HISTORY_DOT=history_dot,
        num_warps=num_warps, num_stages=2,
    )
    return out
