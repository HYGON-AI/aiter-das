# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# [gfx946 minimal test] fused_qk_norm_mrope_3d_cache_pts_quant_shuffle
# (csrc/kernels/fused_qk_norm_mrope_cache_quant.cu). One run: 3 tokens x
# (2q + 1k + 1v) heads x head_size 128, bf16 qkv -> fp8-e4m3fn cache with
# per-tensor scales (k 1.5 / v 2.0), neox-style non-interleaved MRoPE
# (mrope_section [24, 20, 20]), shuffle layout (block_size 16, x 16),
# slot_mapping [3, 16, 19] spanning two blocks, return_kv enabled.
#
# The CPU reference mirrors op_tests/test_fused_qk_norm_mrope_cache_quant.py
# (run_torch_mrope_3d_rms_set_kv_shuffle) and the kernel semantics from
# csrc/kernels/rope/rope_common.h: quant = (x / scale).to(fp8), shuffle store
# k_cache[b, h, j//x, o, j%x] / v_cache[b, h, o//x, j, o%x], q_out = norm+rope
# result, k_out/v_out = quantized k/v in token order.
#
# NOTE: cache dtype is torch.float8_e4m3fn on purpose (the kernel only accepts
# fp8_e4m3fn / fp8_e4m3fnuz when it differs from the qkv dtype); aiter
# dtypes.fp8 falls back to uint8 on gfx946 and would be rejected.
#
# Run: GPU_CHIP=sb python op_tests/gfx946_tests/test_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle.py
import os as _gfx946_os
import sys as _gfx946_sys
_gfx946_dir = _gfx946_os.path.dirname(_gfx946_os.path.abspath(__file__))
if _gfx946_dir not in _gfx946_sys.path:
    _gfx946_sys.path.insert(0, _gfx946_dir)  # single_run_common / scalar_type
_gfx946_root = _gfx946_os.path.dirname(_gfx946_os.path.dirname(_gfx946_dir))
if _gfx946_root not in _gfx946_sys.path:
    _gfx946_sys.path.append(_gfx946_root)  # repo root; appended so an installed aiter still wins

import torch

import aiter  # noqa: F401 (side effects on the target machine)
from single_run_common import checkAllclose
from aiter.ops.fused_qk_norm_mrope_cache_quant import (
    fused_qk_norm_mrope_3d_cache_pts_quant_shuffle,
)

FP8_E4M3 = torch.float8_e4m3fn
DTYPE = torch.bfloat16
NUM_TOKENS, HQ, HK, HV, HEAD_SIZE = 3, 2, 1, 1, 128
MROPE_SECTION = [24, 20, 20]  # must sum to head_size // 2 (full rotary)
EPS = 1e-6
K_SCALE, V_SCALE = 1.5, 2.0
BLOCK_SIZE = 16
X = 16  # 16 // fp8 element size
NUM_BLOCKS = 2
SLOTS = [3, 16, 19]  # block 0 offset 3, block 1 offsets 0 and 3
# (t, h, w) positions per token, all < NUM_BLOCKS * BLOCK_SIZE rows of cos_sin
POSITIONS = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def rms_norm_forward(x, weight, eps):
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    return weight * x


def apply_rotary_emb_torch(x, cos, sin, is_neox_style):
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def mrope_select(cs, mrope_section):
    # cs: [3, num_tokens, head_size // 2] laid out as chunked sections
    # [TTT...HHH...WWW]; section i is taken from axis i.
    return torch.cat(
        [m[i] for i, m in enumerate(cs.split(mrope_section, dim=-1))], dim=-1
    )


def quant_fp8(x, scale):
    return (x.float() / scale).to(FP8_E4M3)


def write_shuffle_cache(kq, vq, k_cache, v_cache, slots, block_size, x):
    # k_cache: [num_blocks, hk, head_size // x, block_size, x]
    # v_cache: [num_blocks, hv, block_size // x, head_size, x]
    hk, head_size = kq.shape[1], kq.shape[2]
    for t, slot in enumerate(slots):
        b, o = slot // block_size, slot % block_size
        k_cache[b, :, :, o, :] = kq[t].reshape(hk, head_size // x, x)
        v_cache[b, :, o // x, :, o % x] = vq[t]


def main():
    torch.manual_seed(0)
    heads = HQ + HK + HV
    qkv = torch.randn(NUM_TOKENS, heads, HEAD_SIZE).to(DTYPE)
    qw = torch.randn(HEAD_SIZE).to(DTYPE)
    kw = torch.randn(HEAD_SIZE).to(DTYPE)
    cos_sin = torch.randn(NUM_BLOCKS * BLOCK_SIZE, HEAD_SIZE).to(DTYPE)
    positions = torch.tensor(POSITIONS, dtype=torch.int64)  # (3, num_tokens)
    kv_loc = torch.tensor(SLOTS, dtype=torch.int64)
    k_cache = torch.rand(NUM_BLOCKS, HK, HEAD_SIZE // X, BLOCK_SIZE, X).to(FP8_E4M3)
    v_cache = torch.rand(NUM_BLOCKS, HV, BLOCK_SIZE // X, HEAD_SIZE, X).to(FP8_E4M3)

    # ---------------- reference (CPU) ----------------
    qkv_flat = qkv.reshape(NUM_TOKENS, heads * HEAD_SIZE)
    q, k, v = qkv_flat.split(
        [HQ * HEAD_SIZE, HK * HEAD_SIZE, HV * HEAD_SIZE], dim=-1
    )
    q = rms_norm_forward(q.reshape(NUM_TOKENS, HQ, HEAD_SIZE), qw, EPS).reshape(q.shape)
    k = rms_norm_forward(k.reshape(NUM_TOKENS, HK, HEAD_SIZE), kw, EPS).reshape(k.shape)
    cos, sin = cos_sin[positions].chunk(2, dim=-1)  # [3, num_tokens, head_size//2]
    cos = mrope_select(cos, MROPE_SECTION)  # [num_tokens, head_size // 2]
    sin = mrope_select(sin, MROPE_SECTION)
    q = apply_rotary_emb_torch(
        q.reshape(NUM_TOKENS, HQ, HEAD_SIZE), cos, sin, True
    ).reshape(q.shape)
    k = apply_rotary_emb_torch(
        k.reshape(NUM_TOKENS, HK, HEAD_SIZE), cos, sin, True
    ).reshape(k.shape)
    q_ref = q.reshape(NUM_TOKENS, HQ, HEAD_SIZE)
    kq = quant_fp8(k.reshape(NUM_TOKENS, HK, HEAD_SIZE), K_SCALE)
    vq = quant_fp8(v.reshape(NUM_TOKENS, HV, HEAD_SIZE), V_SCALE)
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    write_shuffle_cache(kq, vq, k_cache_ref, v_cache_ref, SLOTS, BLOCK_SIZE, X)

    # ---------------- single op run on the perf model ----------------
    q_out = torch.empty(NUM_TOKENS, HQ, HEAD_SIZE, dtype=DTYPE, device="cuda")
    k_out = torch.empty(NUM_TOKENS, HK, HEAD_SIZE, dtype=FP8_E4M3, device="cuda")
    v_out = torch.empty(NUM_TOKENS, HV, HEAD_SIZE, dtype=FP8_E4M3, device="cuda")
    k_cache_dev = k_cache.cuda()
    v_cache_dev = v_cache.cuda()
    fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
        qkv.cuda(),
        qw.cuda(),
        kw.cuda(),
        cos_sin.cuda(),
        positions.cuda(),
        NUM_TOKENS,
        HQ,
        HK,
        HV,
        HEAD_SIZE,
        True,  # is_neox_style
        MROPE_SECTION,
        False,  # is_interleaved (chunked T/H/W sections)
        EPS,
        q_out,
        k_cache_dev,
        v_cache_dev,
        kv_loc.cuda(),
        torch.tensor(K_SCALE),
        torch.tensor(V_SCALE),
        k_out,
        v_out,
        True,  # return_kv
        True,  # use_shuffle_layout
        BLOCK_SIZE,
        X,
    )

    # ---------------- checks (CPU) ----------------
    msg = (
        f"fused_qk_norm_mrope_3d_cache_pts_quant_shuffle "
        f"[tokens={NUM_TOKENS}, hq/hk/hv={HQ}/{HK}/{HV}, head={HEAD_SIZE}, "
        f"bf16->fp8, k/v_scale={K_SCALE}/{V_SCALE}, "
        f"mrope={MROPE_SECTION}, block={BLOCK_SIZE}, x={X}, slots={SLOTS}]"
    )
    total = 0
    total += checkAllclose(
        q_ref.float(), q_out.cpu().float(), rtol=1e-2, atol=0.05, msg=msg + " [q_out]"
    )
    # fp8 payloads compared as int8 views: allow a 1-step (1-ulp) rounding diff
    for name, ref, got in [
        ("k_cache", k_cache_ref, k_cache_dev.cpu()),
        ("v_cache", v_cache_ref, v_cache_dev.cpu()),
        ("k_out", kq, k_out.cpu()),
        ("v_out", vq, v_out.cpu()),
    ]:
        total += checkAllclose(
            ref.view(torch.int8).float(),
            got.view(torch.int8).float(),
            atol=1.0,
            rtol=0.0,
            msg=msg + f" [{name}]",
        )
    return total


if __name__ == "__main__":
    # threshold mirrors op_tests tolerance intent: any real index/quant bug
    # displaces whole rows, far above 1% of elements
    total = main()
    if total < 0.01:
        print("\nfused_qk_norm_mrope_3d_cache_pts_quant_shuffle: ALL PASSED")
    else:
        raise SystemExit(
            "\nfused_qk_norm_mrope_3d_cache_pts_quant_shuffle: FAILED "
            "(see mismatches above)"
        )