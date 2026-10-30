# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# [gfx946 minimal test] aiter.per_token_group_quant_fp8 HIP kernel
# (csrc/kernels/per_token_group_quant_fp8_kernels.cu, e4m3 path). Two tiny runs
# (2x128 fp16, group_size=128; use_ue8m0 False/True) against the pure-torch
# reference from op_tests/test_per_token_group_quant_fp8.py, which matches the
# kernel formula scale = max(amax, eps) / 448 (rounded up to a power of two when
# use_ue8m0). Basic functional correctness only, sized for the CPU perf model.
#
# NOTE: output dtype is torch.float8_e4m3fn on purpose (kernel's is_e4m3 path);
# aiter dtypes.fp8 falls back to uint8 on gfx946 and is only a storage ABI view.
import os as _gfx946_os
import sys as _gfx946_sys
_gfx946_dir = _gfx946_os.path.dirname(_gfx946_os.path.abspath(__file__))
if _gfx946_dir not in _gfx946_sys.path:
    _gfx946_sys.path.insert(0, _gfx946_dir)  # single_run_common / scalar_type
_gfx946_root = _gfx946_os.path.dirname(_gfx946_os.path.dirname(_gfx946_dir))
if _gfx946_root not in _gfx946_sys.path:
    _gfx946_sys.path.append(_gfx946_root)  # repo root; appended so an installed aiter still wins

import torch

import aiter
from single_run_common import checkAllclose

FP8_E4M3 = torch.float8_e4m3fn


def native_per_token_group_quant_fp8(x, group_size, eps=1e-5, use_ue8m0=False):
    finfo = torch.finfo(FP8_E4M3)
    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    scale = amax / finfo.max
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale.clamp(min=eps))))
    x_q = (x_ / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_E4M3)
    return x_q.reshape(x.shape), scale.reshape(
        x.shape[:-1] + (x.shape[-1] // group_size,)
    )


def test_per_token_group_quant_fp8(
    dtype, num_tokens, hidden_size, group_size, eps, use_ue8m0
):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    out_q = torch.empty(x.shape, dtype=FP8_E4M3, device=x.device)
    out_s = torch.empty(
        (num_tokens, hidden_size // group_size), dtype=torch.float32, device=x.device
    )

    aiter.per_token_group_quant_fp8(out_q, x, out_s, group_size, eps, use_ue8m0)
    torch.cuda.synchronize()

    q_ref, s_ref = native_per_token_group_quant_fp8(
        x, group_size, eps=eps, use_ue8m0=use_ue8m0
    )

    msg = (
        f"per_token_group_quant_fp8 {dtype} [{num_tokens}x{hidden_size}] "
        f"group={group_size} ue8m0={use_ue8m0}"
    )
    # compare fp8 payloads as int8 views: allow a 1-step (1-ulp) rounding diff
    ret_q = checkAllclose(
        q_ref.view(torch.int8).float(),
        out_q.view(torch.int8).float(),
        atol=1.0,
        rtol=0.0,
        msg=msg + " [q]",
    )
    atol = 1e-4 if use_ue8m0 else 1e-5
    ret_s = checkAllclose(s_ref, out_s, atol=atol, rtol=atol, msg=msg + " [scale]")
    return ret_q + ret_s


if __name__ == "__main__":
    # smallest meaningful cases: 2 groups of 128, one run per ue8m0 variant
    total = 0
    for use_ue8m0 in (False, True):
        total += test_per_token_group_quant_fp8(
            torch.float16, 2, 128, 128, 1e-5, use_ue8m0
        )
    if total == 0:
        print("\nper_token_group_quant_fp8: ALL PASSED")
    else:
        raise SystemExit("\nper_token_group_quant_fp8: FAILED (see mismatches above)")