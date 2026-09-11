# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""gfx938/gfx946 FP8 paged MQA scoring with Opus tiled-MMA kernels."""

from functools import lru_cache

import torch

from ...jit.core import compile_ops


@compile_ops("module_mqa_logits", fc_name="paged_mqa_logits_opus")
def _paged_mqa_logits_opus(q: torch.Tensor, cache: torch.Tensor,
                          weights: torch.Tensor, context: torch.Tensor,
                          tables: torch.Tensor, output: torch.Tensor,
                          max_len: int, kernel_id: int) -> None: ...


@lru_cache(maxsize=None)
def _require_arch(device_index):
    arch = torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]
    if arch not in ("gfx938", "gfx946"):
        raise NotImplementedError(f"paged_mqa_logits supports gfx938/gfx946, got {arch}")
    return arch


def _resolve_paged_mqa_kernel(kernel_id, *, rows, max_len, arch="gfx938"):
    if arch == "gfx946":
        # PMD verifies correctness, not production performance thresholds.
        if kernel_id is None:
            return 4
        if type(kernel_id) is not int or kernel_id not in (4, 5):
            raise ValueError("gfx946 kernelId must be None, 4 (K32), or 5 (K64 prefetch)")
        return kernel_id
    if arch != "gfx938":
        raise NotImplementedError(f"paged_mqa_logits does not support {arch}")
    if kernel_id is None:
        # On gfx938, compact K32 staging wins at medium grids; register
        # prefetch/K64 wins at small and large grids. Metadata-only selection.
        tiles = rows*((max_len+63)//64)
        return 2 if 128 < tiles <= 1024 else 3
    if type(kernel_id) is not int or not 0 <= kernel_id <= 3:
        raise ValueError("kernelId must be None or an integer in [0,3]")
    return kernel_id


def paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables,
                     max_model_len, *, out=None, clean_logits=True, kernelId=None):
    """Return FP32 weighted-ReLU QK scores ``[B*R,max_model_len]``.

    Supported: gfx938/gfx946; contiguous Q E4M3FN [B,R,H,128], R=1/2/4,
    H=32/64; uint8 KV [P,1,1,132] containing 128 K bytes then one
    positive finite FP32 scale; FP32 weights [B*R,H]; int32 lengths [B]
    and page tables [B,T]. Q/cache addresses must be 4-byte aligned.

    Caller metadata obligations: R<=length[b]<=min(max_model_len,T),
    valid page ids in [0,P), and positive finite scales. These device values
    are not read back to the host in the hot path. Query r sees candidates
    n<length[b]-R+r+1. Weights may be negative. Inputs are inference-only.

    On gfx938, ``kernelId=None`` selects ID2 for 128 < B*R*ceil(max_model_len/64)
    <= 1024, otherwise ID3. Both use four waves, N64 tiles, DPP head
    reduction, and fused invalid-tail writes. ID2 stages K32; ID3 prefetches
    D128 into registers and stages K64. Selection uses host shape metadata.
    Explicit IDs 0/1 preserve the initial K32 kernels for comparison: ID0
    uses full LDS head scores; ID1 uses wave shuffles and LDS partials.
    On gfx946, None selects ID4 (K32); explicit ID5 uses K64 prefetch.
    IDs 4/5 use the N64 DPP pipeline with 8-byte aligned LDS row strides
    and fused invalid-tail writes. gfx946 is validated on Perf Model;
    physical-device correctness and performance tuning remain pending.
    IDs 0-3 are gfx938-only; IDs 4/5 are gfx946-only.
    No Triton fallback is imported. Unsupported shapes/architectures raise.
    ``out`` may be preallocated and must not share storage with inputs.
    With ``clean_logits=True`` (default), invalid output is -inf, including
    the far tail. With False the caller owns tail initialization/masking.
    Allocation/fill occur on the current stream; no host synchronization is
    performed. Warm up the JIT before capturing the call in a GPU graph.
    """
    inputs = (q, kv_cache, weights, context_lens, block_tables)
    if any(not t.is_cuda or t.device != q.device for t in inputs):
        raise ValueError("inputs must be on the same GPU")
    arch = _require_arch(q.device.index)
    if any(not t.is_contiguous() or t.requires_grad for t in inputs):
        raise ValueError("inputs must be contiguous inference tensors")
    if q.ndim != 4 or q.dtype != torch.float8_e4m3fn:
        raise ValueError("q must be E4M3FN [B,R,H,128]")
    b, r, h, d = q.shape
    if b < 1 or b*r > 65535 or r not in (1,2,4) or h not in (32,64) or d != 128:
        raise ValueError("requires B>0, B*R<=65535, R=1/2/4, H=32/64, D=128")
    if (kv_cache.ndim != 4 or kv_cache.dtype != torch.uint8
            or not 0 < kv_cache.shape[0] <= (2**31-1)//132
            or tuple(kv_cache.shape[1:]) != (1,1,132)):
        raise ValueError("cache must be uint8 [P,1,1,132], P*132<=INT_MAX")
    if q.data_ptr()%4 or kv_cache.data_ptr()%4:
        raise ValueError("Q/cache must have 4-byte aligned storage")
    if weights.dtype != torch.float32 or tuple(weights.shape) != (b*r,h):
        raise ValueError("weights must be FP32 [B*R,H]")
    if context_lens.dtype != torch.int32 or tuple(context_lens.shape) != (b,):
        raise ValueError("context_lens must be int32 [B]")
    if (block_tables.dtype != torch.int32 or block_tables.ndim != 2
            or block_tables.shape[0] != b or not 0 < block_tables.shape[1] < 2**31):
        raise ValueError("block_tables must be int32 [B,T]")
    if type(max_model_len) is not int or not 0 < max_model_len < 2**31:
        raise ValueError("max_model_len must be a positive int32 value")
    kid = _resolve_paged_mqa_kernel(kernelId,rows=b*r,max_len=max_model_len,arch=arch)
    if type(clean_logits) is not bool:
        raise ValueError("clean_logits must be bool")
    if out is None:
        out = torch.empty((b*r,max_model_len), device=q.device, dtype=torch.float32)
    if (out.device != q.device or out.dtype != torch.float32 or out.requires_grad
            or not out.is_contiguous() or tuple(out.shape) != (b*r,max_model_len)):
        raise ValueError("out must be contiguous FP32 [B*R,max_model_len] on input GPU")
    if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in inputs):
        raise ValueError("out must not share storage with inputs")
    if clean_logits and kid < 2:
        out.fill_(-float("inf"))
    _paged_mqa_logits_opus(q,kv_cache,weights,context_lens,block_tables,out,max_model_len,kid)
    return out
