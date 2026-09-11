# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Modified by Hygon in 2026: Hygon w8a8 expert-parallel execution and tests.

import pytest
import torch
import itertools
from typing import Optional, List  # Add this import at the top

from op_tests.utility.scalar_type import ScalarType, scalar_types
from op_tests.utility.utils import quantize_weights
from op_tests.utility.utils import torch_moe as torch_score_moe
from aiter.ops.triton.moe_op import fused_moe
from aiter.fused_moe import fused_topk, torch_moe
from aiter import ActivationType
from aiter.test_common import checkAllclose, perftest,benchmark
from aiter import ck_moe, ck_shuffle_moe, dtypes, silu_and_mul, gelu_and_mul, moe_sum
from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config_func
from aiter.ops.triton.fused_moe import fused_experts_impl
from aiter.ops.triton.utils.types import torch_to_triton_dtype
from aiter.fused_moe_asm_wna16 import fused_experts_asm_impl
from aiter import per_token_quant_hip, per_block_quant_wrapper
import pandas as pd
import aiter
import os
BLOCK_SIZE_M = 32
MAX_TOKENS = 65536
os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

@perftest(num_warmup=1, num_iters=2)
def torch_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    expert_mask=None,
):
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        expert_mask,
    )

@perftest(num_warmup=5, num_iters=10,testGraph=False)
def asm_fused_experts_impl(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       dtype,
                       inplace: bool = False,
                       activation: str = "silu",
                       is_gated: Optional[bool] = None,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
                       use_int8_w4a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       per_channel_quant: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       routed_scaling_factor: Optional[float] = 1.0,
                       gemm1_alpha: Optional[float] = None,
                       gemm1_limit: Optional[float] = None):

    return fused_experts_asm_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype,
        inplace,
        activation,
        is_gated,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w4a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
    )

@perftest(num_warmup=5, num_iters=10,testGraph=False)
def triton_fused_experts_impl(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       odtype,
                       inplace: bool = False,
                       activation: str = "silu",
                       is_gated: Optional[bool] = None,
                       b1: Optional[torch.Tensor] = None,
                       b2: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       use_int4_w4a8: bool = False,
                       per_channel_quant: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       no_combine: bool = False,
                       routed_scaling_factor: Optional[float] = 1.0,
                       gemm1_alpha: Optional[float] = None,
                       gemm1_limit: Optional[float] = None):

    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        odtype,
        inplace,
        activation,
        is_gated,
        b1,
        b2,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        use_int4_w4a8,
        per_channel_quant,
        global_num_experts,
        expert_map,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        no_combine,
        routed_scaling_factor,
        gemm1_alpha,
        gemm1_limit,
    )

@benchmark()
def test_fmoe_w8a8_channel_ep(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    has_zp,
    weight_bits,
    quant="No",
    use_g1u1=False,
    shared_E=2,
    ep=8,
):
    # This gpu id in EP, this example use the last id
    ep_id = ep - 1
    # total_expert = unshared_expert + shared_expert + fake_expert(only use this fake expert id to mask)
    # expert_mask = torch.randint(
    #     0, 2, (E + shared_E + 1,), dtype=dtypes.i32, device="cuda"
    # )
    expert_mask = torch.zeros((E + shared_E + 1,), dtype=dtypes.i32, device="cuda")
    expert_mask[ep_id * (E // ep) : (ep_id + 1) * E // ep] = 1

    # The last expert
    fake_expertid = expert_mask.numel() - 1
    # Ensure fake expert to be masked
    expert_mask[-1] = 0
    # Ensure shared expert not to be masked
    expert_mask[E:-1] = 1

    # # Get local expert Number in this gpu
    # local_E = 32
    local_E = torch.sum(expert_mask).item()

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda") / 10
    #only g1u1
    if use_g1u1:
        w1 = (
            torch.randn(
                (local_E, inter_dim * 2, model_dim),
                dtype=dtype,
                device="cuda",
            )
        )
    else:
        w1 = (
            torch.randn(
                (local_E, inter_dim, model_dim), dtype=dtype, device="cuda"
            )
            
        )
    w2 = (
        torch.randn(
            (local_E, model_dim, inter_dim), dtype=dtype, device="cuda"
        )
        
    )

    w1_ref = w1.clone()
    w2_ref = w2.clone()

    max_vals = torch.abs(w1.to(torch.float32)).max(dim=-1, keepdim=True)[0]
    #max_vals = torch.abs(w1).max(dim=-1, keepdim=True)[0]
    max_vals = max_vals.clamp(min=1e-5)
    w1_scales = max_vals / 127.0
    w1_qweight = (w1 / max_vals * 127.0).round().clamp(min=-128, max=127).to(torch.int8)

    max_vals = torch.abs(w2.to(torch.float32)).max(dim=-1, keepdim=True)[0]
    #max_vals = torch.abs(w2).max(dim=-1, keepdim=True)[0]
    max_vals = max_vals.clamp(min=1e-5)
    w2_scales = max_vals / 127.0
    w2_qweight = (w2 / max_vals * 127.0).round().clamp(min=-128, max=127).to(torch.int8)
    
    score = torch.randn((token, E), device="cuda", dtype=dtype)


    # if shared_E > 0:
    shared_E_score = 0.1
    # init total_topk_ids, inference time you just need to fill ns_topk_ids in total_topk_ids
    total_topk_ids = torch.empty(
        (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.i32, device=input.device
    )
    ns_topk_ids, s_topk_ids = total_topk_ids.split([topk, shared_E + 1], dim=1)
    shared_expert_ids = [E + i for i in range(shared_E + 1)]
    s_topk_ids_list = [[fake_expertid] * (shared_E + 1)] * MAX_TOKENS
    for i in range(ep_id, MAX_TOKENS, ep):
        s_topk_ids_list[i] = shared_expert_ids
    s_topk_ids[:] = torch.tensor(s_topk_ids_list, dtype=dtypes.i32, device=input.device)

    # init total_topk_weights, inference time you just need to fill ns_topk_weights in total_topk_weights
    total_topk_weights = torch.empty(
        (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.fp32, device=input.device
    )
    ns_topk_weights, s_topk_weights = total_topk_weights.split(
        [topk, shared_E + 1], dim=1
    )
    s_topk_weights[:] = shared_E_score

    # print(f"ns_topk_ids:{ns_topk_ids.shape}  ns_topk_weights:{ns_topk_weights.shape}")

    # inference time, use fused_topk to fill ns_topk_ids and ns_topk_weights
    fused_topk(input, score, topk, True, ns_topk_ids, ns_topk_weights)
    # inference time, topk_ids simply slices total_topk_ids into the number of input tokens, same for topk_weights
    topk_ids = total_topk_ids[:token]
    topk_weights = total_topk_weights[:token]

    # reference golden

    ############################ Triton Solution ################################################

    # if ep > 1:
    #     # indices = torch.arange(expert_mask.numel(), dtype=dtypes.i32, device="cuda")
    #     # indices = indices -(ep_id * (E // ep))
    #     indices = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
    #     e_map = torch.where(expert_mask == 0, torch.tensor(-1, dtype=dtypes.i32, device="cuda"), expert_mask)
    #     e_map = torch.where(e_map == 1, indices, e_map)
    # else:
    #     # 若e_map=None, 而topk_ids中确存在fake expert id，triton无法通过e_map屏蔽掉fake expert的计算，结果不正确
    #     e_map = None

    indices = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
    e_map = torch.where(
        expert_mask == 0,
        torch.full_like(expert_mask, -1, dtype=dtypes.i32),
        expert_mask,
    )
    e_map = torch.where(e_map == 1, indices, e_map)     # 有效专家映射到本地专家id，无效专家映射到-1


    torch_moe_golden, avg_torch = torch_moe_test(
        input, w1_ref, w2_ref, topk_weights, topk_ids, expert_mask=expert_mask
    )

    input_q, input_scale = per_token_quant_hip(input,quant_dtype=torch.int8)
    triton_output, avg_triton = triton_fused_experts_impl(
        input_q,
        w1_qweight,
        w2_qweight,
        topk_weights,
        topk_ids,
        dtype,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        use_int4_w4a8=False,
        per_channel_quant=True,
        global_num_experts=E + shared_E,
        expert_map=e_map,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=None,
        w2_zp=None,
        a1_scale=input_scale,
        a2_scale=None,
        block_shape=None)
    msg = f"[TRITON_perf] {token=}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_torch:<8.2f} us, triton_avg: {avg_triton:>8.2f} us,uplift: {avg_torch/avg_triton-1:.1%}"
    checkAllclose(torch_moe_golden, triton_output, rtol=0.01, atol=100, msg=msg)


    ###################################### ASM Solution #########################################
    # input_q, input_scale = per_token_quant_hip(input,quant_dtype=torch.int8)
    asm_output, avg_asm = asm_fused_experts_impl(
        input_q,
        w1_qweight,
        w2_qweight,
        topk_weights,
        topk_ids,
        dtype,
        False,
        activation="silu",
        use_int8_w8a8=True,
        global_num_experts=E + shared_E,
        expert_map=expert_mask,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        per_channel_quant=True,
        a1_scale=input_scale)
    msg = f"[ASM_perf] {token=}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_torch:<8.2f} us, asm_avg: {avg_asm:>8.2f} us,uplift: {avg_torch/avg_asm-1:.1%}"
    checkAllclose(torch_moe_golden, asm_output, rtol=0.01, atol=100, msg=msg)
    return {
        "triton_us": avg_triton,
        "asm_us": avg_asm
    }

df = []

for dtype in [dtypes.fp16]:
    for m in [8,16,32,64,128]:
        for dim in [7168]:
            for hdim in [256]:
                for ep in [16, 1]:
                    for sh_e in [0, 1, 2]:
                        ret = test_fmoe_w8a8_channel_ep(dtype, m, dim ,hdim, 256,8, True, 4,quant="No", use_g1u1=True, shared_E=sh_e, ep=ep)
                        df.append(ret)

df = pd.DataFrame(df)
df.to_csv("moe_w8a8_perToken_int8_ep.csv")
aiter.logger.info(f"summary:\n{df}")

