# SPDX-License-Identifier: MIT
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
from aiter.test_common import checkAllclose, perftest
from aiter import ck_moe, ck_shuffle_moe, dtypes, silu_and_mul, gelu_and_mul, moe_sum
from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config_func
from aiter.ops.triton.fused_moe import fused_experts_impl
from aiter.ops.triton.utils.types import torch_to_triton_dtype
from aiter.fused_moe_asm_wna16 import fused_experts_asm_impl
import pandas as pd
import aiter
import os
os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"


BLOCK_SIZE_M = 32
MAX_TOKENS = 65536


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

@perftest(num_warmup=5, num_iters=10,testGraph=True)
def triton_fused_experts_impl(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       dtype,
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
        dtype,
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
        False,
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

@perftest(num_warmup=5, num_iters=10,testGraph=True)
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

def test_fmoe_wn16_ep(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    group_size,
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


    if weight_bits == 4:
        pack_factor = 2
        quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    elif weight_bits == 8:
        pack_factor = 1
        quant_type = scalar_types.uint8 if has_zp else scalar_types.uint8b128

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda") / 10
    #only g1u1
    if use_g1u1:
        w1 = (
            torch.randn(
                (local_E, inter_dim * 2, model_dim),
                dtype=dtype,
                device="cuda",
            )
            / 10
        )
    else:
        w1 = (
            torch.randn(
                (local_E, inter_dim, model_dim), dtype=dtype, device="cuda"
            )
            / 10
        )
    w2 = (
        torch.randn(
            (local_E, model_dim, inter_dim), dtype=dtype, device="cuda"
        )
        / 10
    )

    w1_ref = w1.clone()
    w2_ref = w2.clone()
    w1_qweight = torch.empty((local_E, 2 * inter_dim, model_dim // pack_factor),
                             device="cuda",
                             dtype=torch.uint8)
    w2_qweight = torch.empty((local_E, model_dim, inter_dim // pack_factor),
                             device="cuda",
                             dtype=torch.uint8)
    w1_scales = torch.empty((local_E, 2 * inter_dim, model_dim // group_size),
                            device="cuda",
                            dtype=dtype)
    w2_scales = torch.empty((local_E, model_dim, inter_dim // group_size),
                            device="cuda",
                            dtype=dtype)
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
    w1_qzeros = torch.empty((local_E , 2 * inter_dim// pack_factor, model_dim// group_size ),
                            device="cuda",
                            dtype=torch.uint8)
    w2_qzeros = torch.empty((local_E, model_dim // pack_factor, inter_dim // group_size),
                            device="cuda",
                            dtype=torch.uint8)
    for i in range(local_E * 2):
        expert_id = i % local_E
        if i // local_E == 0:
            w, w_ref, w_qweight, w_scales, w_qzeros = \
                w1, w1_ref, w1_qweight, w1_scales, w1_qzeros
        else:
            w, w_ref, w_qweight, w_scales, w_qzeros = \
                w2, w2_ref, w2_qweight, w2_scales, w2_qzeros
        weight, qweight, scales, qzeros = quantize_weights(
            w[expert_id].T, quant_type, group_size, has_zp, False)
        weight = weight.T
        qweight = qweight.T.contiguous().to(torch.uint8)
        scales = scales.T
        if has_zp:
            qzeros = qzeros.T.contiguous().to(torch.uint8)
        if weight_bits == 4:
            qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]   # 偶数列存储低4位，奇数列存储高4位
            if has_zp:
                qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]
        w_ref[expert_id] = weight
        w_qweight[expert_id] = qweight
        w_scales[expert_id] = scales
        if has_zp:
            w_qzeros[expert_id] = qzeros


    if ep > 1:
        # indices = torch.arange(expert_mask.numel(), dtype=dtypes.i32, device="cuda")
        # indices = indices -(ep_id * (E // ep))
        indices = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        e_map = torch.where(expert_mask == 0, torch.tensor(-1, dtype=dtypes.i32, device="cuda"), expert_mask)
        e_map = torch.where(e_map == 1, indices, e_map)

    torch_moe_golden, avg_torch = torch_moe_test(
        input, w1_ref, w2_ref, topk_weights, topk_ids, expert_mask=expert_mask
    )
    triton_output, avg_triton= triton_fused_experts_impl(
        input,
        w1_qweight,
        w2_qweight,
        topk_weights,
        topk_ids,
        dtype,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        use_int4_w4a8=False,
        global_num_experts=E + shared_E + 1,
        expert_map=e_map,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=w1_qzeros if has_zp else None,
        w2_zp=w2_qzeros if has_zp else None,
        a1_scale=None,
        a2_scale=None,
        block_shape=[0, group_size])
    msg = f"[TRITON_perf] {token=}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_torch:<8.2f} us, triton_avg: {avg_triton:>8.2f} us,uplift: {avg_torch/avg_triton-1:.1%}"
    checkAllclose(torch_moe_golden, triton_output, rtol=0.01, atol=0.01, msg=msg)


    ###################################### ASM Solution #########################################
    w1_qzeros = torch.empty((local_E , 2 * inter_dim, model_dim// group_size // pack_factor),
                            device="cuda",
                            dtype=torch.uint8)
    w2_qzeros = torch.empty((local_E, model_dim, inter_dim // group_size // pack_factor),
                            device="cuda",
                            dtype=torch.uint8)
    for i in range(local_E * 2):
        expert_id = i % local_E
        if i // local_E == 0:
            w, w_ref, w_qweight, w_scales, w_qzeros = \
                w1, w1_ref, w1_qweight, w1_scales, w1_qzeros
        else:
            w, w_ref, w_qweight, w_scales, w_qzeros = \
                w2, w2_ref, w2_qweight, w2_scales, w2_qzeros
        weight, qweight, scales, qzeros = quantize_weights(
            w[expert_id].T, quant_type, group_size, has_zp, False)
        weight = weight.T
        qweight = qweight.T.contiguous().to(torch.uint8)
        scales = scales.T
        if has_zp:
            qzeros = qzeros.T.contiguous().to(torch.uint8)
        if weight_bits == 4:
            qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]   # 偶数列存储低4位，奇数列存储高4位
            if has_zp:
                qzeros = qzeros[:, 1::2] * 16 + qzeros[:, ::2]
        w_ref[expert_id] = weight
        w_qweight[expert_id] = qweight
        w_scales[expert_id] = scales
        if has_zp:
            w_qzeros[expert_id] = qzeros


    asm_output, avg_asm = asm_fused_experts_impl(
        input,
        w1_qweight,
        w2_qweight,
        topk_weights,
        topk_ids,
        dtype,
        False,
        activation="silu",
        use_int4_w4a16=True,
        global_num_experts=E+shared_E+1,
        expert_map=expert_mask,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=w1_qzeros if has_zp else None,
        w2_zp=w2_qzeros if has_zp else None,
        block_shape=[0, group_size])
    msg = f"[ASM_perf] {token=}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_torch:<8.2f} us, asm_avg: {avg_asm:>8.2f} us,uplift: {avg_torch/avg_asm-1:.1%}"
    checkAllclose(torch_moe_golden, asm_output, rtol=0.01, atol=0.01, msg=msg)

    return {
        "triton_us": avg_triton,
        "asm_us": avg_asm
    }

df = []

for dtype in [dtypes.fp16]:
    for m in [8,16,32,64,128]:
        for dim in [7168]:
            for hdim in [2048]:
                for ep in [16]:
                    ret = test_fmoe_wn16_ep(dtype, m, dim ,hdim, 256,8, 64, True, 4,quant="No", use_g1u1=True, shared_E=0, ep=ep)
                    df.append(ret)

df = pd.DataFrame(df)
df.to_csv("moe_wna16_ep.csv")
aiter.logger.info(f"summary:\n{df}")
