import pytest
import torch
import itertools
from typing import Optional, List  # Add this import at the top

from op_tests.utility.utils import torch_moe as torch_score_moe
from aiter.ops.triton.moe_op import fused_moe
from aiter.fused_moe import fused_topk, torch_moe
from aiter import ActivationType, dtypes
from aiter.test_common import checkAllclose, perftest,benchmark
from aiter.ops.triton.fused_moe import fused_experts_impl
from aiter.fused_moe_asm_wna16 import fused_experts_asm_impl
from aiter.ops.shuffle import asm_shuffle_weight_b8
import pandas as pd
import aiter
import os
os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
os.environ["TRITON_FUSED_MOE_CHUNK_SIZE"] = "16384"
torch.set_printoptions(profile="full")  # 完整打印


@perftest(num_warmup=1, num_iters=2)
def torch_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    w1_scale=None,  # [expert, inter_dim, 1]
    w2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    activation=ActivationType.Silu,
):
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        w1_scale,
        w2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        None,
        activation,
    )

@perftest(num_warmup=2, num_iters=10,testGraph=False)
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
                       use_shuffle: Optional[int] = 0):

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
        use_shuffle = use_shuffle
        #solution_id="10002+20000"
    )

@perftest(num_warmup=2, num_iters=10,testGraph=False)
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
def test_fused_moe(m: int,
                   k: int, #hidden_dim
                   n: int, #intermediate_size
                   e: int,
                   topk: int,
                   ep_size: int,
                   dtype: torch.dtype):

    torch.manual_seed(0)
    input = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype)
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype)
    w1_shuffle = asm_shuffle_weight_b8(w1, stage=1)
    w2_shuffle = asm_shuffle_weight_b8(w2, stage=2)
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randint(0,
                              e, (local_e, ),
                              device="cuda",
                              dtype=torch.int32)
        e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None


    ## 1. including token topk score calc

    ## 2. without token topk score calc
    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    torch_output, avg_torch = torch_moe_test(input, w1, w2, topk_weights, topk_ids)


    #Triton Solution
    triton_output, avg_triton = triton_fused_experts_impl(
        input,
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        use_int4_w4a8=False,
        per_channel_quant=False,
        global_num_experts=e,
        expert_map=e_map,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None)

    asm_output, avg_asm = asm_fused_experts_impl(
        input,
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype,
        False,
        activation="silu",
        global_num_experts=e,
        expert_map=e_map)

    asm_output_shuffle, avg_asm_shuffle = asm_fused_experts_impl(
        input,
        w1_shuffle,
        w2_shuffle,
        topk_weights,
        topk_ids,
        dtype,
        False,
        activation="silu",
        global_num_experts=e,
        expert_map=e_map,
        use_shuffle=1)

    msg = f"[TRITON_perf] {m=}, {k=}, {n=}, {e=}, {topk=}, dtype: {dtype}, torch_avg: {avg_torch:<8.2f} us, triton_avg: {avg_triton:>8.2f} us,uplift: {avg_torch/avg_triton-1:.1%}"
    checkAllclose(torch_output, triton_output, rtol=0.01, atol=1, msg=msg)
    #torch.set_printoptions(threshold=10_000)
    #print("golden",triton_output)
    #print("out", asm_output)
    # torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
    msg = f"[ASM_perf] {m=}, {k=}, {n=}, {e=}, {topk=}, dtype: {dtype}, torch_avg: {avg_torch:<8.2f} us, asm_avg: {avg_asm:>8.2f} us,uplift: {avg_torch/avg_asm-1:.1%}"
    checkAllclose(triton_output, asm_output, rtol=0.01, atol=0.01, msg=msg)
    msg = f"[ASM_shuffle_perf] {m=}, {k=}, {n=}, {e=}, {topk=}, dtype: {dtype}, asm_avg: {avg_asm:<8.2f} us, asm_shuffle_avg: {avg_asm_shuffle:>8.2f} us,uplift: {avg_asm/avg_asm_shuffle-1:.1%}"
    checkAllclose(asm_output_shuffle, asm_output, rtol=0.01, atol=0.01, msg=msg)
    return{
        "triton_us": avg_triton,
        "asm_us": avg_asm,
        "asm_shuffle_us": avg_asm_shuffle,
        "shuffle_uplift": f"{avg_asm/avg_asm_shuffle-1:.1%}"
    }

df = []
for dtype in [dtypes.bf16]:
    for m in [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    # for m in [4096]:
        for dim in [4096]:
            for hdim in [352]:
                ret = test_fused_moe(m, dim, hdim, 128, 8, 0, dtype)
                df.append(ret)
df = pd.DataFrame(df)
df.to_csv("moe_w16a16.csv")
aiter.logger.info(f"summary:\n{df}")