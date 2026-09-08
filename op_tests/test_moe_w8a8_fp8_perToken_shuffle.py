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
from aiter import per_token_quant_hip, per_block_quant_wrapper
from aiter import pertoken_quant
from aiter.ops.shuffle import asm_shuffle_weight_b8
import pandas as pd
import aiter
import os
os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
#Notice: Adjust benchmark block_size_m here
BLOCK_SIZE_M_CK = 16
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
                       block_shape: Optional[list[int]] = None,
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
        use_shuffle=use_shuffle
        #solution_id="13000+23101"
    )

@perftest(num_warmup=5, num_iters=10,testGraph=True)
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

def test_fused_moe_w8a8(m: int, 
                        k: int, #hidden_dim
                        n: int, #intermediate_size
                        e: int, 
                        topk: int,
                        ep_size: int, 
                        dtype: torch.dtype, 
                        weight_bits: int):
    torch.manual_seed(0)
    input = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype)
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype)
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    w1_ref = w1.clone()
    w2_ref = w2.clone()

    w1_qweight,w1_scales = pertoken_quant(w1, quant_dtype=dtypes.fp8)
    w2_qweight,w2_scales = pertoken_quant(w2,  quant_dtype=dtypes.fp8)

    # max_vals = torch.abs(w1.to(torch.float32)).max(dim=-1, keepdim=True)[0]
    #max_vals = torch.abs(w1).max(dim=-1, keepdim=True)[0]
    # max_vals = max_vals.clamp(min=1e-5)
    # w1_scales = max_vals / 127.0
    # w1_qweight = (w1 / max_vals * 127.0).round().clamp(min=-128, max=127).to(torch.int8)

    # max_vals = torch.abs(w2.to(torch.float32)).max(dim=-1, keepdim=True)[0]
    # #max_vals = torch.abs(w2).max(dim=-1, keepdim=True)[0]
    # max_vals = max_vals.clamp(min=1e-5)
    # w2_scales = max_vals / 127.0
    # w2_qweight = (w2 / max_vals * 127.0).round().clamp(min=-128, max=127).to(torch.int8)

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randint(0,
                              e, (local_e, ),
                              device="cuda",
                              dtype=torch.int32)
        e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1_ref = w1_ref[e_ids]
        w2_ref = w2_ref[e_ids]
        w1_qweight = w1_qweight[e_ids]
        w2_qweight = w2_qweight[e_ids]
        w1_scales = w1_scales[e_ids]
        w2_scales = w2_scales[e_ids]
    else:
        e_map = None


    w1_qweight_shuffle = asm_shuffle_weight_b8(w1_qweight, 1)
    w2_qweight_shuffle = asm_shuffle_weight_b8(w2_qweight, 2)
    ## 1. including token topk score calc
    
    ## 2. without token topk score calc
    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    
    # debug purpose
    """
    print("###### topk_weights dtype = {}, shape = {}".format(topk_weights.dtype, topk_weights.shape))
    print("###### topk_ids dtype = {}, shape = {}".format(topk_ids.dtype, topk_ids.shape))
    print("###### w1_qweight dtype = {}, shape = {}".format(w1_qweight.dtype, w1_qweight.shape))
    print("###### w2_qweight dtype = {}, shape = {}".format(w2_qweight.dtype, w2_qweight.shape))
    print("###### w1_scales dtype = {}, shape = {}".format(w1_scales.dtype, w1_scales.shape))
    print("###### w2_scales dtype = {}, shape = {}".format(w2_scales.dtype, w2_scales.shape))
    print("###### w1_qzeros dtype = {}, shape = {}".format(w1_qzeros.dtype if has_zp else None, w1_qzeros.shape if has_zp else None))
    print("###### w2_qzeros dtype = {}, shape = {}".format(w2_qzeros.dtype if has_zp else None, w2_qzeros.shape if has_zp else None))
    print("###### w1_ref dtype = {}, shape = {}".format(w1_ref.dtype, w1_ref.shape))
    print("###### w2_ref dtype = {}, shape = {}".format(w2_ref.dtype, w2_ref.shape))
    """
    
    torch_output, avg_torch = torch_moe_test(input, w1_ref, w2_ref, topk_weights, topk_ids)


    #Triton Solution
    #input_q, input_scale = per_token_quant_hip(input,quant_dtype=torch.float8_e4m3fn)
    triton_output, avg_triton = triton_fused_experts_impl(
        input,
        w1_qweight,
        w2_qweight,
        topk_weights,
        topk_ids,
        dtype,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        use_int4_w4a8=False,
        per_channel_quant=True,
        global_num_experts=e,
        expert_map=e_map,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None)

    asm_output, avg_asm = asm_fused_experts_impl(
        input,
        w1_qweight,
        w2_qweight,
        topk_weights,
        topk_ids,
        dtype,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w4a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        global_num_experts=e,
        expert_map=e_map,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        use_shuffle=0)

    asm_shuffle_output, avg_shuffle_asm = asm_fused_experts_impl(
        input,
        w1_qweight_shuffle,
        w2_qweight_shuffle,
        topk_weights,
        topk_ids,
        dtype,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w4a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        global_num_experts=e,
        expert_map=e_map,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        use_shuffle=1)
 
    #msg = f"[TRITON_perf] {m=}, {k=}, {n=}, {e=}, {topk=}, dtype: {dtype}, triton_avg: {avg_triton:>8.2f} us"
    # print(ref_out,triton_output)
    #checkAllclose(ref_out, triton_output, rtol=0.01, atol=100, msg=msg)
    
    # torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
    msg = f"[ASM_perf] {m=}, {k=}, {n=}, {e=}, {topk=}, dtype: {dtype}, asm_avg: {avg_asm:>8.2f} us"
    checkAllclose(triton_output, asm_output, rtol=0.01, atol=0.01, msg=msg)

    msg = f"[ASM_shuffle_perf] {m=}, {k=}, {n=}, {e=}, {topk=}, dtype: {dtype}, avg_shuffle_asm: {avg_shuffle_asm:>8.2f} us"
    checkAllclose(asm_output, asm_shuffle_output, rtol=0.01, atol=0.01, msg=msg)
    #torch.set_printoptions(threshold=10_000)
    #print("golden",asm_output)
    #print("out", asm_shuffle_output)
    return{
        #"triton_us": avg_triton,
        "m": m,
        "asm_us": avg_asm,
        "asm_shuffle_us": avg_shuffle_asm,
        "shuffle_uplight":f"{avg_asm / avg_shuffle_asm*100:.2f}%"
    }


df = []



for dtype in [dtypes.fp16]:
    # for m in [1]:
    #for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,64,96,128,256,512,1024]:
    for m in [1, 2,  4,  8, 16 ,32,64, 96, 128,256,512,1024,2048,4096,8192,16384,32768]:
        for dim in [7168]:
            for hdim in [128]:
                # test_fmoe(dtype, m, dim, hdim, 32, 5)
                # test_fmoe(dtype, m, dim, hdim, 256, 8, quant="No", use_g1u1=True)
                ret = test_fused_moe_w8a8(m, dim, hdim, 256, 8, 0, dtype, 8)
                df.append(ret)
df = pd.DataFrame(df)
df.to_csv("moe_w8a8_perToken_fp8.csv")
aiter.logger.info(f"summary:\n{df}")
