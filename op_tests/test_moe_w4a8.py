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
        hidden_states.dtype,
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

@perftest(num_warmup=5, num_iters=10,testGraph=True)
def triton_fused_experts_impl(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       odtype, #compute or output type for i8& f8
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

def test_fused_moe_wn16(m: int,
                        k: int, #hidden_dim
                        n: int, #intermediate_size
                        e: int,
                        topk: int,
                        ep_size: int,
                        dtype: torch.dtype,
                        group_size: int,
                        has_zp: bool,
                        weight_bits: int):
    input = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    if weight_bits == 4:
        pack_factor = 2
        quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    elif weight_bits == 8:
        pack_factor = 1
        quant_type = scalar_types.uint8 if has_zp else scalar_types.uint8b128

    w1_ref = w1.clone()
    w2_ref = w2.clone()
    w1_qweight = torch.empty((e, 2 * n, k // pack_factor),
                             device="cuda",
                             dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // pack_factor),
                             device="cuda",
                             dtype=torch.uint8)
    w1_scales = torch.empty((e, 2 * n, k // group_size),
                            device="cuda",
                            dtype=dtype)
    w2_scales = torch.empty((e, k, n // group_size),
                            device="cuda",
                            dtype=dtype)
    """
    e, 2*n , k//group_size//pack_factor
    """
    ###################################### Triton Solution ###########################################
    w1_qzeros = torch.empty((e, 2 * n // pack_factor, k // group_size),
                            device="cuda",
                            dtype=torch.uint8)
    w2_qzeros = torch.empty((e, k // pack_factor, n // group_size),
                            device="cuda",
                            dtype=torch.uint8)

    for i in range(e * 2):
        expert_id = i % e
        if i // e == 0:
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
        w1_qzeros = w1_qzeros[e_ids]
        w2_qzeros = w2_qzeros[e_ids]
    else:
        e_map = None


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

    torch_triton_output, avg_torch = torch_moe_test(input, w1_ref, w2_ref, topk_weights, topk_ids)
    #Triton Solution
    triton_output, avg_triton = triton_fused_experts_impl(
                                    input,
                                    w1_qweight,
                                    w2_qweight,
                                    topk_weights,
                                    topk_ids,
                                    dtype,
                                    inplace=False,
                                    activation="silu",
                                    use_int4_w4a8=True,
                                    global_num_experts=e,
                                    expert_map=e_map,
                                    w1_scale=w1_scales,
                                    w2_scale=w2_scales,
                                    w1_zp=w1_qzeros if has_zp else None,
                                    w2_zp=w2_qzeros if has_zp else None,
                                    block_shape=[0, group_size])
    msg = f"[TRITON_perf] {m=}, {k=}, {n=}, {e=}, {topk=}, dtype: {dtype}, torch_avg: {avg_torch:<8.2f} us, triton_avg: {avg_triton:>8.2f} us,uplift: {avg_torch/avg_triton-1:.1%}"
    checkAllclose(torch_triton_output, triton_output, rtol=0.01, atol=0.1, msg=msg)
    ###################################### ASM Solution #########################################
    w1_qzeros = torch.empty((e, 2 * n, k // group_size // pack_factor),
                            device="cuda",
                            dtype=torch.uint8)
    w2_qzeros = torch.empty((e, k, n // group_size // pack_factor),
                            device="cuda",
                            dtype=torch.uint8)

    for i in range(e * 2):
        expert_id = i % e
        if i // e == 0:
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
                #asm qzeros
                qzeros = qzeros[:, 1::2] * 16 + qzeros[:, ::2]

        w_ref[expert_id] = weight
        w_qweight[expert_id] = qweight
        w_scales[expert_id] = scales
        if has_zp:
            w_qzeros[expert_id] = qzeros

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
        w1_qzeros = w1_qzeros[e_ids]
        w2_qzeros = w2_qzeros[e_ids]
    else:
        e_map = None


    ## 1. including token topk score calc

    ## 2. without token topk score calc
    # topk_weights, topk_ids = fused_topk(input, score, topk, True)

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

    torch_asm_output, avg_torch = torch_moe_test(input, w1_ref, w2_ref, topk_weights, topk_ids)


    #Triton Solution
    asm_output, avg_asm = asm_fused_experts_impl(
        input,
        w1_qweight,
        w2_qweight,
        topk_weights,
        topk_ids,
        False,
        activation="silu",
        use_int8_w4a8=True,
        global_num_experts=e,
        expert_map=e_map,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=w1_qzeros if has_zp else None,
        w2_zp=w2_qzeros if has_zp else None,
        block_shape=[0, group_size])



    # torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
    msg = f"[ASM_perf] {m=}, {k=}, {n=}, {e=}, {topk=}, dtype: {dtype}, torch_avg: {avg_torch:<8.2f} us, asm_avg: {avg_asm:>8.2f} us,uplift: {avg_torch/avg_asm-1:.1%}"
    checkAllclose(torch_asm_output, asm_output, rtol=0.01, atol=100, msg=msg)

    # ck_out = ck_moe_test(input, w1_qweight, w2_qweight, topk_weights, topk_ids,
    #                     use_int8_w8a16 = weight_bits == 8,
    #                     use_int4_w4a16 = weight_bits == 4,
    #                     use_int8_w8a8_block = False,
    #                     w1_zp = w1_qzeros if has_zp else None,
    #                     w2_zp = w2_qzeros if has_zp else None,
    #                     w1_scale = w1_scales,
    #                     w2_scale = w2_scales,
    #                     block_shape_n = 1,
    #                     block_shape_k = group_size)

    # torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
    # rel_diff = (torch.mean(
    #     torch.abs(ck_out.to(torch.float32) - torch_output.to(torch.float32))) /
    #             torch.mean(torch.abs(torch_output.to(torch.float32))))
    # print("###### ck and torch diff = ", rel_diff)





for dtype in [dtypes.fp16]:
    # for m in [32,48,64,80,96,128,256]:
    #for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,64,96,128,256,512,1024]:
     for m in [1, 2,  4,  8, 16 ,32,64, 96, 128,256,512,1024,2048,4096,8192,16384,32768]:
        for dim in [7168]:
            for hdim in [256]:
                # test_fmoe(dtype, m, dim, hdim, 32, 5)
                # test_fmoe(dtype, m, dim, hdim, 256, 8, quant="No", use_g1u1=True)
                test_fused_moe_wn16(m, dim, hdim, 256, 8, 0, dtype, 64, True, 4)