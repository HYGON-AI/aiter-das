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
from aiter.fused_moe_ck import ck_fused_experts_2stage_impl
from op_tests.utility.utils import native_w8a8_block_matmul, silu_and_mul
from einops import rearrange
import torch.nn.functional as F
from aiter import per_token_quant_hip, per_block_quant_wrapper
from aiter import pertoken_quant
from aiter.ops.shuffle import asm_shuffle_weight_b8
import pandas as pd
import aiter
import os
os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

#Notice: Adjust benchmark block_size_m here
BLOCK_SIZE_M_CK = 16
# torch.set_printoptions(profile="full")  # 完整打印


# For test
def native_per_token_group_quant_int8(x,
                                      group_size,
                                      eps=1e-10,
                                      dtype=torch.int8):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch.

    It converts the tensor values into int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    """
    assert (x.shape[-1] % group_size == 0
            ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_min = iinfo.min
    int8_max = iinfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    # Use float32 for scale calculation for stability
    amax = x_.abs().max(dim=-1,
                        keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / int8_max
    x_q = (x_.to(torch.float32) / x_s).round().clamp(
        min=int8_min, max=int8_max).to(dtype)  # Round before clamping
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size, ))

    return x_q, x_s


# For test
def torch_moe_blockscale(
    hidden_states,
    w1,  # [expert, inter_dim*2, model_dim]
    w2,  # [expert, model_dim, inter_dim]
    topk_weight,
    topk_ids,
    dtype,
    # following for quant
    scale_blks=(128, 128),
    a_scale=None,
    # [expert, inter_dim/blk_m, model_dim/blk_k]
    fc1_scale=None,
    # [expert, model_dim/blk_m, inter_dim/blk_k]
    fc2_scale=None,
    expert_mask=None,
    computeType=torch.float32,
):
    hidden_states = hidden_states.float().to(computeType)
    w1 = w1.float().to(computeType)
    w2 = w2.float().to(computeType)
    token_num, topk = topk_ids.shape
    expert, model_dim, inter_dim = w2.shape
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    blk_n, blk_k = scale_blks
    if a_scale is not None:
        # print(f'{a_scale.unsqueeze(-1).shape=}, {hidden_states.view(token_num, -1, blk_k).shape=}')
        hidden_states = hidden_states.view(token_num, -1, blk_k) * a_scale.unsqueeze(-1)
        hidden_states = hidden_states.view(token_num, -1)

    hidden_states = hidden_states.view(token_num, 1, model_dim).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=computeType,
        device=hidden_states.device,
    )
    if w2.shape[2] * 2 == w1.shape[1]:
        moeType = "g1u1"
    else:
        moeType = "g1u0"

    nblk_n = inter_dim // blk_n
    nblk_k = model_dim // blk_k
    if fc1_scale is not None:
        fc1_scale = fc1_scale.to(computeType)
        fc2_scale = fc2_scale.to(computeType)
        fc1_scale = rearrange(
            fc1_scale.view(-1, 1)
            .repeat(1, blk_n * blk_k)
            .view(expert, -1, nblk_k, blk_n, blk_k),
            "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
        )
        fc2_scale = rearrange(
            fc2_scale.view(-1, 1)
            .repeat(1, blk_n * blk_k)
            .view(expert, nblk_k, nblk_n, blk_k, blk_n),
            "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
        )
        w1 = w1 * fc1_scale
        w2 = w2 * fc2_scale

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                act_out = F.silu(gate) * up
            else:
                act_out = F.gelu(act_input)
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)





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
                       block_shape: Optional[list[int]] = None,
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
        block_shape
        #block_shape,
        #solution_id="10006+20000"
    )

@perftest(num_warmup=5, num_iters=10,testGraph=False)
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

@perftest(num_warmup=5, num_iters=10,testGraph=False)
def ck_fused_experts(hidden_states: torch.Tensor,
                    w1: torch.Tensor,
                    w2: torch.Tensor,
                    topk_weights: torch.Tensor,
                    topk_ids: torch.Tensor,
                    dtype,
                    inplace: bool = False,
                    activation: str = "silu",
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
                    block_shape: Optional[List[int]] = None):
    return ck_fused_experts_2stage_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype,
        inplace,
        activation,
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
        block_shape
    )

@benchmark()
def test_fused_moe_w8a8(M: int, 
                        K: int, #hidden_dim
                        N: int, #intermediate_size
                        E: int, 
                        topk: int,
                        ep_size: int, 
                        dtype: torch.dtype, 
                        weight_bits: int,
                        block_size:list):


    """Tests the fused_moe kernel with W8A8 INT8 block quantization against a
    native torch reference."""

    torch.manual_seed(0)
    input = torch.randn((M, K), dtype=dtype, device="cuda") / 10

    w1 = torch.rand((E, 2 * N, K), dtype=dtype, device="cuda") 
    w2 = torch.rand((E, K, N), dtype=dtype, device="cuda")

    scale_blk_n, scale_blk_k = block_size[0], block_size[1]
    quant_dtype = dtypes.fp8
    tmp = rearrange(
        w1.view(
            -1,
            w1.shape[1] // scale_blk_n,
            scale_blk_n,
            w1.shape[2] // scale_blk_k,
            scale_blk_k,
        ),
        "e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)",
    ).contiguous()
    w1_qweight, w1_scales = pertoken_quant(tmp, quant_dtype=quant_dtype)
    w1_qweight = rearrange(
        w1_qweight.view(
            -1,
            w1.shape[1] // scale_blk_n,
            w1.shape[2] // scale_blk_k,
            scale_blk_n,
            scale_blk_k,
        ),
        "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
    ).contiguous()
    #w1_qweight_shuffle = asm_shuffle_weight_b8(w1_qweight)
    w1_scales = w1_scales.view(E, w1_scales.shape[1], w1_scales.shape[2])

    # block quant w2
    tmp = rearrange(
        w2.view(
            -1,
            w2.shape[1] // scale_blk_n,
            scale_blk_n,
            w2.shape[2]// scale_blk_k,
            scale_blk_k,
        ),
        "e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)",
    ).contiguous()
    w2_qweight, w2_scales = pertoken_quant(tmp, quant_dtype=quant_dtype)
    w2_qweight = rearrange(
        w2_qweight.view(
            -1,
            w2.shape[1] // scale_blk_n,
            w2.shape[2] // scale_blk_k,
            scale_blk_n,
            scale_blk_k,
        ),
        "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
    ).contiguous()
    w2_scales = w2_scales.view(E, w2_scales.shape[1], w2_scales.shape[2])

    score = torch.randn((M, E), dtype=dtype, device="cuda")

    if ep_size > 1:
        local_e = E // ep_size
        e_ids = torch.randint(0,
                              E, (local_e, ),
                              device="cuda",
                              dtype=torch.int32)
        e_map = torch.full((E, ), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1_ref = w1_ref[e_ids]
        w2_ref = w2_ref[e_ids]
        w1_qweight = w1_qweight[e_ids]
        w2_qweight = w2_qweight[e_ids]
        w1_scales = w1_scales[e_ids]
        w2_scales = w2_scales[e_ids]
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
    
    # ref_out, ref_quant, ref_scale, act_out = torch_w8a8_block_int8_moe(input, w1_qweight, w2_qweight, w1_scales, w2_scales, topk_weights,  topk_ids, topk, block_size)
    ref_out = torch_moe_blockscale(input, w1_qweight, w2_qweight, topk_weights, topk_ids, dtype, block_size,None, w1_scales,w2_scales,e_map, computeType=torch.float16)

    #Triton Solution
    #input_q, input_scale = per_block_quant_wrapper((1,block_size[1]))(per_token_quant_hip)(input)
    triton_output, avg_triton = triton_fused_experts_impl(
        input,
        w1_qweight,
        w2_qweight,
        topk_weights,
        topk_ids,
        dtype,
        False,
        "silu",
        None,
        None,
        None,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        E,
        e_map,
        w1_scales,
        w2_scales,
        None,
        None,
        None,
        None,
        (128,128))

    asm_output, avg_asm = asm_fused_experts_impl(
        input,
        w1_qweight,#w1_qweight_shuffle,
        w2_qweight,
        topk_weights,
        topk_ids,
        dtype,
        False,
        "silu",
        None,
        True,
        False,
        False,
        False,
        False,
        False,
        E,
        e_map,
        w1_scales,
        w2_scales,
        None,
        None,
        None,
        None,
        (128,128))
    
    ck_output, avg_ck = ck_fused_experts(
        input,
        w1_qweight,
        w2_qweight,
        topk_weights,
        topk_ids,
        dtype,
        False,
        "silu",
        True,
        False,
        False,
        False,
        False,
        False,
        E,
        e_map,
        w1_scales,
        w2_scales,
        None,
        None,
        None,
        None,
        (128,128))

    msg = f"[TRITON_perf] {M=}, {K=}, {N=}, {E=}, {topk=}, dtype: {dtype}, triton_avg: {avg_triton:>8.2f} us"
    checkAllclose(ref_out, triton_output, rtol=0.01, atol=300, msg=msg)

    # Cross-backend validation: use triton as reference for ASM and CK
    msg = f"[ASM_vs_TRITON] {M=}, {K=}, {N=}, {E=}, {topk=}, dtype: {dtype}, asm_avg: {avg_asm:>8.2f} us"
    checkAllclose(triton_output, asm_output, rtol=0.01, atol=1, msg=msg)

    msg = f"[CK_vs_TRITON] {M=}, {K=}, {N=}, {E=}, {topk=}, dtype: {dtype}, ck_avg: {avg_ck:>8.2f} us"
    checkAllclose(triton_output, ck_output, rtol=0.01, atol=1, msg=msg)

    return{
        "triton_us": avg_triton,
        "asm_us": avg_asm,
        "asm_ck": avg_ck,
    }


df = []
for dtype in [dtypes.fp16]:
    # for m in [1]:
    for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,64,96,128,256,512,1024]:
    # for m in [1, 2,  4,  8, 16 ,32,64, 96, 128,256,512,1024,2048,4096,8192,16384,32768]:
        for dim in [7168]:
            for hdim in [256]:
                # test_fmoe(dtype, m, dim, hdim, 32, 5)
                # test_fmoe(dtype, m, dim, hdim, 256, 8, quant="No", use_g1u1=True)
                ret = test_fused_moe_w8a8(m, dim, hdim, 256, 8, 0, dtype, 8,(128,128))
                df.append(ret)
df = pd.DataFrame(df)
df.to_csv("moe_w8a8_blockscale_fp8.csv")
aiter.logger.info(f"summary:\n{df}")