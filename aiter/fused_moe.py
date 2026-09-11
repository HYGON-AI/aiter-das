# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
 
import torch
import torch.nn.functional as F
from einops import rearrange
import os
from typing import Optional
import functools
import aiter
from aiter import logger
from aiter import ActivationType, QuantType, dtypes

# from aiter import get_hip_quant as get_quant
# from aiter import get_torch_quant as get_quant
from aiter import get_triton_quant as get_quant
from aiter.jit.core import AITER_ROOT_DIR, PY, get_asm_dir, bd_dir, mp_lock
from aiter.jit.utils.chip_info import get_cu_num
from aiter import pertoken_quant, ck_moe

BLOCK_SIZE_M = 32


def moe_sorting(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=BLOCK_SIZE_M,
    expert_mask=None,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=dtypes.i32, device=device
    )
    tokens_positions_per_expert = torch.empty(
        (num_experts*2,), dtype=dtypes.i32, device=device
    )
    num_valid_ids = torch.empty((1), dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    # for now, moe_sorting_fwd only support int32 topk_ids
    if topk_ids.dtype != dtypes.i32:
        topk_ids = topk_ids.to(dtypes.i32)

    aiter.moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        tokens_positions_per_expert,
        num_valid_ids,
        moe_buf,
        num_experts,
        block_size,
        expert_mask,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, tokens_positions_per_expert, moe_buf


@functools.lru_cache(maxsize=1024)
def get_inter_dim(w1_shape, w2_shape):
    E, _, model_dim = w1_shape
    E, model_dim, inter_dim = w2_shape

    int4_war = model_dim // w1_shape[-1]
    inter_dim *= int4_war
    return E, model_dim, inter_dim


def fused_moe(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    expert_mask=None,  # EP
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    # following for quant
    w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M=None,
):
    """user API"""
    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    assert w1.shape[1] in [
        inter_dim,
        inter_dim * 2,
    ], f"Invalid MoE weight: {w1.shape=} {w2.shape=}"
    isG1U1 = inter_dim != w1.shape[1]

    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    dtype = hidden_states.dtype
    q_dtype_w = w1.dtype
    q_dtype_a = w1.dtype if w1.dtype != torch.uint32 else dtypes.fp8

    #暂时关闭block_size_M 与 2stage 相关的pass

    # if block_size_M is None:
    #     _, _, block_size_M, *_ = get_2stage_cfgs(
    #         M,
    #         model_dim,
    #         inter_dim,
    #         E,
    #         topk,
    #         dtype,
    #         q_dtype_a,
    #         q_dtype_w,
    #         quant_type,
    #         isG1U1,
    #         activation,
    #         doweight_stage1,
    #     )
    # run_1stage = M < 256
    run_1stage = True
    #暂不支持blockwise quant
    # run_1stage = quant_type == QuantType.per_128x128
    block_size_M = 32 if run_1stage else block_size_M
    return aiter.ck_moe(
                        hidden_states = hidden_states,
                        w1 = w1,
                        w2 = w2,
                        topk_weights = topk_weight,
                        topk_ids = topk_ids,
                        w1_scale = w1_scale,
                        w2_scale = w2_scale,
                        a1_scale = a1_scale,
                        a2_scale = a2_scale,
                    ).to(dtypes.fp16)
    
    # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    #     topk_ids, topk_weight, global_E, model_dim, dtype, block_size_M, expert_mask
    # )

    # if run_1stage:
    #     assert (
    #         doweight_stage1 == False
    #     ), "doweight_stage1 not support in fused_moe_1stage"
    #     return fused_moe_1stage(
    #         hidden_states,
    #         w1,
    #         w2,
    #         topk,
    #         sorted_ids,
    #         sorted_weights,
    #         sorted_expert_ids,
    #         num_valid_ids,
    #         moe_buf,
    #         isG1U1,
    #         block_size_M,
    #         activation=activation,
    #         quant_type=quant_type,
    #         q_dtype_a=q_dtype_a,
    #         q_dtype_w=q_dtype_w,
    #         w1_scale=w1_scale,
    #         w2_scale=w2_scale,
    #         a1_scale=a1_scale,
    #         a2_scale=a2_scale,
    #     )
    # else:
    #     return fused_moe_2stages(
    #         hidden_states,
    #         w1,
    #         w2,
    #         topk,
    #         sorted_ids,
    #         sorted_weights,
    #         sorted_expert_ids,
    #         num_valid_ids,
    #         moe_buf,
    #         isG1U1,
    #         block_size_M,
    #         activation=activation,
    #         quant_type=quant_type,
    #         doweight_stage1=doweight_stage1,
    #         q_dtype_a=q_dtype_a,
    #         q_dtype_w=q_dtype_w,
    #         w1_scale=w1_scale,
    #         w2_scale=w2_scale,
    #         a1_scale=a1_scale,
    #         a2_scale=a2_scale,
    #     )


def fused_moe_1stage(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    isG1U1,
    block_size_M=32,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    # following for quant
    q_dtype_a=None,
    q_dtype_w=None,
    w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
):
    raise NotImplementedError("The legacy fused_moe_1stage path is not implemented in this checkout")


@functools.lru_cache(maxsize=1024)
def get_block_size_M(token, topk, expert, inter_dim):
    cu_num = get_cu_num()
    tileN = 128
    tgN = (inter_dim + tileN - 1) // tileN
    support_list = [32, 64, 128]

    tmp = []
    for el in support_list:
        max_num_tokens = token * topk + expert * el - topk
        tg_num = tgN * (max_num_tokens + el - 1) // el
        rnd = (tg_num + cu_num - 1) // cu_num
        empty = cu_num - tg_num % cu_num
        tmp.append((rnd, empty, el))
    return sorted(tmp, key=lambda x: x[:2])[0][-1]


cfg_2stages = None


@functools.lru_cache(maxsize=1024)
def get_2stage_cfgs(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    use_g1u1,
    activation,
    doweight_stage1,
):
    raise NotImplementedError("The legacy two-stage CK/ASM adapters are not implemented in this checkout")


@functools.lru_cache()
def get1tensor(device):
    return torch.tensor(1.0, dtype=torch.float, device=device)


# def fused_moe_2stages(
#     hidden_states,
#     w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
#     w2,  # [expert(local_expert:EP), dim, inter_dim]
#     topk,
#     sorted_ids,
#     sorted_weights,
#     sorted_expert_ids,
#     num_valid_ids,
#     moe_out,
#     isG1U1,
#     block_size_M,
#     activation=ActivationType.Silu,
#     quant_type=QuantType.No,
#     doweight_stage1=False,
#     # following for quant
#     q_dtype_a=None,
#     q_dtype_w=None,
#     w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
#     w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
#     a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
#     a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
# ):

#     quant_func = get_quant(quant_type)

#     token_num, _ = hidden_states.shape
#     E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
#     dtype = hidden_states.dtype
#     device = hidden_states.device

#     stage1, stage2, block_m, ksplit = get_2stage_cfgs(
#         token_num,
#         model_dim,
#         inter_dim,
#         E,
#         topk,
#         dtype,
#         q_dtype_a,
#         q_dtype_w,
#         quant_type,
#         isG1U1,
#         activation,
#         doweight_stage1,
#     )

#     a1, a1_scale = quant_func(hidden_states, scale=a1_scale, quant_dtype=q_dtype_a)
#     if quant_type != QuantType.per_128x128:
#         a2 = torch.empty(
#             (token_num, topk, inter_dim),
#             dtype=dtype,
#             device=device,
#         )
#     else:
#         ratio = a1_scale.element_size() // a1.element_size()
#         a2 = torch.empty(
#             (token_num + (token_num * ratio + 127) // 128, topk, inter_dim),
#             dtype=q_dtype_a,
#             device=device,
#         )

#     a2 = stage1(
#         a1,
#         w1,
#         w2,
#         sorted_ids,
#         sorted_expert_ids,
#         num_valid_ids,
#         a2,
#         block_m=block_m,
#         a1_scale=a1_scale,
#         w1_scale=w1_scale,
#         sorted_weights=sorted_weights if doweight_stage1 else None,
#     )

#     if quant_type != QuantType.per_128x128:
#         if quant_type == QuantType.per_Token:
#             a2 = a2.view(token_num, -1)
#         a2, a2_scale = quant_func(a2, scale=a2_scale, quant_dtype=q_dtype_a)
#         a2 = a2.view(token_num, topk, inter_dim)
#     else:
#         a2_v = a2[:token_num, :, :]
#         a2_scale = (
#             a2[token_num:, ...]
#             .view(-1)[: token_num * topk * inter_dim * ratio // 128]
#             .view(dtypes.fp32)
#             .view(token_num, -1)
#         )
#         a2 = a2_v

#     if quant_type == aiter.QuantType.No:
#         a2_scale = get1tensor(device)

#     stage2(
#         a2,
#         w1,
#         w2,
#         sorted_ids,
#         sorted_expert_ids,
#         num_valid_ids,
#         moe_out,
#         topk,
#         w2_scale,
#         a2_scale,
#         block_size_M,
#         sorted_weights=sorted_weights if not doweight_stage1 else None,
#     )

#     return moe_out


def torch_moe_act(act_input, torch_act, inter_dim):
    if act_input.shape[-1] == inter_dim:
        return torch_act(act_input)
    else:
        gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
        return torch_act(gate) * up


# def asm_stage1(
#     input,
#     w1,
#     w2,
#     sorted_ids,
#     sorted_expert_ids,
#     num_valid_ids,
#     out,  # [token_num, topk, inter_dim]
#     block_m: int,
#     kernelName: str = "",
#     ksplit: int = 0,
#     activation=ActivationType.Silu,
#     quant_type=QuantType.No,
#     a1_scale=None,
#     w1_scale=None,
#     sorted_weights=None,
# ):
#     dtype = dtypes.bf16  # out.dtype, asm only support bf16
#     if quant_type != QuantType.per_128x128:
#         out = out.view(dtype)
#     device = out.device
#     token_num, topk, _ = out.shape
#     E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

#     if quant_type == QuantType.per_Tensor:
#         a1_scale = a1_scale.view(1, 1).repeat(token_num, 1)
#         w1_scale = w1_scale.view(E, 1).repeat(1, w1.shape[1])
#         quant_type = QuantType.per_Token

#     tmp_out = out
#     if ksplit > 0:
#         tmp_out = torch.zeros(
#             (token_num, topk, w1.shape[1]),
#             dtype=dtypes.fp32,
#             device=device,
#         ).view(dtype)

#     aiter.moe_stage1_g1u1(
#         input,
#         w1,
#         w2,
#         sorted_ids,
#         sorted_expert_ids,
#         num_valid_ids,
#         tmp_out,
#         inter_dim,
#         kernelName,
#         block_m,
#         ksplit=ksplit,
#         activation=activation,
#         quant_type=quant_type,
#         a1_scale=a1_scale,
#         w1_scale=w1_scale,
#         sorted_weights=sorted_weights,
#     )
#     if ksplit > 0:
#         if activation == ActivationType.Silu:
#             aiter.silu_and_mul(out, tmp_out.view(dtypes.fp32).to(dtype))
#         else:
#             aiter.gelu_and_mul(out, tmp_out.view(dtypes.fp32).to(dtype))
#     return out


# def ck_stage1(
#     input,  # [token, model_dim]
#     w1,  # [E, inter_dim*2, model_dim]
#     w2,  # [E, model_dim, inter_dim]
#     sorted_ids,  # [max_num_tokens_padded]
#     sorted_expert_ids,  # [max_num_m_blocks]
#     num_valid_ids,  # [1]
#     out,  # [token_num, topk, inter_dim]
#     block_m=32,
#     activation=ActivationType.Silu,
#     a1_scale=None,
#     w1_scale=None,
#     sorted_weights=None,
# ):
#     _, topk, _ = out.shape
#     # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

#     if activation == ActivationType.Silu:
#         act_op = 1
#     else:
#         act_op = 0

#     aiter.ck_moe_stage1(
#         input,
#         w1,
#         w2,
#         sorted_ids,
#         sorted_expert_ids,
#         num_valid_ids,
#         out,
#         topk,
#         w1_scale,
#         a1_scale,
#         block_m,
#         sorted_weights,
#         act_op,
#     )

#     return out


def torch_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    expert_mask=None,
    activation=ActivationType.Silu,
):
    computeType = dtypes.fp32
    dtype = hidden_states.dtype
    torch_act = aiter.get_torch_act(activation)
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=computeType,
        device=hidden_states.device,
    )

    inter_dim = w2.shape[2]

    if fc1_scale is not None:
        # gose to quant D_w8a8/w8a8
        expert = w1.shape[0]
        w2D = w2.shape[-1]
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(expert, -1, D)
        w2 = (w2.view(-1, w2D) * fc2_scale.view(-1, 1)).view(expert, -1, w2D)

    if fc1_smooth_scale is not None:
        expert = fc1_smooth_scale.shape[0]
        fc1_smooth_scale = fc1_smooth_scale.view(expert, -1)
        fc2_smooth_scale = fc2_smooth_scale.view(expert, -1)

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if fc1_smooth_scale is not None:
                sub_tokens = sub_tokens * (fc1_smooth_scale[E_id])

            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            act_out = torch_moe_act(act_input, torch_act, inter_dim)
            if fc2_smooth_scale is not None:
                act_out = act_out * (fc2_smooth_scale[E_id])
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)

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

def torch_moe_stage1(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weight,
    topk_ids,
    dtype=dtypes.fp16,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    # following for quant
    a1_scale=None,  # [token, 1]
    w1_scale=None,  # [expert, inter_dim, 1]
    doweight=False,
    group_by_expert=False,
):
    ctype = dtypes.fp32  # compute type
    hidden_states = hidden_states.to(ctype)
    w1 = w1.to(ctype)

    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    N = w1.shape[1]
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    if quant_type in [QuantType.per_Token, QuantType.per_Tensor]:
        w1 = w1 * w1_scale.view(w1_scale.shape[0], -1, 1)
        hidden_states = hidden_states * a1_scale
    # per_128x128
    elif quant_type == QuantType.per_128x128:
        w1_shape = w1.shape
        w1 = w1.view(
            w1.shape[0], w1.shape[1] // 128, 128, w1.shape[2] // 128, 128
        ) * w1_scale.view(
            w1_scale.shape[0], w1.shape[1] // 128, 1, w1.shape[2] // 128, 1
        )
        w1 = w1.view(w1_shape)

        a1_scale = a1_scale.view(hidden_states.shape[0], -1, 1)
        a1_scale = a1_scale.repeat(
            1, 1, hidden_states.shape[-1] // a1_scale.shape[1]
        ).view(hidden_states.shape[0], -1)
        hidden_states = hidden_states * a1_scale
    elif quant_type == QuantType.No:
        pass
    else:
        assert False, f"Unsupported quant_type: {quant_type}"

    hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1)

    out = torch.zeros(
        (B, topk, N),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if doweight:
                act_input = act_input * topk_weight[mask].view(-1, 1)
            out[mask] = act_input


    if group_by_expert:
        out_flat = out.reshape(-1, out.shape[-1])
        expert_indices = []
        for expert_id in range(E):
            positions = torch.nonzero(topk_ids == expert_id, as_tuple=False)
            if positions.numel() == 0:
                continue
            linear_idx = positions[:, 0] * topk + positions[:, 1]
            expert_indices.append(linear_idx)

        if expert_indices:
            gather_idx = torch.cat(expert_indices).to(
                device=out_flat.device, dtype=torch.long
            )
            out_grouped = out_flat.index_select(0, gather_idx)
        else:
            out_grouped = out_flat[:0]

        out = out_grouped

    use_g1u1 = w1.shape[1] == (2 * inter_dim)
    torch_act = aiter.get_torch_act(activation)
    if use_g1u1:
        gate, up = out.split([inter_dim, inter_dim], dim=-1)
        out = torch_act(gate) * up
    else:
        out = torch_act(out)
    return out.to(dtype)


def torch_moe_stage2(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weights,
    topk_ids,
    dtype=dtypes.fp16,
    quant_type=QuantType.No,
    w2_scale=None,  # [1]
    a2_scale=None,  # [expert]]'
    doweight=True,
):
    ctype = dtypes.fp32  # compute type
    hidden_states = hidden_states.to(ctype)
    w2 = w2.to(ctype)

    token_num, topk = topk_ids.shape
    num_experts, model_dim, inter_dim = w2.shape
    hidden_states = hidden_states.view(token_num, topk, inter_dim)

    if quant_type in [QuantType.per_Token, QuantType.per_Tensor]:
        w2 = w2 * w2_scale.view(w2_scale.shape[0], -1, 1)
    # per_128x128
    elif quant_type == QuantType.per_128x128:
        w2_shape = w2.shape
        w2 = w2.view(
            w2.shape[0], w2.shape[1] // 128, 128, w2.shape[2] // 128, 128
        ) * w2_scale.view(
            w2_scale.shape[0], w2.shape[1] // 128, 1, w2.shape[2] // 128, 1
        )
        w2 = w2.view(w2_shape)

    if quant_type in [QuantType.per_Token, QuantType.per_Tensor]:
        hidden_states = hidden_states * a2_scale.view(a2_scale.shape[0], -1, 1)
    elif quant_type == QuantType.per_128x128:
        a2_scale = a2_scale.view(hidden_states.shape[0], topk, -1, 1)
        a2_scale = a2_scale.repeat(1, 1, 1, 128).view(hidden_states.shape[0], topk, -1)
        hidden_states = hidden_states * a2_scale

    out = torch.zeros(
        (token_num, topk, model_dim),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w2[E_id].transpose(0, 1))
            out[mask] = act_input
    if doweight:
        out = out * topk_weights.view(token_num, -1, 1)
    return out.sum(1).to(dtype)


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    is_softmax: bool = True,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    if topk_weights is None:
        topk_weights = torch.empty(
            M, topk, dtype=dtypes.fp32, device=hidden_states.device
        )
    if topk_ids is None:
        topk_ids = torch.empty(M, topk, dtype=dtypes.i32, device=hidden_states.device)

    if is_softmax:
        token_expert_indicies = torch.empty(
            M, topk, dtype=dtypes.i32, device=hidden_states.device
        )

        aiter.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output.float(),  # TODO(woosuk): Optimize this.
            renormalize,
        )
        del token_expert_indicies  # Not used. Will be used in the future.
    
    # sigmoid
    else:
        # Use sigmoid and simple topk
        # scores = gating_output.sigmoid()
        # topk_weights_temp, topk_ids_temp = torch.topk(scores, k=topk, dim=-1)
        # topk_weights.copy_(topk_weights_temp)
        # topk_ids.copy_(topk_ids_temp.to(dtypes.i32))
        # if renormalize:
        #     topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True))

        aiter.grouped_topk(
            gating_output,
            topk_weights,
            topk_ids,
            1,
            1,
            renormalize,
            False,      # is_softmax=False, go sigmoid
            routed_scaling_factor = 1.0
        )

    return topk_weights, topk_ids
