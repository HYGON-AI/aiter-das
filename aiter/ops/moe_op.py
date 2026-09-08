# SPDX-License-Identifier: MIT
import torch
from torch import Tensor
from typing import Optional,List
from ..jit.core import (
    compile_ops,
)
from .enum import ActivationType, Enum, QuantType


@compile_ops("module_cpp_api")
def topk_softmax(
    topk_weights: Tensor,
    topk_indices: Tensor,
    token_expert_indices: Tensor,
    gating_output: Tensor,
    need_renorm: bool,
) -> None: ...


@compile_ops("module_moe_utils")
def moe_sum(input: Tensor, output: Tensor)->None: ...

@compile_ops("module_moe_sum")
def asm_moe_sum(input: Tensor, output: Tensor, sorted_ids: Tensor)->None: ...

@compile_ops("module_moe_utils")
def sgl_moe_align_block_size(topk_ids: Tensor, num_experts: int,
                             block_size: int, sorted_token_ids: Tensor,
                             experts_ids: Tensor,
                             num_tokens_post_pad: Tensor) -> None: ...

@compile_ops("module_moe_utils")
def moe_align_block_size(
    topk_ids: Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: Tensor,
    experts_ids: Tensor,
    num_tokens_post_pad: Tensor,
) -> None: ...


@compile_ops("module_moe_asm")
def asm_fmoe_stage1(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    top_k: int,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    zero_points: Optional[torch.Tensor] = None,
    mode: Optional[int] = 0,
    solidx: Optional[int] = 0,
    block_size: Optional[int] = 16,
    persist_groups: Optional[int] = 0,
) -> None: ...

@compile_ops("module_moe_asm")
def asm_fmoe_stage2(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    top_k: int,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    zero_points: Optional[torch.Tensor] = None,
    mode: Optional[int] = 0,
    solidx: Optional[int] = 0,
    block_size: Optional[int] = 16,
    persist_groups: Optional[int] = 0,
)-> None: ...

@compile_ops("module_moe_asm")
def asm_fmoe_a8(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    top_k: int,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    zero_points: Optional[torch.Tensor] = None,
    mode: Optional[int] = 0,
    solidx: Optional[int] = 0,
    out_type:Optional[int] = 0,
    persist_groups:Optional[int] = 0,
    use_shuffle:Optional[int] = 0,
)-> None: ...

@compile_ops("module_moe_asm")
def asm_moe_get_solutions(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    use_int8_w8a16: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
    use_int4_w4a8: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    per_channel_quant: Optional[bool] = False,
    w1_zp: Optional[Tensor] = None,
    w2_zp: Optional[Tensor] = None,
    w1_scale: Optional[Tensor] = None,
    w2_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_shape_n: Optional[int] = 0,
    block_shape_k: Optional[int] = 0,
    block_m: Optional[int] = 32,
    expert_mask: Optional[Tensor] = None,
) -> list[str]: ...

# @compile_ops("module_moe_asm")
# def fmoe(
#     out: Tensor,
#     input: Tensor,
#     gate: Tensor,
#     down: Tensor,
#     sorted_token_ids: Tensor,
#     sorted_weights: Tensor,
#     sorted_expert_ids: Tensor,
#     num_valid_ids: Tensor,
#     topk: int,
# ): ...


# @compile_ops("module_moe_asm")
# def fmoe_int8_g1u0(
#     out: Tensor,
#     input: Tensor,
#     gate: Tensor,
#     down: Tensor,
#     sorted_token_ids: Tensor,
#     sorted_weights: Tensor,
#     sorted_expert_ids: Tensor,
#     num_valid_ids: Tensor,
#     topk: int,
#     input_scale: Tensor,
#     fc1_scale: Tensor,
#     fc2_scale: Tensor,
#     fc2_smooth_scale: Tensor,
#     activation: Optional[Enum] = ActivationType.Silu,
# ): ...


# @compile_ops("module_moe_asm")
# def fmoe_g1u1(
#     out: Tensor,
#     input: Tensor,
#     gate: Tensor,
#     down: Tensor,
#     sorted_token_ids: Tensor,
#     sorted_weights: Tensor,
#     sorted_expert_ids: Tensor,
#     num_valid_ids: Tensor,
#     topk: int,
#     input_scale: Tensor,
#     fc1_scale: Tensor,
#     fc2_scale: Tensor,
#     fc2_smooth_scale: Optional[Tensor] = None,
#     activation: Optional[Enum] = ActivationType.Silu,
# ): ...


# @compile_ops("module_moe_asm")
# def fmoe_g1u1_tkw1(
#     out: Tensor,
#     input: Tensor,
#     gate: Tensor,
#     down: Tensor,
#     sorted_token_ids: Tensor,
#     sorted_weights: Tensor,
#     sorted_expert_ids: Tensor,
#     num_valid_ids: Tensor,
#     topk: int,
#     input_scale: Tensor,
#     fc1_scale: Tensor,
#     fc2_scale: Tensor,
#     fc2_smooth_scale: Optional[Tensor] = None,
#     activation: Optional[Enum] = ActivationType.Silu,
# ): ...


# @compile_ops("module_moe_asm")
# def fmoe_int8_g1u0_a16(
#     out: Tensor,
#     input: Tensor,  # bf16
#     gate: Tensor,
#     down: Tensor,
#     sorted_token_ids: Tensor,
#     sorted_weights: Tensor,
#     sorted_expert_ids: Tensor,
#     num_valid_ids: Tensor,
#     topk: int,
#     fc1_scale: Tensor,
#     fc2_scale: Tensor,
#     fc1_smooth_scale: Tensor,
#     fc2_smooth_scale: Tensor,
# ): ...


# @compile_ops("module_moe_asm")
# def fmoe_g1u1_a16(
#     out: Tensor,
#     input: Tensor,  # bf16
#     gate: Tensor,
#     down: Tensor,
#     sorted_token_ids: Tensor,
#     sorted_weights: Tensor,
#     sorted_expert_ids: Tensor,
#     num_valid_ids: Tensor,
#     topk: int,
#     fc1_scale: Tensor,
#     fc2_scale: Tensor,
#     fc1_smooth_scale: Tensor,
#     fc2_smooth_scale: Tensor,
# ): ...


# @compile_ops("module_moe_asm")
# def fmoe_fp8_blockscale_g1u1(
#     out: Tensor,
#     input: Tensor,
#     gate: Tensor,
#     down: Tensor,
#     sorted_token_ids: Tensor,
#     sorted_weights: Tensor,
#     sorted_expert_ids: Tensor,
#     num_valid_ids: Tensor,
#     topk: int,
#     input_scale: Tensor,
#     fc1_scale: Tensor,
#     fc2_scale: Tensor,
#     fc_scale_blkn: int = 128,
#     fc_scale_blkk: int = 128,
#     fc2_smooth_scale: Optional[Tensor] = None,
#     activation: ActivationType = ActivationType.Silu,
# ): ...


# @compile_ops("module_moe_asm")
# def moe_stage1_g1u1(
#     input: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     sorted_token_ids: torch.Tensor,
#     sorted_expert_ids: torch.Tensor,
#     num_valid_ids: torch.Tensor,
#     out: torch.Tensor,
#     inter_dim: int,
#     kernelName: str,
#     block_m: int,
#     ksplit: int = 0,
#     activation: ActivationType = ActivationType.Silu,
#     quant_type: QuantType = QuantType.No,
#     a1_scale: Optional[torch.Tensor] = None,
#     w1_scale: Optional[torch.Tensor] = None,
#     sorted_weights: Optional[torch.Tensor] = None,
# ) -> None: ...


@compile_ops("module_moe")
def ck_moe(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    use_int8_w8a16: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_int8_w8a8_block: Optional[bool] = False,
    use_int4_w4a8_block: Optional[bool] = False,
    w1_zp: Optional[Tensor] = None,
    w2_zp: Optional[Tensor] = None,
    w1_scale: Optional[Tensor] = None,
    w2_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_shape_n: Optional[int] = 0,
    block_shape_k: Optional[int] = 0,
    block_m: Optional[int] = 32,
    solution_id: Optional[int] = 0,
    expert_mask: Optional[Tensor] = None,
    activation: Enum = 0,
)-> torch.Tensor: ...

@compile_ops("module_moe")
def ck_shuffle_moe(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    use_int8_w8a16: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_int8_w8a8_block: Optional[bool] = False,
    use_int4_w4a8_block: Optional[bool] = False,
    w1_zp: Optional[Tensor] = None,
    w2_zp: Optional[Tensor] = None,
    w1_scale: Optional[Tensor] = None,
    w2_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_shape_n: Optional[int] = 0,
    block_shape_k: Optional[int] = 0,
    block_m: Optional[int] = 32,
    solution_id: Optional[int] = 0,
    expert_mask: Optional[Tensor] = None,
    activation: Enum = 0,
)-> torch.Tensor: ...

@compile_ops("module_moe")
def ck_moe_get_solutions(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    use_int8_w8a16: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_int8_w8a8_block: Optional[bool] = False,
    use_int4_w4a8_block: Optional[bool] = False,
    w1_zp: Optional[Tensor] = None,
    w2_zp: Optional[Tensor] = None,
    w1_scale: Optional[Tensor] = None,
    w2_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_shape_n: Optional[int] = 0,
    block_shape_k: Optional[int] = 0,
    block_m: Optional[int] = 32,
    expert_mask: Optional[Tensor] = None,
) -> list[int]: ...

@compile_ops("module_moe")
def ck_moe_stage_1(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    tokens_positions_per_expert: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    use_int8_w8a8_block: Optional[bool] = False,
    use_fp8_w8a8_block: Optional[bool] = False,
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_shape_n: Optional[int] = 0,
    block_shape_k: Optional[int] = 0,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    act_op: Optional[int] = 0,
)->None: ...


@compile_ops("module_moe")
def ck_moe_stage_2(
    inter_states: Tensor,      # the output of stage 1
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    tokens_positions_per_expert: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    use_int8_w8a8_block: Optional[bool] = False,
    use_fp8_w8a8_block: Optional[bool] = False,
    w2_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_shape_n: Optional[int] = 0,
    block_shape_k: Optional[int] = 0,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
)->None: ...

@compile_ops("module_moe")
def ck_moe_per_token_quant(
    input: Tensor,
    out_quant: Tensor,
    out_scale: Tensor,
)->None: ...

# @compile_ops("module_moe_ck2stages")
# def ck_moe_stage1(
#     hidden_states: Tensor,
#     w1: Tensor,
#     w2: Tensor,
#     sorted_token_ids: Tensor,
#     sorted_expert_ids: Tensor,
#     num_valid_ids: Tensor,
#     out: Tensor,
#     topk: int,
#     w1_scale: Optional[Tensor] = None,
#     a1_scale: Optional[Tensor] = None,
#     block_m: Optional[int] = 32,
#     sorted_weights: Optional[Tensor] = None,
#     act_op: Optional[int] = 0,
# ): ...


# @compile_ops("module_moe_ck2stages")
# def ck_moe_stage2(
#     inter_states: Tensor,
#     w1: Tensor,
#     w2: Tensor,
#     sorted_token_ids: Tensor,
#     sorted_expert_ids: Tensor,
#     num_valid_ids: Tensor,
#     out: Tensor,
#     topk: int,
#     w2_scale: Optional[Tensor] = None,
#     a2_scale: Optional[Tensor] = None,
#     block_m: Optional[int] = 32,
#     sorted_weights: Optional[Tensor] = None,
# ): ...
