# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Stage-level router for fused_moe: picks a backend plan for a normalized
# problem, then executes it.
#
# Design:
#   * Tuned tables are read only through aiter.moe's _try_get_{asm,moe_c,ck}_config,
#     the same functions aiter_moe uses. They are imported inside functions, so
#     importing this module stays light.
#   * FusedMoePlan has separate stage1/stage2 backend fields, so mixed plans
#     (e.g. asm stage1 + ck stage2) fit the same dispatch loop.
#   * Each backend is one registry entry (match/lookup/execute); adding a
#     backend does not touch the core. Execute functions import torch/aiter
#     lazily, so the routing core stays torch-free.
#   * The only host-module imports are the MOE_C entry `moe_c_fused_experts`
#     and the aiter.moe lookup functions, both called lazily.

import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

# Resolved locally (env override first) instead of importing aiter.jit.core,
# which would pull in the whole aiter package (torch + pybind) and make this
# module depend on torch.
_AITER_ROOT_DIR_ENV = os.environ.get("AITER_ROOT_DIR", "")
AITER_ROOT_DIR = (
    _AITER_ROOT_DIR_ENV
    if _AITER_ROOT_DIR_ENV
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# ---------------------------------------------------------------------------
# Problem / plan containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedMoeProblem:
    """Normalized routing input (all scalars/strings, torch-free).

    Dtype strings: 'fp16' | 'bf16' for native dtypes; quant dtypes use
    'none' | 'fp8' | 'int8' | 'int4'. ``quant_kind`` is the quantization
    family resolved by the caller (see fused_moe._normalize_problem):
      none | w8a8 | w8a8_block | w4a8 | a8w4 | w4a16 | w8a16
    """

    arch: str
    M: int
    topk: int
    E: int
    model_dim: int
    inter_dim: int
    dtype: str  # activation dtype: 'fp16' | 'bf16'
    q_dtype_a: str  # 'none' | 'fp8' | 'int8'
    q_dtype_w: str  # 'none' | 'fp8' | 'int8' | 'int4'
    quant_kind: str
    quant_type: str = "no"  # raw QuantType name: no/per_Tensor/per_Token/...
    block_shape: tuple | None = None
    g1u1: bool = True
    activation: str = "silu"
    doweight_stage1: bool = False
    use_shuffle: int = 0
    is_ep: bool = False
    splitk: int = 0
    block_size_M: int | None = None  # user override, ck MPerBlock hint
    local_E: int = 0  # experts held on this rank (EP); 0 = same as E
    weight_layout: str = "raw"  # raw | marlin | asm_shuffle; set by
                                # _normalize_problem from the prepared-weight marker


@dataclass
class FusedMoePlan:
    """Stage-level dispatch result.

    stage1_backend / stage2_backend name the per-stage backends ("asm"|"ck");
    both are None when a whole-chain backend is selected, in which case
    ``chained_backend`` carries the name ("moe_c"|"triton") and ``extra``
    holds its keyword arguments.
    """

    stage1_backend: str | None = None
    stage2_backend: str | None = None
    chained_backend: str | None = None
    block_m: int = 32
    act_op: int = 0  # ck: 1 = SiLU fused inside stage1, 0 = external activation
    sol_id1: Any = None  # asm: int "A" part; ck: int
    sol_id2: Any = None
    mode: int = 0  # asm: 0=w16a16, 2=w4a16-fp16, 3=w4a16-bf16, 4=w4a16-bf16-group32
    use_shuffle: int = 0
    flat: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class FusedMoeExecContext:
    """Runtime tensors handed to a backend execute(). Fields are opaque here."""

    hidden_states: Any = None
    w1: Any = None
    w2: Any = None
    topk_weight: Any = None
    topk_ids: Any = None
    expert_mask: Any = None
    activation: Any = None  # aiter ActivationType enum
    quant_type: Any = None  # aiter QuantType enum
    dtype: Any = None  # output dtype (torch.dtype) or None
    output: Any = None
    w1_scale: Any = None
    w2_scale: Any = None
    a1_scale: Any = None
    a2_scale: Any = None
    num_local_tokens: Any = None
    plan: FusedMoePlan | None = None
    problem: FusedMoeProblem | None = None


@dataclass(frozen=True)
class BackendEntry:
    name: str
    match: Callable[[FusedMoeProblem], bool]
    # None = matched, but no tuned config for this shape (try the next candidate)
    lookup: Callable[[FusedMoeProblem], dict | None]
    execute: Callable[[FusedMoeExecContext], Any]


# ---------------------------------------------------------------------------
# Backend match / lookup helpers
# ---------------------------------------------------------------------------


def _to_moe_quant_type(problem: FusedMoeProblem):
    """quant_kind -> aiter.moe.MoeQuantType. Returns None for int4 families,
    which have no routable lookup (plan() raises for them)."""
    from aiter.moe import MoeQuantType

    kind = problem.quant_kind
    if kind == "none":
        return MoeQuantType.W16A16
    if kind == "w8a16":
        return MoeQuantType.INT8_W8A16
    if kind in ("w8a8", "w8a8_block"):
        return (
            MoeQuantType.FP8_W8A8 if problem.q_dtype_a == "fp8" else MoeQuantType.W8A8
        )
    return None


def _moe_block_size(problem: FusedMoeProblem) -> int:
    """block_size argument for the aiter.moe lookups: the K group size for
    w8a8_block (128), 0 for everything else."""
    if problem.quant_kind == "w8a8_block" and problem.block_shape:
        return int(problem.block_shape[1])
    return 0


def _torch_dtype(problem: FusedMoeProblem):
    """dtype string -> torch.dtype."""
    import torch

    return torch.float16 if problem.dtype == "fp16" else torch.bfloat16


def _activation_ok(problem: FusedMoeProblem, allowed=("silu", "gelu")) -> bool:
    return problem.activation in allowed


def _looks_like_mxfp(problem: FusedMoeProblem) -> bool:
    return problem.quant_type in ("per_1x32", "per_256x128", "per_1024x128")


# ---------------------------------------------------------------------------
# Backend execute paths (torch/aiter imported lazily)
# ---------------------------------------------------------------------------


def _moe_sorting(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype,
                 block_size=32, expert_mask=None):
    """Local copy of the moe_sorting wrapper, so the router does not depend on
    aiter.fused_moe. Returns the six sorting outputs."""
    import torch

    import aiter
    from aiter import dtypes

    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty((max_num_tokens_padded,), dtype=dtypes.fp32, device=device)
    sorted_expert_ids = torch.empty((max_num_m_blocks,), dtype=dtypes.i32, device=device)
    tokens_positions_per_expert = torch.empty((num_experts * 2,), dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty((1,), dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
    if topk_ids.dtype != dtypes.i32:
        topk_ids = topk_ids.to(dtypes.i32)
    aiter.moe_sorting_fwd(
        topk_ids, topk_weights, sorted_ids, sorted_weights, sorted_expert_ids,
        tokens_positions_per_expert, num_valid_ids, moe_buf,
        num_experts, block_size, expert_mask,
    )
    return (sorted_ids, sorted_weights, sorted_expert_ids,
            num_valid_ids, tokens_positions_per_expert, moe_buf)


def _external_act(act_name, inter_dim, act_input):
    """Gate/up activation applied between stage1 and stage2 (the asm kernels
    do not apply it). act_name is 'silu' or 'gelu'."""
    import torch.nn.functional as F

    if act_input.shape[-1] == inter_dim:
        return F.gelu(act_input) if act_name == "gelu" else F.silu(act_input)
    gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
    gate = F.gelu(gate) if act_name == "gelu" else F.silu(gate)
    return gate * up


def _asm_execute(ctx: FusedMoeExecContext):
    """Run the asm path by forwarding to fused_moe_asm_wna16.fused_experts_asm_impl.

    This is the same entry aiter_moe uses. The tuned solution ("SOL1+SOL2",
    PADDED_K, USE_SHUFFLE) comes from aiter.moe._try_get_asm_config via
    moe_cfg. The forwarded function handles quantization, K padding, weight
    shuffle tables, sorting, activation and moe_sum itself, and launches only
    solutions whose block size matches the blob's launch bounds.
    """
    import torch

    from aiter.fused_moe_asm_wna16 import fused_experts_asm_impl
    from aiter.ops.shuffle import asm_shuffle_weight_b8

    plan, problem = ctx.plan, ctx.problem
    cfg = plan.extra.get("moe_cfg") or {}
    if "SOL_ID1" not in cfg or "SOL_ID2" not in cfg:
        return None  # no usable solution in the config: do not launch
    out_dtype = ctx.dtype or ctx.hidden_states.dtype
    if out_dtype == torch.float32:
        raise ValueError(
            "fused_moe(asm): fp32 output is unsupported; asm supports only "
            "fp16 and bf16 output"
        )
    is_w8a8 = problem.quant_kind in ("w8a8", "w8a8_block")
    use_shuffle = int(cfg.get("USE_SHUFFLE", problem.use_shuffle) or 0)
    padded_k = cfg.get("PADDED_K")
    w1, w2 = ctx.w1, ctx.w2
    # Weights not already in asm_shuffle layout are shuffled here.
    if use_shuffle and getattr(problem, "weight_layout", "raw") != "asm_shuffle":
        # The forwarded function does not pad weights (it asserts
        # padded_k == w1.shape[2]), so zero-pad first (w1 dim=2, w2 dim=1),
        # then run asm_shuffle_weight_b8 for stage 1/2, as aiter_moe_shfl_weight does.
        import torch as _torch

        def _padk(x, dim):
            # Zero-pad like aiter.moe._pad_tensor_dim: F.pad is unsafe on float8
            # dtypes, so copy into a zero tensor instead.
            if padded_k is None or x.shape[dim] >= int(padded_k):
                return x
            shape = list(x.shape)
            shape[dim] = int(padded_k)
            out = _torch.zeros(shape, dtype=x.dtype, device=x.device)
            idx = [slice(None)] * x.dim()
            idx[dim] = slice(0, int(x.shape[dim]))
            out[tuple(idx)] = x
            return out

        w1 = asm_shuffle_weight_b8(_padk(w1, 2), stage=1)
        w2 = asm_shuffle_weight_b8(_padk(w2, 1), stage=2)
    return fused_experts_asm_impl(
        ctx.hidden_states, w1, w2, ctx.topk_weight, ctx.topk_ids,
        dtype=out_dtype,
        activation=problem.activation,
        use_fp8_w8a8=is_w8a8 and problem.q_dtype_a == "fp8",
        use_int8_w8a8=is_w8a8 and problem.q_dtype_a == "int8",
        per_channel_quant=problem.block_shape is None,
        global_num_experts=problem.E,
        expert_map=_expert_mask_to_map(ctx.expert_mask),
        w1_scale=ctx.w1_scale,
        w2_scale=ctx.w2_scale,
        a1_scale=ctx.a1_scale,
        a2_scale=ctx.a2_scale,
        block_shape=list(problem.block_shape) if problem.block_shape else None,
        use_shuffle=use_shuffle,
        solution_id=f"{cfg['SOL_ID1']}+{cfg['SOL_ID2']}",
        padded_k=padded_k,
    )


def _ck_execute(ctx: FusedMoeExecContext):
    """CK 2-stage execution: sorting -> ck_moe_stage1 -> activation -> ck_moe_stage2.

    Requires module_moe_ck2stages to be built.
    """
    import torch

    import aiter
    from aiter import dtypes

    plan, problem = ctx.plan, ctx.problem
    hidden_states = ctx.hidden_states
    device = hidden_states.device
    E, model_dim, inter_dim = ctx.w2.shape
    M, topk = ctx.topk_ids.shape
    out_dtype = ctx.dtype or hidden_states.dtype

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _, moe_buf = _moe_sorting(
        ctx.topk_ids, ctx.topk_weight, problem.E, model_dim, dtypes.fp32,
        block_size=plan.block_m, expert_mask=ctx.expert_mask,
    )
    inter_states = torch.empty((M * topk, ctx.w1.shape[1] // 2), dtype=out_dtype, device=device)
    aiter.ck_moe_stage1(
        hidden_states, ctx.w1, ctx.w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        inter_states, topk,
        ctx.w1_scale, ctx.a1_scale,
        plan.block_m, None, plan.act_op,
    )
    # Apply the activation outside the kernel when act_op == 0 (otherwise SiLU is fused in stage1)
    if plan.act_op == 0:
        act_out = _external_act(problem.activation, inter_dim,
                                inter_states.view(M, topk, -1))
        inter_states = act_out.reshape(M * topk, -1).to(out_dtype)
    out = torch.zeros((M, model_dim), dtype=dtypes.fp32, device=device)
    aiter.ck_moe_stage2(
        inter_states, ctx.w1, ctx.w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out, topk,
        ctx.w2_scale, ctx.a2_scale,
        plan.block_m, sorted_weights,
    )
    return out.to(out_dtype)


def _expert_mask_to_map(expert_mask):
    """Convert expert_mask (global 0/1 mask) to expert_map (global id -> local id,
    -1 = not on this rank).

    The triton and moe_c entry points take an expert_map and recover the 0/1
    mask themselves with ``expert_map >= 0``; handing them the raw mask would
    mark every expert local (0 >= 0) and index past the rank's weights. Local
    ids follow the same cumulative numbering as torch_moe / moe_sorting.
    """
    if expert_mask is None:
        return None
    import torch

    local_id = expert_mask.cumsum(0, dtype=torch.int32) - 1
    return torch.where(expert_mask != 0, local_id, torch.full_like(local_id, -1))


def _moe_c_execute(ctx: FusedMoeExecContext):
    """MOE_C whole-chain execution: forward to moe_c_fused_experts, shuffling
    the weights into the marlin layout first when the quant family needs it."""
    from aiter.fused_moe_c import moe_c_fused_experts

    problem, plan = ctx.problem, ctx.plan
    out_dtype = ctx.dtype or ctx.hidden_states.dtype
    if out_dtype != ctx.hidden_states.dtype:
        raise ValueError(
            "fused_moe(moe_c): dtype must equal activation dtype; "
            f"got dtype={out_dtype}, activation dtype={ctx.hidden_states.dtype}"
        )
    w1, w2 = ctx.w1, ctx.w2
    # Weights marked "marlin" were already shuffled by the same functions; do
    # not shuffle twice.
    if problem.quant_kind in ("w8a16", "w8a8") and problem.weight_layout != "marlin":
        w1, w2 = _moe_c_shuffle_weights(problem, w1, w2)
    extra = plan.extra
    # int4 families never reach moe_c (plan() rejects them), so no zero points.
    w1_zp = w2_zp = None
    block_shape = list(problem.block_shape) if problem.block_shape else None
    out = moe_c_fused_experts(
        ctx.hidden_states,
        w1,
        w2,
        ctx.topk_weight,
        ctx.topk_ids,
        # inplace=True would overwrite hidden_states; the caller copies the
        # result into `output` instead.
        inplace=False,
        activation=problem.activation,
        use_fp8_w8a8=extra.get("use_fp8_w8a8", False),
        use_int8_w8a8=extra.get("use_int8_w8a8", False),
        use_int8_w8a16=extra.get("use_int8_w8a16", False),
        use_int4_w4a16=extra.get("use_int4_w4a16", False),
        global_num_experts=problem.E,
        expert_map=_expert_mask_to_map(ctx.expert_mask),
        w1_scale=ctx.w1_scale,
        w2_scale=ctx.w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=ctx.a1_scale,
        a2_scale=ctx.a2_scale,
        block_shape=block_shape,
    )
    return out


def _moe_c_shuffle_weights(problem, w1, w2):
    """Shuffle natural-layout weights into the marlin layout using the layout
    functions in aiter.ops.shuffle."""
    import torch

    from aiter.ops import shuffle as _shuffle

    if problem.quant_kind == "w4a16":
        w1s = _shuffle.w4a16_marlin_weight_1(w1)
        w2s = _shuffle.w4a16_marlin_weight_2(w2)
        # marlin returns a uint32-viewed tensor; restore the uint8 layout
        w1s = w1s.view(-1).view(torch.uint8).view(w1.shape).contiguous()
        w2s = w2s.view(-1).view(torch.uint8).view(w2.shape).contiguous()
        return w1s, w2s
    if problem.quant_kind == "w8a16":
        w1s = _shuffle.w8a16_marlin_weight_1(w1)
        w2s = _shuffle.w8a16_marlin_weight_2(w2)
        return w1s.contiguous(), w2s.contiguous()
    if problem.quant_kind == "w8a8":
        # Both weights use moe_layout_shuffle_gemm2, as in aiter_moe_shfl_weight
        # for MOE_C + W8A8.
        w1s = _shuffle.moe_layout_shuffle_gemm2(w1).contiguous().view(*w1.shape)
        w2s = _shuffle.moe_layout_shuffle_gemm2(w2).contiguous().view(*w2.shape)
        return w1s, w2s
    return w1, w2


def _triton_execute(ctx: FusedMoeExecContext):
    """Triton whole-chain execution through boltops fused_experts_impl.

    Uses the same entry and keyword mapping as the TRITON branch of aiter_moe.
    fused_experts_impl looks up the installed ``triton_<capability>`` JSON
    configs itself, so no config object is passed in.
    """
    from boltops.fused_moe.triton.fused_moe import fused_experts_impl

    problem, plan = ctx.problem, ctx.plan
    extra = plan.extra
    use_fp8_w8a8 = extra.get("use_fp8_w8a8", False)
    use_int8_w8a8 = extra.get("use_int8_w8a8", False)
    use_int8_w8a16 = extra.get("use_int8_w8a16", False)
    use_int4_w4a8 = extra.get("use_int4_w4a8", False)
    block_shape = extra.get("block_shape", problem.block_shape)
    # Same as aiter_moe: per-channel only for channelwise quant without block_shape
    # (blockwise w8a8_block passes its 2-D group scales through block_shape).
    per_channel_quant = (
        (use_int8_w8a8 or use_fp8_w8a8 or use_int8_w8a16 or use_int4_w4a8)
        and block_shape is None
    )
    return fused_experts_impl(
        ctx.hidden_states,
        ctx.w1,
        ctx.w2,
        ctx.topk_weight,
        ctx.topk_ids,
        output_dtype=ctx.dtype or ctx.hidden_states.dtype,
        inplace=False,
        activation=problem.activation,
        # Pass is_gated explicitly. Inferring it from the activation is wrong
        # for g1u0 weights (w1 N == inter_dim, used by EP tests): "silu" would
        # be treated as gated, the width misread, and the result silently wrong.
        is_gated=problem.g1u1,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=extra.get("use_int4_w4a16", False),
        use_int4_w4a8=use_int4_w4a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=problem.E,
        expert_map=_expert_mask_to_map(ctx.expert_mask),
        w1_scale=ctx.w1_scale,
        w2_scale=ctx.w2_scale,
        w1_zp=None,
        w2_zp=None,
        a1_scale=ctx.a1_scale,
        a2_scale=ctx.a2_scale,
        block_shape=list(block_shape) if block_shape is not None else None,
        routed_scaling_factor=1.0,
        gemm1_alpha=None,
        gemm1_limit=None,
    )


# ---------------------------------------------------------------------------
# Backend registry + planning
# ---------------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, BackendEntry] = {}

# asm is admitted by quant_kind alone (see _asm_match). int4 families are
# rejected at the start of plan(). none, w8a8 and w8a8_block run through the
# host forwarder, whose tuned solutions match the blobs' launch bounds.


def _asm_match(problem: FusedMoeProblem) -> bool:
    if problem.is_ep or problem.doweight_stage1 or problem.splitk:
        return False
    if not _activation_ok(problem):
        return False
    if _looks_like_mxfp(problem):
        return False
    # w8a8_block is handled by fused_experts_asm_impl too (blockwise scales,
    # K padding, shuffle tables, PADDED_K), so the router does not repeat them.
    # Shapes without a tuned row return None in _asm_lookup and fall through
    # to the next candidate.
    return problem.quant_kind in ("none", "w8a8", "w8a8_block")


def _asm_lookup(problem: FusedMoeProblem) -> dict | None:
    """Read the asm tuned table through aiter.moe._try_get_asm_config, the same
    function aiter_moe uses (K padding, forced USE_SHUFFLE and PADDED_K are
    handled there). SOL_ID1/SOL_ID2/BLOCK_SIZE_M are copied flat into the
    result and the full config is kept in plan.extra["moe_cfg"].
    Returns None if there is no tuned config."""
    from aiter.moe import _try_get_asm_config

    qt = _to_moe_quant_type(problem)
    if qt is None:
        return None
    cfg = _try_get_asm_config(
        qt,
        problem.M,
        problem.E,
        problem.inter_dim,
        problem.model_dim,
        problem.topk,
        _moe_block_size(problem),
        use_shuffle=1 if problem.weight_layout == "asm_shuffle" else 0,
    )
    if cfg is None:
        return None
    return {**cfg, "moe_cfg": cfg, "persist1": 0, "persist2": 0}


def _ck_match(problem: FusedMoeProblem) -> bool:
    if problem.doweight_stage1 or problem.splitk:
        return False
    if not _activation_ok(problem):
        return False
    if _looks_like_mxfp(problem):
        return False
    # Supported: no-quant, int8/fp8 per-token and per-tensor (w8a8), fp8
    # activation + int4 weight (a8w4), and blockwise int8 (w8a8_block).
    return problem.quant_kind in ("none", "w8a8", "w8a8_block", "a8w4")


def _ck_module_ready() -> bool:
    """Whether module_moe_ck2stages is built.

    The moe_ck_2stages_gemm_impl sources depend on CK's device_moe_gemm.hpp,
    which the CK submodule in this checkout does not ship, so the module may
    fail to build. While it is absent the ck entry yields to later candidates
    instead of triggering a JIT build failure at call time. Remove this gate
    once the CK dependency lands and module_moe_ck2stages builds.
    """
    return os.path.exists(
        os.path.join(AITER_ROOT_DIR, "aiter", "jit", "module_moe_ck2stages.so")
    )


def _ck_lookup(problem: FusedMoeProblem) -> dict | None:
    # Return None while module_moe_ck2stages.so is missing, even if the
    # table has rows for this shape.
    if not _ck_module_ready():
        return None
    from aiter.moe import _try_get_ck_config

    qt = _to_moe_quant_type(problem)
    if qt is None:
        return None
    cfg = _try_get_ck_config(
        qt,
        problem.M,
        problem.E,
        problem.inter_dim,
        problem.model_dim,
        problem.topk,
        list(problem.block_shape) if problem.block_shape else None,
    )
    if cfg is None:
        return None
    return {
        "ck_cfg": cfg,
        "sol_id": cfg.get("solution_id", 0),
        "block_m": problem.block_size_M or 32,
    }


def _moe_c_match(problem: FusedMoeProblem) -> bool:
    if problem.doweight_stage1 or problem.splitk:
        return False
    if problem.is_ep:
        # moe_c remaps expert ids through expert_map, so blocks of non-local
        # experts reach the marlin kernels with expert id -1. The w8a16 kernels
        # do not skip -1 and hit a VMFault (E_local=128 of 256, N=768), so EP
        # is left to triton.
        return False
    if not _activation_ok(problem):
        return False
    if _looks_like_mxfp(problem):
        return False
    # w8a8 candidate order follows aiter_moe (asm > moe_c > triton). moe_c
    # w8a8 per-token has a known numeric error (cos ~0.01-0.10); the same
    # error occurs when calling the host op directly.
    # w4a16 is excluded: the marlin op needs a non-None b_zeros, which the
    # fused_moe arguments cannot provide.
    # moe_c takes pre-shuffled weights; see _moe_c_shuffle_weights.
    return problem.quant_kind in ("w8a16", "w8a8")


def _moe_c_lookup(problem: FusedMoeProblem) -> dict | None:
    """Check for a tuned marlin config through aiter.moe._try_get_moe_c_config.

    moe_c re-reads the config itself at execute time, so this call only decides
    whether moe_c is a candidate: None means no tuned config. The returned
    flags are consumed by _moe_c_execute.
    """
    from aiter.moe import _try_get_moe_c_config

    qt = _to_moe_quant_type(problem)
    if qt is None:
        return None
    cfg = _try_get_moe_c_config(
        qt,
        problem.M,
        problem.E,
        problem.inter_dim,
        _moe_block_size(problem),
        k=problem.model_dim,
        top_k=problem.topk,
        # Must match the execute-time key str(compute_dtype); without it fp16
        # problems match bf16 rows and the execute-time lookup asserts.
        indtype=str(_torch_dtype(problem)),
    )
    if cfg is None:
        return None
    return {
        "use_int4_w4a16": problem.quant_kind == "w4a16",
        "use_int8_w8a16": problem.quant_kind == "w8a16",
        "use_fp8_w8a8": problem.quant_kind == "w8a8" and problem.q_dtype_a == "fp8",
        "use_int8_w8a8": problem.quant_kind == "w8a8" and problem.q_dtype_a == "int8",
        "use_shuffle": problem.use_shuffle,
        "tuned_config": cfg,
    }


def _triton_match(problem: FusedMoeProblem) -> bool:
    # Last-resort backend. w4a16/w4a8 are excluded: the gptq kernel needs
    # has_zp=True with explicit qzeros, which the fused_moe arguments cannot
    # provide.
    if _looks_like_mxfp(problem):
        return False
    return problem.quant_kind in ("none", "w8a8", "w8a8_block", "w8a16")


def _triton_lookup(problem: FusedMoeProblem) -> dict | None:
    """Check for a tuned Triton config through aiter.moe._try_get_triton_config.

    W16A16 always returns {} (non-None). Other families return None when no
    triton_<capability> JSON covers the shape, so the next candidate is tried.
    The config is returned as plan.extra["triton_cfg"].
    """
    from aiter.moe import _try_get_triton_config

    qt = _to_moe_quant_type(problem)
    if qt is None:
        return None
    # Use the same key as the boltops executor: local expert count, inter_dim,
    # model_dim and topk.
    cfg = _try_get_triton_config(
        qt, problem.M, problem.local_E or problem.E, problem.inter_dim,
        _moe_block_size(problem), k=problem.model_dim, top_k=problem.topk)
    if cfg is None:
        return None
    return {**_triton_flags(problem), "triton_cfg": cfg}


def _triton_flags(problem: FusedMoeProblem) -> dict:
    """Executor flags for the boltops triton chain. Shared by _triton_lookup and
    the untuned fallback so both build identical arguments."""
    is_w8a8 = problem.quant_kind in ("w8a8", "w8a8_block")
    return {
        "use_fp8_w8a8": is_w8a8 and problem.q_dtype_a == "fp8",
        "use_int8_w8a8": is_w8a8 and problem.q_dtype_a == "int8",
        "use_int4_w4a8": problem.quant_kind == "w4a8",
        "use_int4_w4a16": problem.quant_kind == "w4a16",
        "use_int8_w8a16": problem.quant_kind == "w8a16",
        "per_channel_quant": problem.quant_kind in ("w8a8", "w4a8"),
        "block_shape": problem.block_shape,
    }


BACKEND_REGISTRY["asm"] = BackendEntry("asm", _asm_match, _asm_lookup, _asm_execute)
# ck is registered but not in any _CANDIDATE_ORDER row, because
# module_moe_ck2stages.so is not built here (see _ck_module_ready). Add it back
# to the rows once the module builds.
BACKEND_REGISTRY["ck"] = BackendEntry("ck", _ck_match, _ck_lookup, _ck_execute)
BACKEND_REGISTRY["moe_c"] = BackendEntry("moe_c", _moe_c_match, _moe_c_lookup, _moe_c_execute)
BACKEND_REGISTRY["triton"] = BackendEntry("triton", _triton_match, _triton_lookup, _triton_execute)

# Candidate backends per quant family, highest priority first.
_CANDIDATE_ORDER = {
    "none": ("asm", "triton", "moe_c"),
    "w8a8": ("asm", "moe_c", "triton"),
    "w8a8_block": ("asm", "triton"),
    "w8a16": ("moe_c", "triton"),
}

# Weight layouts each backend can consume. A backend is skipped when the
# weights were prepared in a layout it does not support; the same check
# applies to backends recommended by the route table.
BACKEND_LAYOUTS = {
    "asm": ("raw", "asm_shuffle"),
    "ck": ("raw",),
    "moe_c": ("raw", "marlin"),
    "triton": ("raw",),
}

# ---------------------------------------------------------------------------
# Untuned-shape fallback
#
# When no backend has a tuned config for a shape, run boltops triton with its
# own default config and log one warning per shape. Set
# AITER_FMOE_TRITON_FALLBACK=strict to raise NotImplementedError instead. The
# variable is read at import time, so set it before `import aiter`.
#
# boltops fused_experts_impl falls back to its default config on a table miss
# by itself and gives correct results on an EP int8 per-token shape. The
# in-tree aiter.ops.triton path does not (cos 0.975 on the same shape), so it
# is not used here.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_TRITON_FALLBACK_MODE = os.environ.get(
    "AITER_FMOE_TRITON_FALLBACK", "warn").strip().lower()
_FALLBACK_STRICT = _TRITON_FALLBACK_MODE in ("strict", "raise", "off", "0")
if not _FALLBACK_STRICT and _TRITON_FALLBACK_MODE != "warn":
    logger.warning(
        "AITER_FMOE_TRITON_FALLBACK=%r is not recognized; using 'warn'. "
        "Use 'strict' to raise on untuned shapes.", _TRITON_FALLBACK_MODE)
# Shapes already warned about in this process. Test runners count fallbacks by
# grepping the log for "fused_moe fallback:".
_fallback_seen: set[tuple] = set()


def _fallback_allowed(problem: FusedMoeProblem) -> bool:
    if _FALLBACK_STRICT:
        return False
    if problem.quant_kind not in _CANDIDATE_ORDER:
        return False  # no candidate order for this family
    if _looks_like_mxfp(problem):
        return False
    if problem.weight_layout not in BACKEND_LAYOUTS["triton"]:
        return False  # triton cannot consume this weight layout
    return True


def _triton_fallback_params(problem: FusedMoeProblem) -> dict | None:
    """Same flags as _triton_lookup, but without a tuned config.

    triton_cfg=None is fine: boltops looks up its own configs at execute time
    and uses its default on a miss. The "fallback" flag in plan.extra lets
    route logs count fallback uses.
    """
    if _to_moe_quant_type(problem) is None:
        return None
    return {**_triton_flags(problem), "triton_cfg": None, "fallback": True}


def _warn_triton_fallback(problem: FusedMoeProblem) -> None:
    key = (
        problem.quant_kind, problem.E, problem.local_E, problem.inter_dim,
        problem.model_dim, problem.topk, problem.q_dtype_a, problem.dtype,
        problem.is_ep,
    )
    if key in _fallback_seen:
        return
    _fallback_seen.add(key)
    local_e = problem.local_E or problem.E  # local_E == 0 means "same as E"
    logger.warning(
        "fused_moe fallback: no tuned triton config for this shape -- running "
        "boltops triton DEFAULT config (performance NOT tuned; tune it or set "
        "AITER_FMOE_TRITON_FALLBACK=strict to raise instead): "
        "E_global=%d E_local=%d N=%d K=%d topk=%d M=%d quant_kind=%s "
        "q_dtype_a=%s dtype=%s act=%s is_ep=%s (executor JSON key space: "
        "triton_3_6/E=%d,N=%d,K=%d,topk=%d,arch=%s,dtype=...)",
        problem.E, local_e, problem.inter_dim, problem.model_dim,
        problem.topk, problem.M, problem.quant_kind, problem.q_dtype_a,
        problem.dtype, problem.activation, problem.is_ep,
        local_e, problem.inter_dim, problem.model_dim, problem.topk,
        problem.arch,
    )

# ---------------------------------------------------------------------------
# Backend route table
#
# Maps a problem shape and token count to the fastest backend found by tuning.
# Written by moe_tune/unified_tune/route_table.py; the column layout must stay
# in sync with it. A missing file or unparsable rows mean no entries, and
# plan() then uses _CANDIDATE_ORDER. Set FMOE_ROUTE_LOG=1 to print which
# route was used.
# ---------------------------------------------------------------------------

_ROUTE_GROUP = ("arch", "quant_type", "indtype", "inter_dim", "model_dim",
                "expert", "topk", "q_size_n", "q_size_k")


def _route_csv_path() -> str:
    return os.path.join(AITER_ROOT_DIR, "aiter", "configs", "tuned_fused_moe_route.csv")


def _route_quant_key(problem: FusedMoeProblem) -> str | None:
    """Map quant_kind and q_dtype_a to the quant_type name used in the route CSV
    (tune_spec.QUANTS). int4 families never get here."""
    kind = problem.quant_kind
    if kind == "none":
        return "no_quant"
    if kind == "w8a16":
        return "int8_w8a16"
    if kind == "w8a8":
        return "f8_w8a8_channel" if problem.q_dtype_a == "fp8" else "int8_w8a8_channel"
    if kind == "w8a8_block":
        return "f8_w8a8_block" if problem.q_dtype_a == "fp8" else "int8_w8a8_block"
    return None


@functools.lru_cache(maxsize=2)
def _load_route_table(path: str):
    """Read the route CSV into {group key: (sorted tokens, backends, times)};
    None if the file is missing or has no valid rows."""
    import csv
    if not os.path.exists(path):
        return None
    groups = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                key = tuple(str(row[c]).strip() for c in _ROUTE_GROUP)
                token = int(row["token"])
                backend = str(row["backend"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if backend not in BACKEND_REGISTRY:
                continue
            entry = groups.setdefault(key, ([], [], []))
            entry[0].append(token)
            entry[1].append(backend)
            entry[2].append(row.get("time_us", ""))
    if not groups:
        return None
    for entry in groups.values():
        order = sorted(range(len(entry[0])), key=lambda i: entry[0][i])
        entry[0][:] = [entry[0][i] for i in order]
        entry[1][:] = [entry[1][i] for i in order]
        entry[2][:] = [entry[2][i] for i in order]
    return groups


def _lookup_route_table(problem: FusedMoeProblem) -> tuple | None:
    """Return (backend, time_us) for the nearest tuned token count, or None.
    On a tie the smaller token count wins."""
    index = _load_route_table(_route_csv_path())
    if not index:
        return None
    quant = _route_quant_key(problem)
    if quant is None:
        return None
    block = problem.block_shape or (0, 0)
    key = (problem.arch, quant,
           "torch.float16" if problem.dtype == "fp16" else "torch.bfloat16",
           str(problem.inter_dim), str(problem.model_dim), str(problem.E),
           str(problem.topk), str(int(block[0])), str(int(block[1])))
    entry = index.get(key)
    if not entry:
        return None
    tokens, backends, times = entry
    token = int(problem.M)
    if token <= tokens[0]:
        idx = 0
    elif token >= tokens[-1]:
        idx = len(tokens) - 1
    else:
        from bisect import bisect_left
        pos = bisect_left(tokens, token)
        if tokens[pos] == token:
            idx = pos
        else:
            left, right = pos - 1, pos
            idx = left if token - tokens[left] <= tokens[right] - token else right
    return backends[idx], times[idx]


_STAGE_BACKENDS = ("asm", "ck")
_CHAIN_BACKENDS = ("moe_c", "triton")


_INT4_UNSUPPORTED_MSG = (
    "fused_moe: int4-weight families currently have no routable "
    "backend -- asm .co blobs fault on this arch (launch bounds), "
    "moe_c marlin requires explicit zero points, which fused_moe has "
    "no argument for, and the triton gptq kernel is only verified "
    "with has_zp=True group-32 scales. Use aiter_moe for these "
    "families."
)


def _supported_order(problem: FusedMoeProblem) -> tuple:
    """Return the candidate backends for the problem's quant family, highest
    priority first. Raise NotImplementedError for problems fused_moe rejects."""
    # int4-weight families are not supported: fused_moe has no zero-point
    # argument, which marlin and the gptq kernel need, and the stage .co blobs
    # fault on this arch. Use aiter_moe for them.
    if problem.quant_kind in ("w4a16", "w4a8", "a8w4"):
        raise NotImplementedError(_INT4_UNSUPPORTED_MSG)
    if problem.doweight_stage1:
        raise NotImplementedError(
            "fused_moe: doweight_stage1=True is not supported because the "
            "1-stage kernel backends are not available"
        )
    order = _CANDIDATE_ORDER.get(problem.quant_kind)
    if order is None:
        raise NotImplementedError(
            f"fused_moe routing: unsupported quantization family "
            f"{problem.quant_kind!r} (quant_type={problem.quant_type!r}, "
            f"q_dtype_a={problem.q_dtype_a!r}, q_dtype_w={problem.q_dtype_w!r})"
        )
    return order


def _plan_from_route_table(problem: FusedMoeProblem) -> FusedMoePlan | None:
    """Use the backend recommended by the route table if it can run this
    problem. Return None if there is no table or row, the backend does not
    match, the weight layout is unsupported, or it has no tuned config."""
    rec = _lookup_route_table(problem)
    if rec is None:
        return None
    name = rec[0]
    entry = BACKEND_REGISTRY.get(name)
    if (entry is None or not entry.match(problem)
            or problem.weight_layout not in BACKEND_LAYOUTS[name]):
        return None
    params = entry.lookup(problem)
    routed = _build_plan(name, params, problem) if params is not None else None
    if routed is not None and os.environ.get("FMOE_ROUTE_LOG"):
        print(f"ROUTE-TABLE use backend={name} token={problem.M} time_us={rec[1]}",
              flush=True)
    return routed


def _plan_from_candidates(problem: FusedMoeProblem, order: tuple) -> FusedMoePlan | None:
    """Return the first backend in `order` that matches and has a tuned config."""
    for name in order:
        entry = BACKEND_REGISTRY[name]
        if not entry.match(problem):
            continue
        if problem.weight_layout not in BACKEND_LAYOUTS[name]:
            continue  # backend cannot consume this weight layout
        params = entry.lookup(problem)
        if params is None:
            continue  # matched, but no tuned config for this shape
        routed = _build_plan(name, params, problem)
        if routed is not None:
            return routed
    return None


def _plan_from_fallback(problem: FusedMoeProblem) -> FusedMoePlan | None:
    """No tuned config anywhere: run triton with its default config and warn
    (see "Untuned-shape fallback" above). Return None if not allowed."""
    if not _fallback_allowed(problem):
        return None
    params = _triton_fallback_params(problem)
    if params is None:
        return None
    fallback = _build_plan("triton", params, problem)
    if fallback is None:
        return None
    _warn_triton_fallback(problem)
    if os.environ.get("FMOE_ROUTE_LOG"):
        print("ROUTE-FALLBACK use backend=triton(default-config)", flush=True)
    return fallback


def _no_backend_error(problem: FusedMoeProblem) -> NotImplementedError:
    message = (
        f"fused_moe routing: no backend covers this problem "
        f"(quant_kind={problem.quant_kind!r}, dtype={problem.dtype!r}, "
        f"q_dtype_a={problem.q_dtype_a!r}, q_dtype_w={problem.q_dtype_w!r}, "
        f"activation={problem.activation!r}, arch={problem.arch!r}, "
        f"weight_layout={problem.weight_layout!r})"
    )
    if problem.q_dtype_w == "int4":
        message += "\n" + _INT4_UNSUPPORTED_MSG
    if _FALLBACK_STRICT:
        message += (
            "\nAITER_FMOE_TRITON_FALLBACK=warn (the default) would run this "
            "shape on the boltops triton default config instead of raising."
        )
    return NotImplementedError(message)


def plan(problem: FusedMoeProblem) -> FusedMoePlan:
    """Route a normalized problem to a stage-level FusedMoePlan.

    Order: route table recommendation, then the static candidate order, then
    the untuned-shape triton fallback. Raises NotImplementedError when no
    registered backend covers the problem (per_1x32 / MXFP families, a16w4,
    unsupported activations, ...).
    """
    order = _supported_order(problem)
    routed = _plan_from_route_table(problem)
    if routed is None:
        routed = _plan_from_candidates(problem, order)
    if routed is None:
        routed = _plan_from_fallback(problem)
    if routed is None:
        raise _no_backend_error(problem)
    return routed


def _build_plan(chosen: str, params: dict, problem: FusedMoeProblem) -> FusedMoePlan:
    if chosen in _STAGE_BACKENDS:
        return FusedMoePlan(
            stage1_backend=chosen,
            stage2_backend=chosen,
            block_m=int(params.get("BLOCK_SIZE_M", params.get("block_m", 32))),
            act_op=1 if problem.activation == "silu" else 0,
            sol_id1=params.get("SOL_ID1", params.get("sol_id")),
            sol_id2=params.get("SOL_ID2", params.get("sol_id")),
            mode=int(params.get("mode", 0)),
            use_shuffle=int(problem.use_shuffle),
            flat=False,
            extra=dict(params),
        )
    if chosen in _CHAIN_BACKENDS:
        chain_keys = (
            "use_int4_w4a16", "use_int8_w8a16", "use_fp8_w8a8", "use_int8_w8a8",
            "use_int4_w4a8", "per_channel_quant", "block_shape", "tuned_config",
            "triton_cfg", "fallback",
        )
        return FusedMoePlan(
            stage1_backend=None,
            stage2_backend=None,
            chained_backend=chosen,
            act_op=1 if problem.activation == "silu" else 0,
            mode=0,
            use_shuffle=int(problem.use_shuffle),
            flat=False,
            extra={k: v for k, v in params.items() if k in chain_keys},
        )
    return None
