# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# fused_moe 的权重布局标记、离线布局查询与一次性准备接口。
#
# 契约：
#   * 已准备的张量带 ``weight_layout`` 属性；未标记视为 raw，旧调用方不受影响。
#   * 合法布局只有三种：
#       raw          自然布局（不重排）
#       marlin       moe_c 家族布局
#       asm_shuffle  asm 家族布局
#   * 布局由 ``decide_layout`` 在加载期一次性决定，准备函数只按决策执行重排；
#     公开接口不提供强制布局入口（测试需要时直接调 ``_shared_shuffle_impl``）。
#   * 准备函数固定返回 ``(w1p, w2p, layout)``。
#   * 本模块不 import fused_moe / fused_moe_router 于模块级（避免循环导入），
#     需要时在函数内延迟 import。
"""Weight layout markers and one-shot preparation API for fused_moe."""

import logging
import weakref

import torch

WEIGHT_LAYOUT_ATTR = "weight_layout"

RAW = "raw"
MARLIN = "marlin"
ASM_SHUFFLE = "asm_shuffle"

VALID_LAYOUTS = (RAW, MARLIN, ASM_SHUFFLE)

logger = logging.getLogger(__name__)

# id(tensor) -> (layout, shape, dtype)；张量释放时由 weakref.finalize 清理。
_PREPARED_LAYOUTS = {}


def _attr_or_registry(t):
    """读布局标记：张量属性优先，属性丢失时回退到准备函数登记的弱引用表。

    回退表只为「准备过一次、随后属性被张量操作丢掉」这一场景兜底；
    普通调用方的张量不在表里，行为与之前完全一致。
    """
    val = getattr(t, WEIGHT_LAYOUT_ATTR, None)
    if val:
        return val
    rec = _PREPARED_LAYOUTS.get(id(t))
    if rec is not None and rec[1] == tuple(t.shape) and rec[2] == t.dtype:
        return rec[0]
    return None


def get_weight_layout(w1, w2=None) -> str:
    """Read the layout marker of a prepared weight pair; missing/None -> RAW.

    Raises when w1/w2 carry DIFFERENT markers (a half-shuffled pair is a
    configuration error, never a legal routing state).
    """
    l1 = _attr_or_registry(w1) or RAW
    if w2 is None:
        if l1 not in VALID_LAYOUTS:
            raise ValueError(f"unknown weight_layout {l1!r} on w1")
        return l1
    l2 = _attr_or_registry(w2) or RAW
    if l1 != l2:
        raise ValueError(
            f"fused_moe weight layout mismatch: w1={l1!r} w2={l2!r} "
            "(prepare both weights together via fused_moe_prepare_weights)")
    if l1 not in VALID_LAYOUTS:
        raise ValueError(f"unknown weight_layout {l1!r}")
    return l1


def _mark(w1, w2, layout):
    for t in (w1, w2):
        setattr(t, WEIGHT_LAYOUT_ATTR, layout)
        key = id(t)
        _PREPARED_LAYOUTS[key] = (layout, tuple(t.shape), t.dtype)
        try:
            weakref.finalize(t, _PREPARED_LAYOUTS.pop, key, None)
        except TypeError:
            pass  # 该对象不支持弱引用：只保留属性标记


def resolve_weight_layout(w1, w2, explicit=None) -> str:
    """显式 ``weight_layout`` 与张量属性交叉校验后返回生效布局。

    规则：
      * 显式参数优先；
      * 显式参数与张量属性同时存在且不一致 -> ValueError；
      * 张量未标记（raw）时接受显式参数（常见于属性丢失场景）；
      * 两者都没有 -> raw。
    """
    attr = get_weight_layout(w1, w2)
    if explicit is None:
        return attr
    if explicit not in VALID_LAYOUTS:
        raise ValueError(
            f"fused_moe: unknown weight_layout {explicit!r} "
            f"(valid: {VALID_LAYOUTS})")
    if attr != RAW and attr != explicit:
        raise ValueError(
            f"fused_moe: weight_layout={explicit!r} conflicts with the tensor "
            f"marker {attr!r}; pass only one of them or prepare the weights again")
    return explicit


def _build_problem(w1, w2, quant_type, dtype, topk, activation, expert_mask,
                   w1_scale=None, w2_scale=None, block_size_M=None,
                   doweight_stage1=False, splitk=0, quant_type_a=None,
                   quant_dtype_a=None):
    """用权重形状构造 FusedMoeProblem（不依赖激活张量，M 固定 1）。"""
    from aiter.fused_moe import _normalize_problem

    return _normalize_problem(
        hidden_states=None,
        w1=w1,
        w2=w2,
        topk_ids=None,
        expert_mask=expert_mask,
        activation=activation,
        quant_type=quant_type,
        quant_type_a=quant_type_a,
        quant_dtype_a=quant_dtype_a,
        block_size_M=block_size_M,
        doweight_stage1=doweight_stage1,
        splitk=splitk,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        M=1,
        topk=topk,
        dtype=dtype,
    )


def _shuffle_kind(problem):
    """FusedMoeProblem -> 共享核心的家族词汇（aiter.moe._shared_shuffle_impl）。"""
    kind = problem.quant_kind
    if kind == "w8a8":
        return "w8a8_fp8" if problem.q_dtype_a == "fp8" else "w8a8_int8"
    if kind == "w8a8_block":
        return "w8a8_fp8" if problem.q_dtype_a == "fp8" else "w8a8_int8"
    if kind == "none":
        return "none"
    return kind


def describe_fused_moe_layout(
    w1: torch.Tensor,
    w2: torch.Tensor,
    quant_type=None,
    w1_scale=None,
    w2_scale=None,
    dtype=torch.bfloat16,
    topk=8,
    activation="silu",
    expert_mask=None,
    backend="auto",
    max_num_tokens=None,
    block_size_M=None,
    doweight_stage1=False,
    splitk=0,
    quant_type_a=None,
    quant_dtype_a=None,
):
    """只读查询：返回布局决策（不重排、不修改权重）。

    返回字典：
      layout / family / reason / padded_k / original_k /
      feasible_layouts / weighted_gain / quant_kind / arch
    """
    from aiter.fused_moe_router import decide_layout, feasible_layouts

    problem = _build_problem(
        w1, w2, quant_type, dtype, topk, activation, expert_mask,
        w1_scale=w1_scale, w2_scale=w2_scale, block_size_M=block_size_M,
        doweight_stage1=doweight_stage1, splitk=splitk,
        quant_type_a=quant_type_a, quant_dtype_a=quant_dtype_a)
    dec = decide_layout(problem, backend=backend, max_num_tokens=max_num_tokens)
    feasible = feasible_layouts(problem)
    return {
        "layout": dec.layout,
        "family": dec.family,
        "reason": dec.reason,
        "padded_k": dec.padded_k,
        "original_k": dec.original_k,
        "feasible_layouts": sorted(feasible),
        "weighted_gain": dec.weighted_gain,
        "quant_kind": problem.quant_kind,
        "arch": problem.arch,
        "shape": {
            "E": problem.E,
            "local_E": problem.local_E,
            "inter_dim": problem.inter_dim,
            "model_dim": problem.model_dim,
            "topk": problem.topk,
        },
        "current_marker": get_weight_layout(w1, w2),
    }


def fused_moe_prepare_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    quant_type=None,
    w1_scale=None,
    w2_scale=None,
    dtype=torch.bfloat16,
    topk=8,
    activation="silu",
    expert_mask=None,
    backend="auto",
    max_num_tokens=None,
    block_size_M=None,
    doweight_stage1=False,
    splitk=0,
    quant_type_a=None,
    quant_dtype_a=None,
):
    """按决策一次性重排 w1/w2，返回 ``(w1p, w2p, layout)``。

    重复调用：传入的权重已带目标布局标记时，直接返回同一对张量，不做第二次
    重排。这个判断只看标记，不校验形状、``padded_k``、``topk`` 是否变化；决定
    布局的参数变了就要从原始权重重新准备，否则会把旧张量按新布局原样返回。
    不提供强制布局入口；需要覆盖决策的测试请直接调用
    ``aiter.moe._shared_shuffle_impl``。
    """
    from aiter.fused_moe_router import decide_layout

    problem = _build_problem(
        w1, w2, quant_type, dtype, topk, activation, expert_mask,
        w1_scale=w1_scale, w2_scale=w2_scale, block_size_M=block_size_M,
        doweight_stage1=doweight_stage1, splitk=splitk,
        quant_type_a=quant_type_a, quant_dtype_a=quant_dtype_a)
    dec = decide_layout(problem, backend=backend, max_num_tokens=max_num_tokens)
    layout = dec.layout
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"fused_moe_prepare_weights: bad decision layout {layout!r}")

    logger.info(
        "fused_moe prepare: quant_kind=%s arch=%s E=%d N=%d K=%d topk=%d "
        "layout=%s family=%s padded_k=%s reason=%s",
        problem.quant_kind, problem.arch, problem.E, problem.inter_dim,
        problem.model_dim, problem.topk, layout, dec.family, dec.padded_k,
        dec.reason)

    if layout == RAW:
        # raw 不做任何处理，只打标记，便于运行期校验
        _mark(w1, w2, RAW)
        return w1, w2, RAW

    if get_weight_layout(w1, w2) == layout:
        return w1, w2, layout

    from aiter.moe import _shared_shuffle_impl

    o1, o2, _s1, _s2 = _shared_shuffle_impl(
        w1, w2, quant_kind=_shuffle_kind(problem), padded_k=dec.padded_k,
        block_shape=problem.block_shape, layout=layout)
    _mark(o1, o2, layout)
    return o1, o2, layout
