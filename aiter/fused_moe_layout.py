# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Weight-layout markers and a one-stop weight preparation API for
# aiter.fused_moe.
#
# Contract:
#   * A prepared tensor carries a ``weight_layout`` attribute. Unmarked tensors
#     are treated as raw, so existing callers are unaffected.
#   * Layout families:
#       raw          natural layout (no marker / explicit)
#       marlin       moe_c family (w8a16 / w8a8 shuffle layout, host parity)
#       asm_shuffle  asm family (asm_shuffle_weight_b8 stage1/stage2 layout)
#   * In the prepare API, an explicit ``layout=`` wins over ``backend=``
#     inference. With neither, the weights are returned unchanged (raw).
#   * This module only wraps the layout functions in aiter.ops.shuffle. It must
#     not import fused_moe or fused_moe_router (import cycle).
#   * Scale shuffling is backend-specific (moe_c w4a16 only), and int4 families
#     are not supported, so only the weights are handled here. ``block_shape``
#     is accepted but not used yet.
#
# Keep the marlin w8a8 branch in sync with the two places it mirrors:
# aiter.moe.aiter_moe_shfl_weight (MOE_C + W8A8, gemm2 for both weights) and
# fused_moe_router._moe_c_shuffle_weights (w8a8 branch).
"""Weight layout markers and weight preparation API for fused_moe."""

from typing import Callable, Optional, Tuple

import torch

from aiter.ops import shuffle as _shuffle

WEIGHT_LAYOUT_ATTR = "weight_layout"

RAW = "raw"
MARLIN = "marlin"
ASM_SHUFFLE = "asm_shuffle"

VALID_LAYOUTS = (RAW, MARLIN, ASM_SHUFFLE)

_BACKEND_TO_LAYOUT = {"moe_c": MARLIN, "asm": ASM_SHUFFLE}


def get_weight_layout(w1, w2=None) -> str:
    """Read the layout marker of a prepared weight pair; missing/None -> RAW.

    Raises when w1/w2 carry DIFFERENT markers (a half-shuffled pair is a
    configuration error, never a legal routing state).
    """
    l1 = getattr(w1, WEIGHT_LAYOUT_ATTR, None) or RAW
    if w2 is None:
        if l1 not in VALID_LAYOUTS:
            raise ValueError(f"unknown weight_layout {l1!r} on w1")
        return l1
    l2 = getattr(w2, WEIGHT_LAYOUT_ATTR, None) or RAW
    if l1 != l2:
        raise ValueError(
            f"fused_moe weight layout mismatch: w1={l1!r} w2={l2!r} "
            "(prepare both weights together via fused_moe_prepare_weights)")
    if l1 not in VALID_LAYOUTS:
        raise ValueError(f"unknown weight_layout {l1!r}")
    return l1


def _marlin_w8a16(w1, w2):
    # parity with fused_moe_router._moe_c_shuffle_weights w8a16 branch
    # (aiter_moe_shfl_weight INT8_W8A16: w8a16_marlin_weight_1/2 + contiguous).
    return (_shuffle.w8a16_marlin_weight_1(w1).contiguous(),
            _shuffle.w8a16_marlin_weight_2(w2).contiguous())


def _marlin_w8a8(w1, w2):
    # MOE_C + W8A8 shuffles both weights with moe_layout_shuffle_gemm2, as in
    # aiter.moe.aiter_moe_shfl_weight. Keep in sync with
    # fused_moe_router._moe_c_shuffle_weights (w8a8 branch).
    return (_shuffle.moe_layout_shuffle_gemm2(w1).contiguous().view(*w1.shape),
            _shuffle.moe_layout_shuffle_gemm2(w2).contiguous().view(*w2.shape))


def _asm_shuffle(w1, w2):
    # plain asm-table layout; PADDED_K shapes must be padded by the caller
    # FIRST (host asserts padded_k == w1.shape[2]); the router's live path in
    # _asm_execute still zero-pads+shuffles unmarked raw weights.
    return (_shuffle.asm_shuffle_weight_b8(w1, stage=1),
            _shuffle.asm_shuffle_weight_b8(w2, stage=2))


# (layout, quant_kind-need) -> shuffle pair. quant_kind=None: family-agnostic.
_LAYOUT_SHUFFLERS: dict[Tuple[str, Optional[str]], Callable] = {
    (MARLIN, "w8a16"): _marlin_w8a16,
    (MARLIN, "w8a8"): _marlin_w8a8,
    (ASM_SHUFFLE, None): _asm_shuffle,
}


def fused_moe_prepare_weights(w1: torch.Tensor,
                              w2: torch.Tensor,
                              layout: Optional[str] = None,
                              backend: Optional[str] = None,
                              quant_kind: Optional[str] = None,
                              q_dtype_a: str = "none",
                              block_shape=None):
    """Prepare a w1/w2 pair for fused_moe.

    Returns ``(w1', w2', ...)``-shuffled pair marked with ``weight_layout``:
      * ``layout=`` direct pass wins; ``backend="moe_c"|"asm"`` infers the
        layout family (marlin needs ``quant_kind`` in w8a16/w8a8);
      * both missing -> tensors returned unchanged (raw);
      * already marked with the target layout -> returned as-is (idempotent);
      * the result pair carries matching markers readable by
        ``get_weight_layout`` and by ``fused_moe`` routing.
    """
    if layout is None:
        if backend is None:
            return w1, w2
        if backend not in _BACKEND_TO_LAYOUT:
            raise ValueError(
                f"fused_moe_prepare_weights: cannot infer layout from "
                f"backend={backend!r} (supported: {sorted(_BACKEND_TO_LAYOUT)})")
        layout = _BACKEND_TO_LAYOUT[backend]
    if layout not in VALID_LAYOUTS:
        raise ValueError(
            f"fused_moe_prepare_weights: unknown layout {layout!r} "
            f"(valid: {VALID_LAYOUTS})")
    if layout == RAW:
        return w1, w2

    if get_weight_layout(w1, w2) == layout:
        return w1, w2  # idempotent: already prepared

    if layout == MARLIN:
        if quant_kind is None:
            raise ValueError(
                "fused_moe_prepare_weights: marlin layout needs quant_kind "
                "(w8a16|w8a8) to pick the host-matching shuffle pair")
        key = (MARLIN, quant_kind)
    else:
        key = (ASM_SHUFFLE, None)
    fn = _LAYOUT_SHUFFLERS.get(key)
    if fn is None:
        raise ValueError(
            f"fused_moe_prepare_weights: no shuffle registered for "
            f"layout={layout!r} quant_kind={quant_kind!r} "
            f"(int4 families are not supported)")
    o1, o2 = fn(w1, w2)
    setattr(o1, WEIGHT_LAYOUT_ATTR, layout)
    setattr(o2, WEIGHT_LAYOUT_ATTR, layout)
    return o1, o2
