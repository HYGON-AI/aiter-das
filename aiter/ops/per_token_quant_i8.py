# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Dedicated per-token int8 quantization (HIP).

Optimized backend for aiter.fused_moe_c.per_token_quant_int8; see
op_tests/per_token_quant_int8_opt.md for the analysis and measurements.
"""

import torch

from .. import dtypes
from ..jit.core import compile_ops


@compile_ops("module_quant", fc_name="per_token_quant_i8")
def c_per_token_quant_i8(
    out: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    pass


def per_token_quant_i8(x: torch.Tensor):
    """Per-row dynamic int8 quantization, (x_q int8, scale fp32[*shape, 1])."""
    shape = x.shape
    cols = shape[-1]
    x_q = torch.empty(shape, dtype=dtypes.i8, device=x.device)
    scale = torch.empty((*shape[:-1], 1), dtype=dtypes.fp32, device=x.device)
    c_per_token_quant_i8(x_q, x, scale)
    return x_q, scale
