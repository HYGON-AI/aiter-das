# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from torch import Tensor
from ..jit.core import compile_ops


MD_NAME = "module_activation"


@compile_ops("module_activation")
def silu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation")
def scaled_silu_and_mul(out: Tensor, input: Tensor, scale: Tensor) -> None: ...


@compile_ops("module_activation")
def gelu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation")
def gelu_tanh_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation")
def swiglu_variant(
    out: Tensor,
    input: Tensor,
    alpha: float,
    limit: float,
    mode: int,
    rows_per_block: int = 1,
    vec_size: int = 2,
) -> None: ...
