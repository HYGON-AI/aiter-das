#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <torch/all.h>

namespace aiter {
namespace ptq_i8 {

// Per-token dynamic int8 quantization with golden-matching rounding (rintf).
// out:   [rows, cols] int8, rows = input.numel() / input.size(-1)
// input: [*, cols] bf16/fp16, contiguous
// scale: [rows] float32
void per_token_quant_i8(torch::Tensor& out,
                        const torch::Tensor& input,
                        torch::Tensor& scale);

} // namespace ptq_i8
} // namespace aiter
