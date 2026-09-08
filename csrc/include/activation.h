#pragma once
// SPDX-License-Identifier: MIT
 
#include <torch/extension.h>

namespace aiter {

void silu_and_mul(torch::Tensor &out, torch::Tensor &input);
void scaled_silu_and_mul(torch::Tensor &out, torch::Tensor &input, torch::Tensor &scale);
void gelu_and_mul(torch::Tensor &out, torch::Tensor &input);
void gelu_tanh_and_mul(torch::Tensor &out, torch::Tensor &input);
void swiglu_variant(torch::Tensor &out, torch::Tensor &input, float alpha,
                    float limit, int mode, int rows_per_block = 1,
                    int vec_size = 2);

} // namespace aiter
