// SPDX-License-Identifier: MIT
#pragma once

#include <torch/extension.h>

void moe_c_align_moe_align_block_size(torch::Tensor topk_ids,
                                      int64_t num_experts,
                                      int64_t block_size,
                                      torch::Tensor sorted_token_ids,
                                      torch::Tensor experts_ids,
                                      torch::Tensor num_tokens_post_pad);

void moe_c_align_sgl_moe_align_block_size(torch::Tensor topk_ids,
                                          int64_t num_experts,
                                          int64_t block_size,
                                          torch::Tensor sorted_token_ids,
                                          torch::Tensor experts_ids,
                                          torch::Tensor num_tokens_post_pad);
