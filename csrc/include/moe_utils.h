#pragma once
#include <torch/all.h>
#include <torch/extension.h>
#include "aiter_enum.h"
namespace aiter{
void topk_softmax(torch::Tensor &topk_weights, torch::Tensor &topk_indices,
                  torch::Tensor &token_expert_indices,
                  torch::Tensor &gating_output,
                  bool need_renorm);
void moe_sum(torch::Tensor &input, torch::Tensor &output);

                  
}

void biased_grouped_topk(torch::Tensor& gating_output,   // [num_tokens, num_experts]
                         torch::Tensor& correction_bias, // [num_expert]
                         torch::Tensor& topk_weights,    // [num_tokens, topk]
                         torch::Tensor& topk_ids,        // [num_tokens, topk]
                         int num_expert_group,
                         int topk_group,
                         bool renormalize,
                         const float routed_scaling_factor = 1.);

void grouped_topk(torch::Tensor& gating_output, // [num_tokens, num_experts]
                  torch::Tensor& topk_weights,  // [num_tokens, topk]
                  torch::Tensor& topk_ids,      // [num_tokens, topk]
                  int num_expert_group,
                  int topk_grp,
                  bool need_renorm,
                  bool is_softmax                   = true,
                  const float routed_scaling_factor = 1.);

std::vector<at::Tensor> moe_fused_gate(at::Tensor& input,
                                       at::Tensor& bias,
                                       at::Tensor& topk_weights,
                                       at::Tensor& topk_ids,
                                       int64_t num_expert_group,
                                       int64_t topk_group,
                                       int64_t topk,
                                       int64_t n_share_experts_fusion,
                                       double routed_scaling_factor);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);

void sgl_moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                              int64_t block_size,
                              torch::Tensor sorted_token_ids,
                              torch::Tensor experts_ids,
                              torch::Tensor num_tokens_post_pad);



