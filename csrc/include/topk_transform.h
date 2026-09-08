// These interfaces were ported from sglang.
#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

#include <torch/extension.h>


void fast_topk_interface(
    const torch::Tensor& score,
    torch::Tensor& indices,
    const torch::Tensor& lengths,
    std::optional<torch::Tensor> row_starts_opt = std::nullopt);

void fast_topk_transform_interface(
    const torch::Tensor& score,
    const torch::Tensor& lengths,
    torch::Tensor& dst_page_table,
    const torch::Tensor& src_page_table,
    const torch::Tensor& cu_seqlens_q,
    std::optional<torch::Tensor> row_starts_opt = std::nullopt);

void fast_topk_transform_ragged_interface(
    const torch::Tensor& score,
    const torch::Tensor& lengths,
    torch::Tensor& topk_indices_ragged,
    const torch::Tensor& topk_indices_offset,
    std::optional<torch::Tensor> row_starts_opt = std::nullopt);
