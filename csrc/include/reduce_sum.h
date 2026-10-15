// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <torch/extension.h>

namespace aiter {

// Row-wise reduce-sum along the last dimension:
//   input: [..., D] contiguous (or last-dim contiguous)
//   out:   [...], same dtype as input
// waves_per_block: number of waves per block (= rows per block).
//   -1 -> auto select from batch size and CU count.
void reduce_sum(torch::Tensor& out,
                torch::Tensor& input,
                int64_t waves_per_block = -1);

} // namespace aiter
