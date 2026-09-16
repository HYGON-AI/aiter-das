// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 
#include "rocm_ops.hpp"
#include "quant.h"
#include "per_token_quant_i8.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    QUANT_PYBIND PTQ_I8_PYBIND;
}
