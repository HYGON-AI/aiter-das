// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (C) 2026, Hygon Info Technologies Ltd. All rights reserved.

#ifndef __HIP_DEVICE_COMPILE__

#include "rocm_ops.hpp"
#include "opus_gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    OPUS_GEMM_A16W16_HCU_PYBIND;
    OPUS_GEMM_A8W8_GFX938_PYBIND;
}

#endif
