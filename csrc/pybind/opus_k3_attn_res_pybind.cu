// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Hygon Info Technologies Ltd. All rights reserved.

#ifndef __HIP_DEVICE_COMPILE__

#include "rocm_ops.hpp"
#include "opus_k3_attn_res.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    OPUS_K3_ATTN_RES_HCU_PYBIND;
}

#endif

