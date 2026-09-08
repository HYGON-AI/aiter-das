#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define __quickreduce_device_inline__ __device__ __forceinline__
#define __quickreduce_launch_bounds_two_shot__ __launch_bounds__(256, 4)
#define __quickreduce_launch_bounds_one_shot__ __launch_bounds__(512, 4)
typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

namespace aiter {

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using fp32x2_t  = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using fp32x4_t  = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using fp32x8_t  = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using fp32x16_t = __attribute__((__vector_size__(16 * sizeof(float)))) float;



union BufferResource
{
    __quickreduce_device_inline__ constexpr BufferResource() : config(0x00020000U) {}

    __quickreduce_device_inline__ constexpr BufferResource(void* buffer_address,
                                                           uint32_t buffer_size)
        : address(buffer_address), range(buffer_size), config(0x00020000U)
    {
    }

    int32x4_t descriptor;
    struct
    {
        void* address; // 8B, out of which first 48b is address, and 16b is stride
        // (unused)
        uint32_t range;  // Byte range for the buffer resource
        uint32_t config; // Constant, DFMT=32b
    };
};


} // namespace aiter