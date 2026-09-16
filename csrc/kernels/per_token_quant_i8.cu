// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
//
// Dedicated per-token (per-row) dynamic int8 quantization for MoE w8a8
// channel-wise path. Semantics follow the Triton reference in
// aiter/fused_moe_c.py::per_token_quant_int8 / torch golden:
//   scale = max(row_absmax, 1e-10) / 127
//   q     = rint(x * 127 / absmax)   (round half to even, |q| <= 127)
// The generic dynamic_per_token_scaled_quant entry point must not be reused
// here: its int8 conversion truncates instead of rounding (46% mismatch vs
// golden), see op_tests/per_token_quant_int8_opt.md.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/hipcub.hpp>
#include <cstdint>

#include "hip_reduce.h"
#include "per_token_quant_i8.h"

namespace aiter {
namespace ptq_i8 {

namespace {

constexpr int32_t BLOCK_SIZE = 256;
constexpr int32_t VEC        = 8; // bf16/fp16 elements per 16B vector load

__device__ __forceinline__ float to_f32(uint16_t raw, bool is_bf16)
{
    if(is_bf16)
    {
        // bf16 is the upper 16 bits of fp32
        union
        {
            uint32_t u;
            float f;
        } r;
        r.u = static_cast<uint32_t>(raw) << 16;
        return r.f;
    }
    else
    {
        return __half2float(*reinterpret_cast<const __half*>(&raw));
    }
}

__device__ __forceinline__ int8_t quant_one(float v, float inv)
{
    int q = static_cast<int>(rintf(v * inv));
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    return static_cast<int8_t>(q);
}

// Vectorized kernel: each thread owns VEC consecutive elements per iteration;
// per-thread chunks stay in registers between the absmax and quantize passes.
// Requires cols % VEC == 0 (guaranteed by the host dispatch).
template <bool IS_BF16, int VPT>
__global__ void __launch_bounds__(BLOCK_SIZE) ptq_i8_vec_kernel(int8_t* __restrict__ out,
                                                                const uint16_t* __restrict__ in,
                                                                float* __restrict__ scale,
                                                                const int cols)
{
    const int64_t row    = blockIdx.x;
    const int tid        = threadIdx.x;
    const int nvec       = cols / VEC;
    const uint4* row_in  = reinterpret_cast<const uint4*>(in + row * cols);
    uint2* row_out       = reinterpret_cast<uint2*>(out + row * cols);

    uint4 raw[VPT];
    float amax = 0.f;
    int idx    = tid;
#pragma unroll
    for(int j = 0; j < VPT; ++j, idx += BLOCK_SIZE)
    {
        if(idx < nvec)
        {
            raw[j]            = row_in[idx];
            const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw[j]);
#pragma unroll
            for(int k = 0; k < VEC; ++k)
            {
                amax = fmaxf(amax, fabsf(to_f32(h[k], IS_BF16)));
            }
        }
    }

    amax = fmaxf(block_reduce<float, hipcub::Max, BLOCK_SIZE, true>(amax, hipcub::Max()),
                 1e-10f);
    const float inv = 127.f / amax;

    idx = tid;
#pragma unroll
    for(int j = 0; j < VPT; ++j, idx += BLOCK_SIZE)
    {
        if(idx < nvec)
        {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw[j]);
            uint32_t packed[VEC / 4] = {0, 0};
#pragma unroll
            for(int k = 0; k < VEC; ++k)
            {
                packed[k / 4] |=
                    static_cast<uint32_t>(static_cast<uint8_t>(quant_one(to_f32(h[k], IS_BF16), inv)))
                    << (8 * (k % 4));
            }
            row_out[idx] = make_uint2(packed[0], packed[1]);
        }
    }

    if(tid == 0)
    {
        scale[row] = amax / 127.f;
    }
}

// Generic fallback for cols % VEC != 0 or very wide rows: two passes over
// global memory (the second read is expected to be L2-resident).
template <bool IS_BF16>
__global__ void __launch_bounds__(BLOCK_SIZE) ptq_i8_stream_kernel(int8_t* __restrict__ out,
                                                                   const uint16_t* __restrict__ in,
                                                                   float* __restrict__ scale,
                                                                   const int cols)
{
    const int64_t row = blockIdx.x;
    const int tid     = threadIdx.x;
    const uint16_t* row_in = in + row * cols;

    float amax = 0.f;
    for(int e = tid; e < cols; e += BLOCK_SIZE)
    {
        amax = fmaxf(amax, fabsf(to_f32(row_in[e], IS_BF16)));
    }

    amax = fmaxf(block_reduce<float, hipcub::Max, BLOCK_SIZE, true>(amax, hipcub::Max()),
                 1e-10f);
    const float inv = 127.f / amax;

    int8_t* row_out = out + row * cols;
    for(int e = tid; e < cols; e += BLOCK_SIZE)
    {
        row_out[e] = quant_one(to_f32(row_in[e], IS_BF16), inv);
    }

    if(tid == 0)
    {
        scale[row] = amax / 127.f;
    }
}

} // anonymous namespace

void per_token_quant_i8(torch::Tensor& out,         // [rows, cols] int8
                        const torch::Tensor& input, // [rows, cols] bf16/fp16
                        torch::Tensor& scale)       // [rows] float
{
    TORCH_CHECK(input.is_contiguous() && out.is_contiguous() && scale.is_contiguous());
    TORCH_CHECK(input.dim() >= 1);
    const int cols = static_cast<int>(input.size(-1));
    const int64_t rows = input.numel() / cols;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const dim3 grid(static_cast<unsigned int>(rows));
    const dim3 block(BLOCK_SIZE);

    auto dispatch = [&](auto dtype_tag) {
        using T = decltype(dtype_tag);
        constexpr bool BF16 = std::is_same_v<T, __hip_bfloat16>;
        const uint16_t* in_raw = reinterpret_cast<const uint16_t*>(input.data_ptr());
        if(cols % VEC == 0)
        {
            if(cols <= VEC * BLOCK_SIZE)
            {
                ptq_i8_vec_kernel<BF16, 1><<<grid, block, 0, stream>>>(
                    out.data_ptr<int8_t>(), in_raw, scale.data_ptr<float>(), cols);
            }
            else if(cols <= 2 * VEC * BLOCK_SIZE)
            {
                ptq_i8_vec_kernel<BF16, 2><<<grid, block, 0, stream>>>(
                    out.data_ptr<int8_t>(), in_raw, scale.data_ptr<float>(), cols);
            }
            else if(cols <= 4 * VEC * BLOCK_SIZE)
            {
                ptq_i8_vec_kernel<BF16, 4><<<grid, block, 0, stream>>>(
                    out.data_ptr<int8_t>(), in_raw, scale.data_ptr<float>(), cols);
            }
            else
            {
                ptq_i8_stream_kernel<BF16><<<grid, block, 0, stream>>>(
                    out.data_ptr<int8_t>(), in_raw, scale.data_ptr<float>(), cols);
            }
        }
        else
        {
            ptq_i8_stream_kernel<BF16><<<grid, block, 0, stream>>>(
                out.data_ptr<int8_t>(), in_raw, scale.data_ptr<float>(), cols);
        }
    };

    if(input.scalar_type() == at::kBFloat16)
    {
        dispatch(__hip_bfloat16{});
    }
    else if(input.scalar_type() == at::kHalf)
    {
        dispatch(__half{});
    }
    else
    {
        TORCH_CHECK(false, "per_token_quant_i8: unsupported input dtype ",
                     input.scalar_type());
    }
}

} // namespace ptq_i8
} // namespace aiter
