// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
//
// Reduce-sum along the last dimension on gfx936/gfx938.
//
// Design (v2):
//   - 1 wave = 1 row; W waves/block from tuned pick_waves_per_block
//   - global -> VGPR via buffer_load_dwordx4 (fla/arch.h + quick_all_reduce pattern)
//   - fp32 accumulate; no LDS

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>

#include "dispatch_utils.h"
#include "reduce_sum.h"

namespace aiter {
namespace {

constexpr int kWaveSize         = 64;
constexpr int kMaxWavesPerBlock = 16;
constexpr int kLoadBytes        = 16; // dwordx4

using i32x4 = int __attribute__((ext_vector_type(4)));
using u32x4 = unsigned int __attribute__((ext_vector_type(4)));

union BufferResource
{
    __device__ __forceinline__ BufferResource(const void* buffer_address, uint32_t buffer_size)
        : address(const_cast<void*>(buffer_address))
        , range(buffer_size)
        , config(0x00020000U)
    {
    }

    i32x4 descriptor;
    struct
    {
        void* address;
        uint32_t range;
        uint32_t config;
    };
};

__device__ i32x4 llvm_buffer_load_dwordx4(i32x4 srsrc,
                                          int32_t voffset,
                                          int32_t soffset,
                                          int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ __forceinline__ u32x4
buffer_load_dwordx4_vgpr(i32x4 srsrc, int32_t voffset, int32_t aux = 0)
{
    return __builtin_bit_cast(u32x4,
                              llvm_buffer_load_dwordx4(srsrc, voffset, /*soffset=*/0, aux));
}

constexpr int32_t kAuxSlc = 2;

__device__ __forceinline__ float wave_reduce_sum(float val)
{
#pragma unroll
    for(int offset = kWaveSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__device__ __forceinline__ float u16_bits_to_float_as_bf16(uint16_t bits)
{
    return __builtin_bit_cast(float, static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ float u16_bits_to_float_as_fp16(uint16_t bits)
{
    return __half2float(__builtin_bit_cast(__half, bits));
}

template <typename scalar_t>
__device__ __forceinline__ float unpack_u32x4_sum(u32x4 raw)
{
    float acc = 0.f;
    if constexpr(std::is_same_v<scalar_t, float>)
    {
        const float* f = reinterpret_cast<const float*>(&raw);
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            acc += f[i];
        }
    }
    else if constexpr(std::is_same_v<scalar_t, c10::Half>)
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const uint32_t w = raw[i];
            acc += u16_bits_to_float_as_fp16(static_cast<uint16_t>(w & 0xffffu));
            acc += u16_bits_to_float_as_fp16(static_cast<uint16_t>(w >> 16));
        }
    }
    else
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const uint32_t w = raw[i];
            acc += u16_bits_to_float_as_bf16(static_cast<uint16_t>(w & 0xffffu));
            acc += u16_bits_to_float_as_bf16(static_cast<uint16_t>(w >> 16));
        }
    }
    return acc;
}

template <typename scalar_t>
__device__ __forceinline__ float scalar_load_sum(const scalar_t* ptr)
{
    return static_cast<float>(*ptr);
}

template <typename scalar_t>
__global__ void reduce_sum_kernel_buf(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ out,
                                      int64_t batch,
                                      int64_t dim,
                                      int waves_per_block)
{
    const int lane    = threadIdx.x & (kWaveSize - 1);
    const int wave_id = threadIdx.x >> 6;
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * waves_per_block + wave_id;
    if(row >= batch)
    {
        return;
    }

    const scalar_t* row_ptr = input + row * dim;
    const uint32_t row_bytes =
        static_cast<uint32_t>(dim * static_cast<int64_t>(sizeof(scalar_t)));
    BufferResource rsrc(row_ptr, row_bytes);

    constexpr int kElemsPerLoad = kLoadBytes / static_cast<int>(sizeof(scalar_t));
    const int64_t n_vec         = dim / kElemsPerLoad;

    float acc = 0.f;
    for(int64_t idx = lane; idx < n_vec; idx += kWaveSize)
    {
        u32x4 raw = buffer_load_dwordx4_vgpr(
            rsrc.descriptor, static_cast<int32_t>(idx * kLoadBytes), kAuxSlc);
        acc += unpack_u32x4_sum<scalar_t>(raw);
    }

    for(int64_t i = n_vec * kElemsPerLoad + lane; i < dim; i += kWaveSize)
    {
        acc += scalar_load_sum(row_ptr + i);
    }

    acc = wave_reduce_sum(acc);
    if(lane == 0)
    {
        out[row] = static_cast<scalar_t>(acc);
    }
}

template <typename scalar_t>
__global__ void reduce_sum_kernel_scalar(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ out,
                                         int64_t batch,
                                         int64_t dim,
                                         int waves_per_block)
{
    const int lane    = threadIdx.x & (kWaveSize - 1);
    const int wave_id = threadIdx.x >> 6;
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * waves_per_block + wave_id;
    if(row >= batch)
    {
        return;
    }

    const scalar_t* row_ptr = input + row * dim;
    float acc               = 0.f;
    for(int64_t i = lane; i < dim; i += kWaveSize)
    {
        acc += scalar_load_sum(row_ptr + i);
    }
    acc = wave_reduce_sum(acc);
    if(lane == 0)
    {
        out[row] = static_cast<scalar_t>(acc);
    }
}

inline int pick_waves_per_block(int64_t batch, int64_t dim, int /*num_cu*/, int override_w)
{
    if(override_w > 0)
    {
        TORCH_CHECK(override_w <= kMaxWavesPerBlock,
                    "waves_per_block must be <= ",
                    kMaxWavesPerBlock);
        return override_w;
    }

    int w;
    if(dim <= 1024)
    {
        w = 4;
    }
    else if(dim <= 2048)
    {
        w = (batch >= 131072) ? 8 : 2;
    }
    else
    {
        w = 1;
    }

    if(static_cast<int64_t>(w) > batch)
    {
        w = static_cast<int>(batch);
    }
    return std::max(w, 1);
}

} // namespace

void reduce_sum(torch::Tensor& out, torch::Tensor& input, int64_t waves_per_block)
{
    TORCH_CHECK(input.is_cuda(), "reduce_sum: input must be CUDA/HIP tensor");
    TORCH_CHECK(out.is_cuda(), "reduce_sum: out must be CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == out.scalar_type(),
                "reduce_sum: input/out dtype mismatch");
    TORCH_CHECK(input.dim() >= 1, "reduce_sum: input must be at least 1-D");
    TORCH_CHECK(input.stride(-1) == 1, "reduce_sum: last dim must be contiguous");

    const int64_t dim   = input.size(-1);
    const int64_t batch = input.numel() / dim;

    TORCH_CHECK(out.numel() == batch, "reduce_sum: out numel must equal batch=", batch);
    TORCH_CHECK(dim > 0, "reduce_sum: dim must be > 0");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto* props      = at::cuda::getCurrentDeviceProperties();
    const int num_cu = props ? props->multiProcessorCount : 72;
    const int W =
        pick_waves_per_block(batch, dim, num_cu, static_cast<int>(waves_per_block));

    const dim3 block(static_cast<unsigned>(W * kWaveSize));
    const dim3 grid(static_cast<unsigned>((batch + W - 1) / W));

    const bool use_buf =
        ((dim * static_cast<int64_t>(input.element_size())) % kLoadBytes == 0);

    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reduce_sum_kernel", [&] {
        if(use_buf)
        {
            reduce_sum_kernel_buf<scalar_t><<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                batch,
                dim,
                W);
        }
        else
        {
            reduce_sum_kernel_scalar<scalar_t><<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                batch,
                dim,
                W);
        }
    });
}

} // namespace aiter
