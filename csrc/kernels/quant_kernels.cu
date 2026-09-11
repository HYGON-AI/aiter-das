/*
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Modified by Hygon Information Technology Co., Ltd.: adapt vector loads and
// quantization dispatch, handle zero scales, and add fused SwiGLU quantization.
// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
 

#include "aiter_hip_common.h"
#include "dispatch_utils.h"
#include "hip_reduce.h"
#include "quant_common.cuh"
#include "rocprim/rocprim.hpp"
#include "vec_convert.h"
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hipcub/hipcub.hpp>

const int32_t BlockSize           = 256;
const int32_t groupQuantBlockSize = 64;

namespace aiter {
template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 32>
__global__ void
dynamic_per_group_scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                      float* __restrict__ scale,
                                      DTYPE_I const* __restrict__ input,
                                      float const* __restrict__ scale_ub,
                                      const int32_t group_size,
                                      int64_t ori_rows,
                                      int32_t ori_cols,
                                      int32_t ori_row_stride,
                                      bool shuffle_scale                   = true,
                                      int32_t const* __restrict__ num_rows = nullptr,
                                      const int32_t num_cols_factor        = 1)
{
    auto fp4_scale_shuffle_id = [](int32_t scaleN_pad, int32_t x, int32_t y) {
        return (x / 32 * scaleN_pad) * 32 + (y / 8) * 256 + (y % 4) * 64 + (x % 16) * 4 +
               (y % 8) / 4 * 2 + (x % 32) / 16;
    };
    if(num_rows != nullptr)
    {
        ori_rows = *num_rows * num_cols_factor;
    }
    int num_thread_per_group = group_size / thread_data_size;
    int64_t row_offset       = blockIdx.x * groupQuantBlockSize;
    int64_t groupId          = (row_offset + threadIdx.x) / num_thread_per_group;
    int32_t scaleN           = ori_cols / group_size;
    int32_t scaleN_pad       = (std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> && shuffle_scale)
                                   ? (((scaleN + 7) / 8) * 8)
                                   : scaleN;
    int64_t x                = groupId / scaleN_pad;
    int32_t y                = groupId % scaleN_pad;
    if constexpr(std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
    {
        if(x >= ori_rows || y >= scaleN)
        {
            // if (shuffle_scale && threadIdx.x % num_thread_per_group == 0)
            // {
            //   auto *tmp = reinterpret_cast<uint8_t *>(scale);
            //   groupId = fp4_scale_shuffle_id(scaleN_pad, x, y);
            //   tmp[groupId] = 0x7f;
            // }
            return;
        }
    }
    else
    {
        if(x >= ori_rows)
            return;
    }

    row_offset  = x * ori_row_stride + y * group_size;
    using vec_i = ck_tile::vec_t<DTYPE_I, thread_data_size>;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? thread_data_size / 2 : thread_data_size;
    using vec_o = ck_tile::vec_t<DTYPE_O, vec_size_o>;
    const float inverted_DTYPE_MAX =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
            ? 0.25
            : (1. / ck_tile::type_convert<float>(ck_tile::numeric<DTYPE_O>::max()));

    // static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    static constexpr int32_t ooba_o = 4 / sizeof(DTYPE_O);
    // const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    const int64_t oob_o = (ori_rows * ori_cols + ooba_o - 1) / ooba_o * ooba_o;
    // auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(input +
    // row_offset, oob_i); buffer_i.init_raw();

    auto const* input_vecs = reinterpret_cast<vec_i const*>(input + row_offset);
    // vec_i thread_data      = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
    vec_i thread_data = input_vecs[threadIdx.x % num_thread_per_group];
    float absMax      = 1e-10f;
    for(size_t j = 0; j < thread_data_size; j++)
    {
        absMax = max(absMax, abs(ck_tile::type_convert<float>(thread_data[j])));
    }
    absMax = multithread_reduce(absMax, hipcub::Max(), num_thread_per_group);

    auto fp4_scale = [](float tmp) {
        uint32_t u32      = ck_tile::bit_cast<uint32_t>(tmp);
        uint32_t exponent = (u32 >> 23) & 0b11111111;
        if(exponent == 0b11111111)
        {
            return ck_tile::bit_cast<float>(exponent << 23);
        }
        if(((u32 & 0x400000)) && (((u32 & 0x200000)) || ((u32 & 0x1FFFFF)) || (exponent)))
            exponent += 1;
        return ck_tile::bit_cast<float>(exponent << 23);
    };
    float inverted_scale = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                               ? fp4_scale(absMax) * inverted_DTYPE_MAX
                               : absMax * inverted_DTYPE_MAX;
    row_offset           = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                               ? groupId * group_size / 2 + (threadIdx.x % num_thread_per_group) * vec_size_o
                               : groupId * group_size + (threadIdx.x % num_thread_per_group) * vec_size_o;
    if(threadIdx.x % num_thread_per_group == 0)
    {
        if constexpr(std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
        {
            auto* tmp        = reinterpret_cast<uint8_t*>(scale);
            uint8_t exponent = (ck_tile::bit_cast<uint32_t>(inverted_scale) >> 23) & 0b11111111;
            if(shuffle_scale)
            {
                groupId = fp4_scale_shuffle_id(scaleN_pad, x, y);
            }
            tmp[groupId] = exponent;
        }
        else
        {
            if(shuffle_scale)
            {
                groupId = y * ori_rows + x;
            }
            scale[groupId] = inverted_scale;
        }
    }
    inverted_scale =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? inverted_scale : 1.0f / inverted_scale;

    using DTYPE_STORE = typename ck_tile::vector_traits<DTYPE_O>::scalar_type;
    auto* out_ptr     = reinterpret_cast<DTYPE_STORE*>(out);
    auto buffer_o =
        ck_tile::make_buffer_view<ck_tile::address_space_enum::global,
                                  ck_tile::amd_buffer_coherence_enum::glc>(out_ptr, oob_o);
    buffer_o.init_raw();

    auto out_s =
        ck_tile::vec_convert<DTYPE_O, DTYPE_I, thread_data_size>(thread_data, inverted_scale)
            .template get_as<DTYPE_STORE>();
    if constexpr(thread_data_size <= 16)
    {
        buffer_o.set(row_offset, 0, true, out_s);
    }
    else
    {
        static constexpr int32_t o_step = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? 8 : 16;
        assert(thread_data_size % 16 == 0);
        using vecT                        = ck_tile::vec_t<DTYPE_STORE, o_step>;
        auto vec                          = out_s.template get_as<vecT>();
        static constexpr int32_t num_iter = thread_data_size / 16;

        for(size_t j = 0; j < num_iter; j++)
        {
            buffer_o.set(row_offset + j * o_step, 0, true, vec[j]);
        }
    }
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 16>
__device__ std::tuple<float, DTYPE_I*> data_to_per_row_scale(const DTYPE_I* __restrict__ input,
                                                             const int32_t cols)
{
    static constexpr int32_t vec_size_i =
        thread_data_size == 0 ? 16 / sizeof(DTYPE_O) : thread_data_size;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? vec_size_i / 2 : vec_size_i;
    using vec_i = ck_tile::vec_t<DTYPE_I, vec_size_i>;
    const float inverted_DTYPE_MAX =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
            ? 0.25
            : (1. / ck_tile::type_convert<float>(ck_tile::numeric<DTYPE_O>::max()));

    const int64_t row_offset        = blockIdx.x * cols;
    auto const* ptr_i               = reinterpret_cast<DTYPE_I const*>(input + row_offset);
    auto const* input_vecs          = reinterpret_cast<vec_i const*>(ptr_i);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_i, oob_i);
    buffer_i.init_raw();

    // double load core loop start
    const int32_t num_elems_tail = cols % vec_size_i;
    const int32_t num_vecs       = (cols + vec_size_i - 1) / vec_size_i;

    vec_i vec_cur;
    int32_t vec_idx    = threadIdx.x;
    int32_t vec_stride = BlockSize;
    static constexpr int32_t max_vec_size_i = 16 / sizeof(DTYPE_I);
    static constexpr int32_t vec_i_iter = vec_size_i > max_vec_size_i ? vec_size_i / max_vec_size_i : 1; 
    auto load_vec = [&](int32_t idx) {
        vec_i vec;
        #pragma unroll
        for(int i = 0; i < vec_i_iter; i++)
        {
            if constexpr(vec_size_i > max_vec_size_i)
            {
                using max_vec_i = ck_tile::vec_t<DTYPE_I, max_vec_size_i>;
                max_vec_i vec_tmp =
                    buffer_i.template get<max_vec_i>(idx * vec_size_i, i * max_vec_size_i, true);

                #pragma unroll
                for(int j = 0; j < max_vec_size_i; j++)
                {
                    vec[i * max_vec_size_i + j] = vec_tmp[j];
                }
            }
            else
            {
                vec = buffer_i.template get<vec_i>(idx * vec_size_i, 0, true);
            }
        }
        return vec;
    };
    if(vec_idx < num_vecs)
    {
        vec_cur = load_vec(vec_idx);
    }

    float absMax = 0.f;
    if constexpr(thread_data_size == 0)
    {
        vec_i vec_nxt;
        for(vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
        {
            vec_nxt = load_vec(vec_idx);
            for(size_t j = 0; j < vec_size_i; j++)
            {
                absMax = max(absMax, abs(ck_tile::type_convert<float>(vec_cur[j])));
            }
            vec_cur = vec_nxt;
        }
        vec_idx -= vec_stride;
    }
    if(vec_idx < num_vecs)
    {
#pragma unroll
        for(size_t j = 0; j < vec_size_i; j++)
        {
            absMax = max(absMax, abs(ck_tile::type_convert<float>(vec_cur[j])));
        }
    }
    // double load core loop end

    // using BlockReduce = hipcub::BlockReduce<float, BlockSize>;
    // __shared__ typename BlockReduce::TempStorage temp_storage;
    // absMax = BlockReduce(temp_storage).Reduce(absMax, hipcub::Max());
    absMax = block_reduce<float, hipcub::Max, BlockSize, true>(absMax, hipcub::Max());

    auto fp4_scale = [](float tmp) {
        uint32_t u32      = ck_tile::bit_cast<uint32_t>(tmp);
        uint32_t exponent = (u32 >> 23) & 0b11111111;
        if(exponent == 0b11111111)
        {
            return ck_tile::bit_cast<float>(exponent << 23);
        }
        if(((u32 & 0x400000)) && (((u32 & 0x200000)) || ((u32 & 0x1FFFFF)) || (exponent)))
            exponent += 1;
        return ck_tile::bit_cast<float>(exponent << 23);
    };
    float row_scale = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                          ? fp4_scale(absMax) * inverted_DTYPE_MAX
                          : absMax * inverted_DTYPE_MAX;
    if constexpr(!std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
    {
        row_scale = row_scale == 0.0f ? 1.0f : row_scale;
    }
    return std::make_tuple(row_scale, reinterpret_cast<DTYPE_I*>(&vec_cur));
}

template <typename DTYPE_I, typename DTYPE_O>
__global__ void
data_to_scale_kernel(float* __restrict__ scale, const DTYPE_I* __restrict__ input, const int cols)
{
    auto res        = data_to_per_row_scale<DTYPE_I, DTYPE_O, 0>(input, cols);
    float row_scale = std::get<0>(res);
    if(threadIdx.x == 0)
    {
        vllm::atomicMaxFloat(scale, row_scale);
    }
}

template <typename DTYPE_I, typename DTYPE_O>
__device__ void scaled_quant_impl(DTYPE_O* __restrict__ out,
                                  const DTYPE_I* __restrict__ input,
                                  const float* __restrict__ scale,
                                  const int32_t cols)
{

    const float inverted_scale =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? (*scale) : 1.0f / (*scale);
    static constexpr int32_t vec_size_i = 16 / sizeof(DTYPE_O);
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? vec_size_i / 2 : vec_size_i;

    using vec_i       = ck_tile::vec_t<DTYPE_I, vec_size_i>;
    using vec_o       = ck_tile::vec_t<DTYPE_O, vec_size_o>;
    using DTYPE_STORE = typename ck_tile::vector_traits<DTYPE_O>::scalar_type;

    const int64_t row_offset        = blockIdx.x * cols;
    auto const* ptr_i               = reinterpret_cast<DTYPE_I const*>(input + row_offset);
    auto const* input_vecs          = reinterpret_cast<vec_i const*>(ptr_i);
    auto* ptr_o                     = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                                          ? reinterpret_cast<DTYPE_STORE*>(out + row_offset / 2)
                                          : reinterpret_cast<DTYPE_STORE*>(out + row_offset);
    auto* out_vecs                  = reinterpret_cast<vec_o*>(ptr_o);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    static constexpr int32_t ooba_o = 4 / sizeof(DTYPE_O);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    const int32_t oob_o             = (cols + ooba_o - 1) / ooba_o * ooba_o;

    auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_i, oob_i);
    buffer_i.init_raw();
    auto buffer_o = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_o, oob_o);
    buffer_o.init_raw();

    // double load core loop start
    const int32_t num_elems_tail = cols % vec_size_i;
    const int32_t num_vecs       = (cols + vec_size_i - 1) / vec_size_i;
    const int32_t tail_thread    = num_vecs % BlockSize;
    vec_i vec_nxt;
    vec_i vec_cur;
    // size_t vec_idx = threadIdx.x * vec_size_i;
    // size_t vec_stride = BlockSize * vec_size_i;
    int32_t vec_idx    = threadIdx.x;
    int32_t vec_stride = BlockSize;
    static constexpr int32_t max_vec_size_i = 16 / sizeof(DTYPE_I);
    static constexpr int32_t vec_i_iter = vec_size_i > max_vec_size_i ? vec_size_i / max_vec_size_i : 1;
    auto load_vec = [&](int32_t idx) {
        vec_i vec;
        #pragma unroll
        for(int i = 0; i < vec_i_iter; i++)
        {
            if constexpr(vec_size_i > max_vec_size_i)
            {
                using max_vec_i = ck_tile::vec_t<DTYPE_I, max_vec_size_i>;
                max_vec_i vec_tmp =
                    buffer_i.template get<max_vec_i>(idx * vec_size_i, i * max_vec_size_i, true);

                #pragma unroll
                for(int j = 0; j < max_vec_size_i; j++)
                {
                    vec[i * max_vec_size_i + j] = vec_tmp[j];
                }
            }
            else
            {
                vec = buffer_i.template get<vec_i>(idx * vec_size_i, 0, true);
            }
        }
        return vec;
    };
    if(vec_idx < num_vecs)
    {
        vec_cur = load_vec(vec_idx);
    }

    for(vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
    {
        vec_nxt = load_vec(vec_idx);
        buffer_o.set(
            (vec_idx - vec_stride) * vec_size_o,
            0,
            true,
            ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size_i>(vec_cur, inverted_scale)
                .template get_as<DTYPE_STORE>());
        vec_cur = vec_nxt;
    }

    if(vec_idx - vec_stride < num_vecs)
    {
        buffer_o.set(
            (vec_idx - vec_stride) * vec_size_o,
            0,
            true,
            ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size_i>(vec_cur, inverted_scale)
                .template get_as<DTYPE_STORE>());
    }
    // double load core loop end
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 16>
__device__ void scaled_quant_vgpr_impl(DTYPE_O* __restrict__ out,
                                       DTYPE_I* __restrict__ input,
                                       const float* __restrict__ scale,
                                       const int cols)
{

    const float inverted_scale =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? (*scale) : 1.0f / (*scale);
    static constexpr int32_t vec_size_i = thread_data_size;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? vec_size_i / 2 : vec_size_i;

    using vec_i       = ck_tile::vec_t<DTYPE_I, vec_size_i>;
    using vec_o       = ck_tile::vec_t<DTYPE_O, vec_size_o>;
    using DTYPE_STORE = typename ck_tile::vector_traits<DTYPE_O>::scalar_type;

    const int64_t row_offset        = blockIdx.x * cols;
    auto const* ptr_i               = reinterpret_cast<DTYPE_I const*>(input);
    auto const* input_vecs          = reinterpret_cast<vec_i const*>(ptr_i);
    auto* out_ptr                   = reinterpret_cast<DTYPE_O*>(out);
    auto* ptr_o                     = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                                          ? reinterpret_cast<DTYPE_STORE*>(out + row_offset / 2)
                                          : reinterpret_cast<DTYPE_STORE*>(out + row_offset);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    static constexpr int32_t ooba_o = 4 / sizeof(DTYPE_O);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    const int32_t oob_o             = (cols + ooba_o - 1) / ooba_o * ooba_o;

    auto buffer_o = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_o, oob_o);
    buffer_o.init_raw();

    const int32_t num_vecs = (cols + vec_size_i - 1) / vec_size_i;

    if(threadIdx.x < num_vecs)
    {
        auto out = ck_tile::vec_convert<DTYPE_O, DTYPE_I, vec_size_i>(*input_vecs, inverted_scale)
                       .template get_as<DTYPE_STORE>();
        if constexpr(vec_size_i <= 16)
        {

            buffer_o.set(threadIdx.x * vec_size_o, 0, true, out);
        }
        else
        {
            static constexpr int32_t o_step = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? 8 : 16;
            assert(vec_size_i % 16 == 0);
            using vecT                        = ck_tile::vec_t<DTYPE_STORE, o_step>;
            auto vec                          = out.template get_as<vecT>();
            static constexpr int32_t num_iter = vec_size_i / 16;

            for(size_t j = 0; j < num_iter; j++)
            {
                buffer_o.set(threadIdx.x * vec_size_o + j * o_step, 0, true, vec[j]);
            }
        }
    }
}

template <typename DTYPE_I, typename DTYPE_O>
__global__ void scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                    const DTYPE_I* __restrict__ input,
                                    const float* __restrict__ scale,
                                    const int cols)
{
    scaled_quant_impl<DTYPE_I>(out, input, scale, cols);
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 16>
__global__ void
dynamic_per_token_scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                      float* __restrict__ scale,
                                      DTYPE_I* __restrict__ input,
                                      float const* __restrict__ scale_ub,
                                      const int32_t cols,
                                      int32_t const* __restrict__ num_rows = nullptr,
                                      const int32_t num_rows_factor        = 1)
{
    const int token_idx = blockIdx.x;
    if(num_rows != nullptr)
    {
        int32_t rows = *num_rows * num_rows_factor;
        if(token_idx >= rows)
            return;
    }
    auto res         = data_to_per_row_scale<DTYPE_I, DTYPE_O, thread_data_size>(input, cols);
    float row_scale  = std::get<0>(res);
    DTYPE_I* vec_ptr = std::get<1>(res);

    if(threadIdx.x == 0)
    {
        if constexpr(std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
        {
            auto* tmp        = reinterpret_cast<uint8_t*>(scale);
            uint8_t exponent = (ck_tile::bit_cast<uint32_t>(row_scale) >> 23) & 0b11111111;
            tmp[token_idx]   = exponent;
        }
        else
        {
            scale[token_idx] = row_scale;
        }
    }

    if constexpr(thread_data_size == 0)
    {
        scaled_quant_impl<DTYPE_I>(out, input, &row_scale, cols);
    }
    else
    {
        scaled_quant_vgpr_impl<DTYPE_I, DTYPE_O, thread_data_size>(out, vec_ptr, &row_scale, cols);
    }
}

template <typename DTYPE_I, typename DTYPE_O, int block_size, int thread_data_size = 16>
__device__ std::tuple<float, float*>
smooth_data_to_per_row_scale(const DTYPE_I* __restrict__ input,
                             const float* __restrict__ smooth_scale,
                             const int32_t* __restrict__ smooth_scale_map,
                             const int32_t cols,
                             const int32_t token_idx)
{
    static constexpr int32_t vec_size_i =
        thread_data_size == 0 ? 16 / sizeof(DTYPE_O) : thread_data_size;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t> ? vec_size_i / 2 : vec_size_i;
    using vec_i = ck_tile::vec_t<DTYPE_I, vec_size_i>;
    using vec_s = ck_tile::vec_t<float, vec_size_i>;
    const float inverted_DTYPE_MAX =
        std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
            ? 0.25
            : (1. / ck_tile::type_convert<float>(ck_tile::numeric<DTYPE_O>::max()));

    const int32_t smscale_map_idx = smooth_scale_map == nullptr ? 0 : smooth_scale_map[blockIdx.x];
    const int64_t row_offset     = token_idx * cols;
    auto const* ptr_i            = reinterpret_cast<DTYPE_I const*>(input + row_offset);
    auto const* input_vecs       = reinterpret_cast<vec_i const*>(ptr_i);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_i, oob_i);
    buffer_i.init_raw();

    auto const* ptr_smscale  = reinterpret_cast<float const*>(smooth_scale + smscale_map_idx * cols);
    auto const* smscale_vecs = reinterpret_cast<vec_s const*>(ptr_smscale);
    auto buffer_s =
        ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_smscale, cols);
    buffer_s.init_raw();

    const int32_t num_vecs = (cols + vec_size_i - 1) / vec_size_i;
    vec_i vec_cur;
    vec_s smscale_cur;
    int32_t vec_idx = threadIdx.x;
    float absMax   = 0.f;
    if(vec_idx < num_vecs)
    {
        vec_cur     = buffer_i.template get<vec_i>(vec_idx * vec_size_i, 0, true);
        smscale_cur = buffer_s.template get<vec_s>(vec_idx * vec_size_i, 0, true);
#pragma unroll
        for(size_t j = 0; j < vec_size_i; j++)
        {
            smscale_cur[j] = ck_tile::type_convert<float>(vec_cur[j]) * smscale_cur[j];
            absMax         = max(absMax, abs(smscale_cur[j]));
        }
    }

    absMax = block_reduce<float, hipcub::Max, block_size, true>(absMax, hipcub::Max());

    auto fp4_scale = [](float tmp) {
        uint32_t u32      = ck_tile::bit_cast<uint32_t>(tmp);
        uint32_t exponent = (u32 >> 23) & 0b11111111;
        if(exponent == 0b11111111)
        {
            return ck_tile::bit_cast<float>(exponent << 23);
        }
        if(((u32 & 0x400000)) && (((u32 & 0x200000)) || ((u32 & 0x1FFFFF)) || (exponent)))
            exponent += 1;
        return ck_tile::bit_cast<float>(exponent << 23);
    };
    float row_scale = std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>
                          ? fp4_scale(absMax) * inverted_DTYPE_MAX
                          : absMax * inverted_DTYPE_MAX;
    return std::make_tuple(row_scale, reinterpret_cast<float*>(&smscale_cur));
}

template <typename DTYPE_I, typename DTYPE_O, int block_size, int thread_data_size = 16>
__global__ void smooth_per_token_scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                                     float* __restrict__ scale,
                                                     DTYPE_I* __restrict__ input,
                                                     float* __restrict__ smooth_scale,
                                                     int* __restrict__ smooth_scale_map,
                                                     const int32_t cols,
                                                     int32_t const* __restrict__ num_rows = nullptr,
                                                     const int32_t num_rows_factor        = 1,
                                                     const int32_t input_dim0             = 1,
                                                     const int32_t input_dim1             = 1,
                                                     const int32_t input_stride0          = 1,
                                                     const int32_t input_stride1          = 1)
{
    int token_idx = blockIdx.x;
    if(num_rows != nullptr)
    {
        int32_t rows = *num_rows * num_rows_factor;
        if(token_idx >= rows)
            return;
    }
    int real_token_idx = token_idx % input_dim1 * (input_stride1 / cols) +
                         (token_idx / input_dim1) % input_dim0 * (input_stride0 / cols);
    auto res = smooth_data_to_per_row_scale<DTYPE_I, DTYPE_O, block_size, thread_data_size>(
        input, smooth_scale, smooth_scale_map, cols, real_token_idx);
    float row_scale = std::get<0>(res);
    float* vec_ptr  = std::get<1>(res);

    if(threadIdx.x == 0)
    {
        if constexpr(std::is_same_v<DTYPE_O, ck_tile::fp4x2_t>)
        {
            auto* tmp        = reinterpret_cast<uint8_t*>(scale);
            uint8_t exponent = (ck_tile::bit_cast<uint32_t>(row_scale) >> 23) & 0b11111111;
            tmp[token_idx]   = exponent;
        }
        else
        {
            scale[token_idx] = row_scale;
        }
    }

    scaled_quant_vgpr_impl<float, DTYPE_O, thread_data_size>(out, vec_ptr, &row_scale, cols);
}

void static_per_tensor_quant(torch::Tensor& out,         // [..., d]
                             torch::Tensor const& input, // [..., d]
                             torch::Tensor const& scale) // [1]
{
    const int cols = input.size(-1);
    int rows       = input.numel() / cols;
    dim3 grid(rows);
    dim3 block(BlockSize);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    if(out.dtype() == torch::kInt8)
    {
        AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "scaled_quant_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<ck_tile::int8_t*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                scale.data_ptr<float>(),
                cols);
        });
    }
#ifdef GPU_ENABLE_FP8
    else if(out.dtype() == torch_fp8)
    {
        AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "scaled_quant_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<FP8_TYPE*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                scale.data_ptr<float>(),
                cols);
        });
    }
#endif 
    else
    {
        TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
    }
}

#define DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA)      \
    AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "quant_kernel", [&] {              \
        using input_dtype = typename t2ck<scalar_t>::type;                                  \
        aiter::quant_kernel<input_dtype, DTYPE_O, THREAD_DATA><<<grid, block, 0, stream>>>( \
            reinterpret_cast<DTYPE_O*>(out.data_ptr()),                                     \
            scales.data_ptr<float>(),                                                       \
            reinterpret_cast<input_dtype*>(input.data_ptr()),                               \
            scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,                   \
            cols,                                                                           \
            num_rows_ptr,                                                                   \
            num_rows_factor);                                                               \
    });

#define DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(quant_kernel, DTYPE_O, cols) \
    if(cols <= 8 * BlockSize)                                                       \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 8)        \
    }                                                                               \
    else if(cols <= 16 * BlockSize)                                                 \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 16)       \
    }                                                                               \
    else if(cols <= 32 * BlockSize)                                                 \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 32)       \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 0)        \
    }

void dynamic_per_tensor_quant(torch::Tensor& out,         // [..., d]
                              torch::Tensor const& input, // [..., d]
                              torch::Tensor& scale)       // [1]
{
    const int cols = input.size(-1);
    int rows       = input.numel() / cols;
    dim3 grid(rows);
    dim3 block(BlockSize);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    if(out.dtype() == torch::kInt8)
    {
        AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "scaled_quant_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            vllm::initializeScale<<<dim3(1), dim3(64), 0, stream>>>(
                scale.data_ptr<float>(), 1, 0.0f);
            aiter::data_to_scale_kernel<input_dtype, ck_tile::int8_t><<<grid, block, 0, stream>>>(
                scale.data_ptr<float>(), reinterpret_cast<input_dtype*>(input.data_ptr()), cols);
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<ck_tile::int8_t*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                scale.data_ptr<float>(),
                cols);
        });
    }
#ifdef GPU_ENABLE_FP8
    else if(out.dtype() == torch_fp8)
    {
        AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "scaled_quant_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            vllm::initializeScale<<<dim3(1), dim3(64), 0, stream>>>(
                scale.data_ptr<float>(), 1, 0.0f);
            aiter::data_to_scale_kernel<input_dtype, FP8_TYPE><<<grid, block, 0, stream>>>(
                scale.data_ptr<float>(), reinterpret_cast<input_dtype*>(input.data_ptr()), cols);
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<FP8_TYPE*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                scale.data_ptr<float>(),
                cols);
        });
    }
#endif
    else
    {
        TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
    }
}

void dynamic_per_token_scaled_quant(torch::Tensor& out,         // [..., d]
                                    torch::Tensor const& input, // [..., d]
                                    torch::Tensor& scales,
                                    std::optional<at::Tensor> const& scale_ub,
                                    bool shuffle_scale                        = false,
                                    std::optional<at::Tensor> const& num_rows = std::nullopt,
                                    int num_rows_factor                       = 1)
{
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    int const cols        = input.size(-1);
    int const rows        = input.numel() / cols;
    int32_t* num_rows_ptr = num_rows.has_value() ? num_rows->data_ptr<int32_t>() : nullptr;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();


    if(cols == 32 || cols == 64 || cols == 128)
    {
        int group_size           = cols;
        int thread_data_size     = 32;
        int num_thread_per_group = group_size / thread_data_size;
        int num_group_per_tg     = groupQuantBlockSize / num_thread_per_group;
        if(out.dtype() == torch::kInt8)
        {
            int ori_cols  = cols;
            int scaleN    = ori_cols / cols;
            int ori_rows  = rows / scaleN;
            int num_group = rows;
            dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
            dim3 const block(groupQuantBlockSize);
            AITER_DISPATCH_FLOATING16_TYPES(
                input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&] {
                    using input_dtype = typename t2ck<scalar_t>::type;
                    aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<ck_tile::int8_t*>(out.data_ptr()),
                        scales.data_ptr<float>(),
                        reinterpret_cast<input_dtype*>(input.data_ptr()),
                        scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                        group_size,
                        ori_rows,
                        ori_cols,
                        ori_cols,
                        shuffle_scale,
                        num_rows_ptr,
                        num_rows_factor);
                });
        }
#ifdef GPU_ENABLE_FP8
        else if(out.dtype() == torch_fp8)
        {
            int ori_cols  = out.size(-1);
            int scaleN    = ori_cols / cols;
            int ori_rows  = rows / scaleN;
            int num_group = rows;
            dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
            dim3 const block(groupQuantBlockSize);
            AITER_DISPATCH_FLOATING16_TYPES(
                input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&] {
                    using input_dtype = typename t2ck<scalar_t>::type;
                    aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<FP8_TYPE*>(out.data_ptr()),
                        scales.data_ptr<float>(),
                        reinterpret_cast<input_dtype*>(input.data_ptr()),
                        scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                        group_size,
                        ori_rows,
                        ori_cols,
                        ori_cols,
                        shuffle_scale,
                        num_rows_ptr,
                        num_rows_factor);
                });
        }
#endif
#if defined(__Float4_e2m1fn_x2) && defined(TORCH_Float4_e2m1fn_x2)
        else if(out.dtype() == torch::kFloat4_e2m1fn_x2)
        {
            int ori_cols  = out.size(-1) * 2;
            int scaleN    = ori_cols / cols;
            int ori_rows  = rows / scaleN;
            int num_group = shuffle_scale ? ori_rows * ((scaleN + 7) / 8 * 8) : rows;
            // int num_group = shuffle_scale ? ((ori_rows + 255) / 256 * 256) * ((scaleN + 7) / 8 *
            // 8) : rows;
            dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
            dim3 const block(groupQuantBlockSize);
            AITER_DISPATCH_FLOATING16_TYPES(
                input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&] {
                    using input_dtype = typename t2ck<scalar_t>::type;
                    aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                        reinterpret_cast<ck_tile::fp4x2_t*>(out.data_ptr()),
                        reinterpret_cast<float*>(scales.data_ptr()),
                        reinterpret_cast<input_dtype*>(input.data_ptr()),
                        scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                        group_size,
                        ori_rows,
                        ori_cols,
                        ori_cols,
                        shuffle_scale,
                        num_rows_ptr,
                        num_rows_factor);
                });
        }
#endif
        else
        {
            TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
        }
    }
    else
    {
        dim3 const grid(rows);
        dim3 const block(BlockSize);
        if(out.dtype() == torch::kInt8)
        {
            DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
                dynamic_per_token_scaled_quant_kernel, ck_tile::int8_t, cols);
        }
#ifdef GPU_ENABLE_FP8
        else if(out.dtype() == torch_fp8)
        {
            DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
                dynamic_per_token_scaled_quant_kernel, FP8_TYPE, cols);
        }
#endif
#if defined(__Float4_e2m1fn_x2) && defined(TORCH_Float4_e2m1fn_x2)
        else if(out.dtype() == torch::kFloat4_e2m1fn_x2)
        {
            DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
                dynamic_per_token_scaled_quant_kernel, ck_tile::fp4x2_t, cols);
        }
#endif
        else
        {
            TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
        }
    }
}

void dynamic_per_group_scaled_quant_fp4(torch::Tensor& out,         // [..., d]
                                        torch::Tensor const& input, // [..., d]
                                        torch::Tensor& scales,
                                        int group_size                            = 32,
                                        bool shuffle_scale                        = true,
                                        std::optional<at::Tensor> const& num_rows = std::nullopt,
                                        int num_rows_factor                       = 1)
{
    TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
                __func__,
                " only support group_size [32, 64 , 128]");
    TORCH_CHECK(out.is_contiguous());

    int const cols        = input.size(-1);
    int const rows        = input.numel() / cols;
    int const row_stride  = input.stride(-2);
    int32_t* num_rows_ptr = num_rows.has_value() ? num_rows->data_ptr<int32_t>() : nullptr;

    TORCH_CHECK(cols % group_size == 0, __func__, " cols is not divisible by group_size");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    int thread_data_size     = 32;
    int num_thread_per_group = group_size / thread_data_size;
    int num_group_per_tg     = groupQuantBlockSize / num_thread_per_group;

    int scaleN    = cols / group_size;
    int num_group = shuffle_scale ? rows * ((scaleN + 7) / 8 * 8) : rows * scaleN;
    // int num_group = shuffle_scale ? ((rows + 255) / 256 * 256) * ((scaleN + 7) / 8 * 8) : rows *
    // scaleN;
    dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
    dim3 const block(groupQuantBlockSize);

#if defined(__Float4_e2m1fn_x2)
    AITER_DISPATCH_FLOATING16_TYPES(
        input.scalar_type(), "dynamic_per_group_scaled_quant_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            aiter::dynamic_per_group_scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<ck_tile::fp4x2_t*>(out.data_ptr()),
                reinterpret_cast<float*>(scales.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                nullptr,
                group_size,
                rows,
                cols,
                row_stride,
                shuffle_scale,
                num_rows_ptr,
                num_rows_factor);
        });
#else
    TORCH_CHECK(false, __func__, " device not support Float4_e2m1fn_x2 dtype");
#endif
}

#define SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE) \
    AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "quant_kernel", [&] {                    \
        using input_dtype = typename t2ck<scalar_t>::type;                                        \
        aiter::quant_kernel<input_dtype, DTYPE_O, BLOCK_SIZE, THREAD_DATA>                        \
            <<<grid, dim3(BLOCK_SIZE), 0, stream>>>(                                              \
                reinterpret_cast<DTYPE_O*>(out.data_ptr()),                                       \
                scales.data_ptr<float>(),                                                         \
                reinterpret_cast<input_dtype*>(input.data_ptr()),                                 \
                smooth_scale.data_ptr<float>(),                                                   \
                smooth_scale_map_ptr,                                                             \
                cols,                                                                             \
                num_rows_ptr,                                                                     \
                num_rows_factor,                                                                  \
                input_dim0,                                                                       \
                input_dim1,                                                                       \
                input_stride0,                                                                    \
                input_stride1);                                                                   \
    });

#define SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(quant_kernel, DTYPE_O, cols)           \
    if(cols <= 8 * BlockSize)                                                                \
    {                                                                                        \
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 8, BlockSize)       \
    }                                                                                        \
    else if(cols <= 16 * BlockSize)                                                          \
    {                                                                                        \
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 16, BlockSize)      \
    }                                                                                        \
    else if(cols <= 16 * BlockSize * 2)                                                      \
    {                                                                                        \
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 16, BlockSize * 2)  \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
        TORCH_CHECK(false, "input last dim has exceeded the maximum value ", 32 * BlockSize) \
    }

void smooth_per_token_scaled_quant(
    torch::Tensor& out,         // [..., d]
    torch::Tensor const& input, // [..., d]
    torch::Tensor& scales,
    torch::Tensor const& smooth_scale,
    std::optional<torch::Tensor> const& smooth_scale_map = std::nullopt,
    bool shuffle_scale                                   = false,
    std::optional<torch::Tensor> const& num_rows         = std::nullopt,
    int num_rows_factor                                  = 1)
{
    TORCH_CHECK(out.is_contiguous());

    int const cols        = input.size(-1);
    int const rows        = input.numel() / cols;
    int32_t* num_rows_ptr = num_rows.has_value() ? num_rows->data_ptr<int32_t>() : nullptr;
    int32_t* smooth_scale_map_ptr =
        smooth_scale_map.has_value() ? smooth_scale_map->data_ptr<int32_t>() : nullptr;

    TORCH_CHECK(
        input.dim() < 4, __func__, " only support input dim <=3, but get dim: ", input.dim());
    int32_t input_dim0    = input.size(0);
    int32_t input_dim1    = input.dim() > 2 ? input.size(1) : 1;
    int32_t input_stride0 = input.stride(0);
    int32_t input_stride1 = input.dim() > 2 ? input.stride(1) : cols;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    dim3 const grid(rows);
    dim3 const block(BlockSize);
    if(out.dtype() == torch::kInt8)
    {
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
            smooth_per_token_scaled_quant_kernel, ck_tile::int8_t, cols);
    }
#ifdef GPU_ENABLE_FP8
    else if(out.dtype() == torch_fp8)
    {
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
            smooth_per_token_scaled_quant_kernel, FP8_TYPE, cols);
    }
#endif
#if defined(__Float4_e2m1fn_x2)
#if defined(TORCH_Float4_e2m1fn_x2)
    else if(out.dtype() == torch::kFloat4_e2m1fn_x2 || out.dtype() == torch::kUInt8)
#else
    else if(out.dtype() == torch::kUInt8)
#endif
    {
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
            smooth_per_token_scaled_quant_kernel, ck_tile::fp4x2_t, cols);
    }
#endif
    else
    {
        TORCH_CHECK(false, __func__, " not support output type: ", out.dtype());
    }
}

template <typename DTYPE, int BLOCK_SIZE = 256, int thread_data_size = 4, int MAX_ITERS = 10000>
__global__ void partial_transpose_kernel(DTYPE* __restrict__ out,
                                         DTYPE* __restrict__ input,
                                         const int* __restrict__ num_rows,
                                         const int cols)
{
    using vec_i                     = ck_tile::vec_t<DTYPE, thread_data_size>;
    int GRID_SIZE                   = gridDim.x;
    int ori_rows                    = *num_rows;
    int thread_per_row              = (cols + thread_data_size - 1) / thread_data_size;
    auto const* ptr_i               = reinterpret_cast<DTYPE const*>(input);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE);
    const int32_t oob_i             = (ori_rows * cols + ooba_i - 1) / ooba_i * ooba_i;
    auto buffer_i = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr_i, oob_i);
    buffer_i.init_raw();
    for(int i = 0; i < MAX_ITERS; i++)
    {
        int64_t y = i * GRID_SIZE * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE + threadIdx.x;
        int x     = y % thread_per_row * thread_data_size;
        y         = y / thread_per_row;
        if(y >= ori_rows)
            return;
        vec_i input_vecs   = buffer_i.template get<vec_i>(y * cols + x, 0, true);
        int64_t out_offset = x * ori_rows + y;
        // printf("blockIdx: %d, threadIdx:%d, y: %d, x: %d, ori_rows: %d, cols: %d, val:%f\n",
        // blockIdx.x, threadIdx.x, y, x, ori_rows, cols,
        // ck_tile::type_convert<float>(input_vecs[0]));
        for(int j = 0; j < thread_data_size; j++)
        {
            if((x + j) < cols)
            {
                out[out_offset + j * ori_rows] = input_vecs[j];
            }
        }
    }
}

void partial_transpose(torch::Tensor& out,         // [rows, d]
                       torch::Tensor const& input, // [rows, d]
                       torch::Tensor const& num_rows)
{
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(input.is_contiguous());

    uint32_t num_cu       = get_num_cu_func();
    int const cols        = input.size(-1);
    int const rows        = input.numel() / cols;
    int32_t* num_rows_ptr = num_rows.data_ptr<int32_t>();

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    if(cols <= 1024)
    {
        const int BlockSize        = 256;
        const int GridSize         = num_cu * 8; // Adjust as needed
        const int thread_data_size = 1024 / BlockSize;

        dim3 grid(GridSize);
        dim3 block(BlockSize);

        VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "partial_transpose_kernel", [&] {

            using input_dtype = typename t2ck<scalar_t>::type;
            aiter::partial_transpose_kernel<input_dtype, BlockSize, thread_data_size>
                <<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(out.data_ptr()),
                                             reinterpret_cast<input_dtype*>(input.data_ptr()),
                                             num_rows_ptr,
                                             cols);
        });
    }
    else if(cols <= 2048)
    {
        const int BlockSize        = 256;
        const int GridSize         = num_cu * 4;
        const int thread_data_size = 2048 / BlockSize;

        dim3 grid(GridSize);
        dim3 block(BlockSize);

        VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "partial_transpose_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            aiter::partial_transpose_kernel<input_dtype, BlockSize, thread_data_size>
                <<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(out.data_ptr()),
                                             reinterpret_cast<input_dtype*>(input.data_ptr()),
                                             num_rows_ptr,
                                             cols);
        });
    }
    else if(cols <= 4096)
    {
        const int BlockSize        = 256;
        const int GridSize         = num_cu * 2;
        const int thread_data_size = 4096 / BlockSize;

        dim3 grid(GridSize);
        dim3 block(BlockSize);

        VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "partial_transpose_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            aiter::partial_transpose_kernel<input_dtype, BlockSize, thread_data_size>
                <<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(out.data_ptr()),
                                             reinterpret_cast<input_dtype*>(input.data_ptr()),
                                             num_rows_ptr,
                                             cols);
        });
    }
    else if(cols <= 8192)
    {
        const int BlockSize        = 512;
        const int GridSize         = num_cu;
        const int thread_data_size = 8192 / BlockSize;

        dim3 grid(GridSize);
        dim3 block(BlockSize);

        VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "partial_transpose_kernel", [&] {
            using input_dtype = typename t2ck<scalar_t>::type;
            aiter::partial_transpose_kernel<input_dtype, BlockSize, thread_data_size>
                <<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(out.data_ptr()),
                                             reinterpret_cast<input_dtype*>(input.data_ptr()),
                                             num_rows_ptr,
                                             cols);
        });
    }
    else
    {
        TORCH_CHECK(false, __func__, " cols is not supported: ", cols);
    }
}

template <typename T, int N>
struct alignas(sizeof(T) * N) aligned_vector {
    T val[N];
    __host__ __device__ inline T& operator[](int i) { return val[i]; }
    __host__ __device__ inline const T& operator[](int i) const { return val[i]; }
};

// float -> int8 四舍五入
static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  float dst = std::nearbyint(x);
  dst = fminf(fmaxf(dst, i8_min), i8_max);
  return static_cast<int8_t>(dst);
#else
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

// Warp Reduce Max（使用 WARP_SIZE 64）
template <typename T, int reducesize = WARP_SIZE>
__inline__ __device__ T WarpReduceMax_ROW(T val) {
#pragma unroll
  for (int offset = reducesize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down(val, offset));
  }
  return val;
}

// Block Reduce Max
template <typename T, int block_size = 512>
__inline__ __device__ T BlockReduceMax_ROW(T val, T* shared) {
  constexpr int share_size = block_size / WARP_SIZE;
  val = WarpReduceMax_ROW<T>(val);
  if constexpr (block_size == WARP_SIZE) {
    return val;
  } else {
    const int lid = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    if (lid == 0 && wid < share_size) {
      shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0 && lid < share_size) {
      val = WarpReduceMax_ROW<T, share_size>(shared[lid]);
    }
    return val;
  }
}

// SwiGLU 的 Silu 激活
template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  constexpr float LOG2E = 1.44269504088896340736f;
  return (T)(((float)x) / (1.0f + __builtin_amdgcn_exp2f(-((float)x) * LOG2E)));
}

template <typename scalar_t, bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? silu_kernel(x) * y : x * silu_kernel(y);
}

//------------------------------------------------------------------------------
// Kernel 1: 通用 fallback
//------------------------------------------------------------------------------
template <typename scalar_t, bool act_first, int block_size>
__global__ void moe_swiglu_dynamic_quant_kernel_gernel(
    int64_t num_tokens,
    int8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const float* __restrict__ smooth,
    int* __restrict__ experts_tokens_count,
    int* __restrict__ experts_tokens_start,
    const int d,
    const int num_experts) {
  int64_t token_idx = blockIdx.x;
  const int64_t input_offset = token_idx * 2 * d;
  const int64_t output_offset = token_idx * d;

  constexpr int share_size = block_size / WARP_SIZE;
  __shared__ float shared_mem[share_size];
  __shared__ int s_expert_index;

  if (threadIdx.x == 0) {
    int expert_idx = -1;
    for (int i = 0; i < num_experts; ++i) {
      int start = experts_tokens_start[i];
      int count = experts_tokens_count[i];
      if (token_idx >= start && token_idx < start + count) {
        expert_idx = i;
        break;
      }
    }
    s_expert_index = expert_idx;
  }
  __syncthreads();

  for (; token_idx < num_tokens; token_idx += gridDim.x) {
    int expert_index = s_expert_index;
    if (expert_index == -1) {
      for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        out[output_offset + idx] = 0;
      }
      if (threadIdx.x == 0) {
        scales[token_idx] = 0.0f;
      }
      return;
    }

    const int64_t smooth_offset = expert_index * d;

    float row_max = 0.0f;
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      const float smooth_val = VLLM_LDG(&smooth[smooth_offset + idx]);
      float val = static_cast<float>(compute<scalar_t, act_first>(x, y)) * smooth_val;
      row_max = fmaxf(row_max, fabsf(val));
    }

    row_max = BlockReduceMax_ROW<float, block_size>(row_max, shared_mem);

    __shared__ float s_token_scale;
    if (threadIdx.x == 0) {
      s_token_scale = row_max;
      scales[token_idx] = s_token_scale / 127.f;
    }
    __syncthreads();

    float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;

    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&input[input_offset + idx]);
      const scalar_t y = VLLM_LDG(&input[input_offset + d + idx]);
      const float smooth_val = VLLM_LDG(&smooth[smooth_offset + idx]);
      float val = static_cast<float>(compute<scalar_t, act_first>(x, y)) * smooth_val;
      int8_t q_val = float_to_int8_rn(val * inv_s);
      out[output_offset + idx] = q_val;
    }
  }
}

//------------------------------------------------------------------------------
// Kernel 2: 单 warp 版（d <= 1024，VEC=16）
//------------------------------------------------------------------------------
template <typename scalar_t, int VEC, bool act_first, int block_size>
__global__ void moe_swiglu_dynamic_quant_kernel_one_warp(
    int64_t num_tokens,
    int8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const float* __restrict__ smooth,
    int* __restrict__ experts_tokens_count,
    int* __restrict__ experts_tokens_start,
    const int d,
    const int num_experts) {
  int64_t token_idx = blockIdx.x;
  int tidx = threadIdx.x;
  int idx = threadIdx.x * VEC;
  constexpr int MAX_EXPERTS = 64;
  __shared__ int sec[MAX_EXPERTS];
  __shared__ int ses[MAX_EXPERTS];

  using VecType = aiter::aligned_vector<scalar_t, VEC>;
  using VecInt8Type = aiter::aligned_vector<int8_t, VEC>;
  using VecFloatType = aiter::aligned_vector<float, VEC>;

  if (tidx < num_experts && tidx < MAX_EXPERTS) {
    sec[tidx] = experts_tokens_count[tidx];
    ses[tidx] = experts_tokens_start[tidx];
  }
  __syncthreads();

  for (; token_idx < num_tokens; token_idx += gridDim.x) {
    int expert_index = -1;
    if (tidx == 0) {
      int left = 0, right = num_experts - 1, res = -1;
      while (left <= right) {
        int mid = (left + right) >> 1;
        int start = ses[mid];
        if (start <= token_idx) {
          res = mid;
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
      if (res != -1) {
        int start = ses[res];
        int count = sec[res];
        if (token_idx >= start && token_idx < start + count) {
          expert_index = res;
        }
      }
    }
    expert_index = __shfl(expert_index, 0, WARP_SIZE);
    if (expert_index == -1) return;

    const int64_t y_index = token_idx * d + idx;
    VecInt8Type* y = (VecInt8Type*)(out + y_index);

    const int64_t x_index = token_idx * 2 * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecFloatType* smooth_vec = (VecFloatType*)(smooth + expert_index * d + idx);

    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    float r_smooth[VEC];
    float r_y[VEC];

    if (idx < d) {
      *(VecType*)r_x1 = *x1;
      *(VecType*)r_x2 = *x2;
      *(VecFloatType*)r_smooth = *smooth_vec;
#pragma unroll
      for (int i = 0; i < VEC; i++) {
        float silu1 = static_cast<float>(silu_kernel(r_x1[i]));
        float silu2 = static_cast<float>(r_x2[i]);
        r_y[i] = silu1 * silu2 * r_smooth[i];
      }
    }

    float row_max = 0.f;
    if (idx < d) {
#pragma unroll
      for (int ii = 0; ii < VEC; ii++) {
        row_max = fmaxf(row_max, fabsf(r_y[ii]));
      }
    }

    row_max = WarpReduceMax_ROW<float>(row_max);
    float quant_scale = 1.0f;
    if (tidx == 0) {
      quant_scale = 127.0f / row_max;
      scales[token_idx] = row_max / 127.f;
    }
    quant_scale = __shfl(quant_scale, 0, WARP_SIZE);

    int8_t out_vec[VEC];
    if (idx < d) {
#pragma unroll
      for (int ii = 0; ii < VEC; ii++) {
        out_vec[ii] = float_to_int8_rn(r_y[ii] * quant_scale);
      }
      *y = *(VecInt8Type*)out_vec;
    }
  }
}

//------------------------------------------------------------------------------
// Kernel 3: 主版本（block 级，多 warp）
//------------------------------------------------------------------------------
template <typename scalar_t, int VEC, bool act_first, int block_size>
__global__ void moe_swiglu_dynamic_quant_kernel(
    int64_t num_tokens,
    int8_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    float* __restrict__ scales,
    const float* __restrict__ smooth,
    int* __restrict__ experts_tokens_count,
    int* __restrict__ experts_tokens_start,
    const int d,
    const int num_experts) {
  int64_t token_idx = blockIdx.x;
  int tidx = threadIdx.x;
  int idx = threadIdx.x * VEC;
  constexpr int MAX_EXPERTS = 64;
  __shared__ int sec[MAX_EXPERTS];
  __shared__ int ses[MAX_EXPERTS];

  constexpr int share_size = block_size / WARP_SIZE;
  __shared__ float val_shared[share_size];
  __shared__ int s_expert_index;

  using VecType = aiter::aligned_vector<scalar_t, VEC>;
  using VecInt8Type = aiter::aligned_vector<int8_t, VEC>;
  using VecFloatType = aiter::aligned_vector<float, VEC>;

  if (tidx < num_experts && tidx < MAX_EXPERTS) {
    sec[tidx] = experts_tokens_count[tidx];
    ses[tidx] = experts_tokens_start[tidx];
  }
  __syncthreads();

  for (; token_idx < num_tokens; token_idx += gridDim.x) {
    int local_expert_index = -1;
    if (tidx == 0) {
      int left = 0, right = num_experts - 1, res = -1;
      while (left <= right) {
        int mid = (left + right) >> 1;
        int start = ses[mid];
        if (start <= token_idx) {
          res = mid;
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
      if (res != -1) {
        int start = ses[res];
        int count = sec[res];
        if (token_idx >= start && token_idx < start + count) {
          local_expert_index = res;
        }
      }
      s_expert_index = local_expert_index;
    }
    __syncthreads();
    int expert_index = s_expert_index;
    if (expert_index == -1) return;

    const int64_t y_index = token_idx * d + idx;
    VecInt8Type* y = (VecInt8Type*)(out + y_index);

    const int64_t x_index = token_idx * 2 * d + idx;
    VecType* x1 = (VecType*)(input + x_index);
    VecType* x2 = (VecType*)(input + x_index + d);
    VecFloatType* smooth_vec = (VecFloatType*)(smooth + expert_index * d + idx);

    scalar_t r_x1[VEC];
    scalar_t r_x2[VEC];
    float r_smooth[VEC];
    float r_y[VEC];

    if (idx < d) {
      *(VecType*)r_x1 = *x1;
      *(VecType*)r_x2 = *x2;
      *(VecFloatType*)r_smooth = *smooth_vec;
#pragma unroll
      for (int i = 0; i < VEC; i++) {
        float silu1 = static_cast<float>(silu_kernel(r_x1[i]));
        float silu2 = static_cast<float>(r_x2[i]);
        r_y[i] = silu1 * silu2 * r_smooth[i];
      }
    }

    float row_max = 0.f;
    if (idx < d) {
#pragma unroll
      for (int ii = 0; ii < VEC; ii++) {
        row_max = fmaxf(row_max, fabsf(r_y[ii]));
      }
    }

    row_max = BlockReduceMax_ROW<float, block_size>(row_max, val_shared);
    __shared__ float s_token_scale;

    if (tidx == 0) {
      s_token_scale = row_max;
      scales[token_idx] = s_token_scale / 127.f;
    }
    __syncthreads();

    float inv_s = (s_token_scale == 0.f) ? 0.f : 127.f / s_token_scale;
    int8_t out_vec[VEC];

    if (idx < d) {
#pragma unroll
      for (int ii = 0; ii < VEC; ii++) {
        out_vec[ii] = float_to_int8_rn(r_y[ii] * inv_s);
      }
      *y = *(VecInt8Type*)out_vec;
    }
  }
}

//------------------------------------------------------------------------------
// Host Launcher
//------------------------------------------------------------------------------
void moe_swiglu_dynamic_quant(torch::Tensor& scatter_tokens,
                              torch::Tensor& smooth,
                              torch::Tensor& experts_tokens_count,
                              torch::Tensor& experts_tokens_start,
                              torch::Tensor& output,
                              torch::Tensor& scales,
                              float beta) {
  int d = scatter_tokens.size(-1) / 2;
  int64_t num_tokens = scatter_tokens.numel() / scatter_tokens.size(-1);
  int num_experts = experts_tokens_count.size(0);
  int grid_opt = num_tokens;

  if (num_tokens == 9216 || num_tokens == 10240 || num_tokens == 11264 ||
      num_tokens == 12288 || num_tokens == 13312 || num_tokens == 14336) {
    grid_opt = 8192;
  } else if (num_tokens == 3072 || num_tokens == 4096 ||
             num_tokens == 5120 || num_tokens == 6144 || num_tokens == 7168) {
    grid_opt = 2048;
  } else if (num_tokens <= 2048 && num_tokens >= 1024) {
    grid_opt = 1024;
  } else {
    grid_opt = num_tokens;
  }

  dim3 grid(grid_opt);
  if (num_tokens == 0) {
    return;
  }

  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(scatter_tokens));
  const hipStream_t stream = at::hip::getCurrentHIPStream();

  AITER_DISPATCH_FLOATING16_TYPES(
      scatter_tokens.scalar_type(), "moe_swiglu_dynamic_quant_kernel", [&] {
        if (d <= 512) {
          moe_swiglu_dynamic_quant_kernel<scalar_t, 2, true, 256>
              <<<grid, 256, 0, stream>>>(num_tokens,
                                         output.data_ptr<int8_t>(),
                                         scatter_tokens.data_ptr<scalar_t>(),
                                         scales.data_ptr<float>(),
                                         smooth.data_ptr<float>(),
                                         experts_tokens_count.data_ptr<int>(),
                                         experts_tokens_start.data_ptr<int>(),
                                         d, num_experts);
        } else if (d <= 1024) {
          moe_swiglu_dynamic_quant_kernel_one_warp<scalar_t, 16, true, 64>
              <<<grid, 64, 0, stream>>>(num_tokens,
                                        output.data_ptr<int8_t>(),
                                        scatter_tokens.data_ptr<scalar_t>(),
                                        scales.data_ptr<float>(),
                                        smooth.data_ptr<float>(),
                                        experts_tokens_count.data_ptr<int>(),
                                        experts_tokens_start.data_ptr<int>(),
                                        d, num_experts);
        } else if (d <= 2048) {
          moe_swiglu_dynamic_quant_kernel<scalar_t, 16, true, 128>
              <<<grid, 128, 0, stream>>>(num_tokens,
                                         output.data_ptr<int8_t>(),
                                         scatter_tokens.data_ptr<scalar_t>(),
                                         scales.data_ptr<float>(),
                                         smooth.data_ptr<float>(),
                                         experts_tokens_count.data_ptr<int>(),
                                         experts_tokens_start.data_ptr<int>(),
                                         d, num_experts);
        } else if (d <= 4096) {
          moe_swiglu_dynamic_quant_kernel<scalar_t, 16, true, 256>
              <<<grid, 256, 0, stream>>>(num_tokens,
                                         output.data_ptr<int8_t>(),
                                         scatter_tokens.data_ptr<scalar_t>(),
                                         scales.data_ptr<float>(),
                                         smooth.data_ptr<float>(),
                                         experts_tokens_count.data_ptr<int>(),
                                         experts_tokens_start.data_ptr<int>(),
                                         d, num_experts);
        } else if (d <= 8192) {
          moe_swiglu_dynamic_quant_kernel<scalar_t, 16, true, 512>
              <<<grid, 512, 0, stream>>>(num_tokens,
                                         output.data_ptr<int8_t>(),
                                         scatter_tokens.data_ptr<scalar_t>(),
                                         scales.data_ptr<float>(),
                                         smooth.data_ptr<float>(),
                                         experts_tokens_count.data_ptr<int>(),
                                         experts_tokens_start.data_ptr<int>(),
                                         d, num_experts);
        } else if (d <= 16384) {
          moe_swiglu_dynamic_quant_kernel<scalar_t, 16, true, 1024>
              <<<grid, 1024, 0, stream>>>(num_tokens,
                                          output.data_ptr<int8_t>(),
                                          scatter_tokens.data_ptr<scalar_t>(),
                                          scales.data_ptr<float>(),
                                          smooth.data_ptr<float>(),
                                          experts_tokens_count.data_ptr<int>(),
                                          experts_tokens_start.data_ptr<int>(),
                                          d, num_experts);
        } else if (d <= 32768) {
          moe_swiglu_dynamic_quant_kernel<scalar_t, 32, true, 1024>
              <<<grid, 1024, 0, stream>>>(num_tokens,
                                          output.data_ptr<int8_t>(),
                                          scatter_tokens.data_ptr<scalar_t>(),
                                          scales.data_ptr<float>(),
                                          smooth.data_ptr<float>(),
                                          experts_tokens_count.data_ptr<int>(),
                                          experts_tokens_start.data_ptr<int>(),
                                          d, num_experts);
        } else {
          moe_swiglu_dynamic_quant_kernel_gernel<scalar_t, true, 1024>
              <<<grid, 1024, 0, stream>>>(num_tokens,
                                          output.data_ptr<int8_t>(),
                                          scatter_tokens.data_ptr<scalar_t>(),
                                          scales.data_ptr<float>(),
                                          smooth.data_ptr<float>(),
                                          experts_tokens_count.data_ptr<int>(),
                                          experts_tokens_start.data_ptr<int>(),
                                          d, num_experts);
        }
      });
}
} // namespace aiter
