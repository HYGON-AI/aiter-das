// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/host/hip_device_info.hpp"
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#ifndef AITER_QUANT_ENABLE_MULTIDIE
#define AITER_QUANT_ENABLE_MULTIDIE 0
#endif

namespace aiter {
namespace quant_detail {

inline unsigned int read_unsigned(const char* name, unsigned int fallback)
{
    const char* value = std::getenv(name);
    if(value == nullptr)
        return fallback;
    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtoull(value, &end, 0);
    if(value[0] == '\0' || value[0] == '-' || end == value || *end != '\0' ||
       errno == ERANGE || parsed > std::numeric_limits<unsigned int>::max())
        throw std::invalid_argument(std::string("invalid unsigned value for ") + name);
    return static_cast<unsigned int>(parsed);
}

#if AITER_QUANT_ENABLE_MULTIDIE
// The target HIP SDK stores a dieConfig pointer in captured nodes. Preserve it
// through Graph/GraphExec lifetimes, as in the CK LayerNorm launcher.
inline hipLaunchMultiDieConfig* retain_graph_config(hipStream_t stream,
                                                    const hipLaunchMultiDieConfig& config)
{
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    hipGraph_t graph = nullptr;
    HIP_CHECK_ERROR(hipStreamGetCaptureInfo_v2(stream, &capture, nullptr, &graph));
    if(capture != hipStreamCaptureStatusActive)
        return nullptr;
    auto owned = std::make_unique<hipLaunchMultiDieConfig>(config);
    hipUserObject_t object = nullptr;
    HIP_CHECK_ERROR(hipUserObjectCreate(
        &object, owned.get(),
        [](void* ptr) { delete static_cast<hipLaunchMultiDieConfig*>(ptr); },
        1, hipUserObjectNoDestructorSync));
    auto* pointer = owned.release();
    const auto status = hipGraphRetainUserObject(graph, object, 1, hipGraphUserObjectMove);
    if(status != hipSuccess)
    {
        HIP_CHECK_ERROR(hipUserObjectRelease(object));
        HIP_CHECK_ERROR(status);
    }
    return pointer;
}
#endif

// Deduce argument types from the kernel signature, then convert call arguments.
// In particular group kernels take int64_t rows even when host rows is int32_t.
template <typename T> struct kernel_argument { using type = T; };
template <typename... Args>
inline void launch(void (*kernel)(Args...), dim3 grid, dim3 block, hipStream_t stream,
                   typename kernel_argument<Args>::type... args)
{
    const char* mode = std::getenv("AITER_QUANT_DIE_MODE");
    if(mode == nullptr)
        mode = "auto";
    const bool automatic = std::strcmp(mode, "auto") == 0;
    const bool off = std::strcmp(mode, "off") == 0;
    if(!automatic && !off && std::strcmp(mode, "linear") != 0)
        throw std::invalid_argument("AITER_QUANT_DIE_MODE must be auto, off or linear");

    bool multi = false;
    int xcds = 0;
    unsigned int chunk = 0;
#if AITER_QUANT_ENABLE_MULTIDIE
    if(!off)
    {
        const auto& info = ck_tile::get_hip_device_info();
        xcds = info.num_xcds;
        multi = info.arch_name == "gfx946" && xcds >= 2;
        if(!automatic && !multi)
            throw std::runtime_error("Quantization linear launch requires gfx946 with at least 2 reported XCDs");
        multi = multi && (!automatic || grid.x >= 2);
    }
#else
    if(!automatic && !off)
        throw std::runtime_error("Quantization linear launch requires a build targeting gfx946");
#endif
    if(multi)
    {
        chunk = read_unsigned("AITER_QUANT_DIE_CHUNK", 1);
        if(chunk == 0)
            throw std::invalid_argument("AITER_QUANT_DIE_CHUNK must be positive");
        if(read_unsigned("AITER_QUANT_DIE_MASK", 0) != 0)
            throw std::invalid_argument("AITER_QUANT_DIE_MASK supports only 0 (all dies)");
    }
    const char* log = std::getenv("AITER_QUANT_LOG_LAUNCH");
    if(log != nullptr && std::strcmp(log, "0") != 0)
        std::cerr << "[AITER quant launch] mode=" << (multi ? "linear" : "ordinary")
                  << " grid=" << grid.x << " block=" << block.x
                  << " chunk=" << chunk << " xcds=" << xcds << '\n';

    void* parameters[] = {static_cast<void*>(&args)...};
#if AITER_QUANT_ENABLE_MULTIDIE
    if(multi)
    {
        hipLaunchMultiDieConfig config{};
        config.linearMode.mode = hipDispatchLinearMode;
        config.linearMode.dieMask = hipDieAll;
        config.linearMode.chunkSize = chunk;
        auto* graph_config = retain_graph_config(stream, config);
        HIP_CHECK_ERROR(hipLaunchKernelMultiDie(reinterpret_cast<const void*>(kernel),
                                              grid, block, parameters, 0, stream,
                                              graph_config ? graph_config : &config));
        return;
    }
#endif
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<const void*>(kernel),
                                   grid, block, parameters, 0, stream));
}
} // namespace quant_detail
} // namespace aiter
