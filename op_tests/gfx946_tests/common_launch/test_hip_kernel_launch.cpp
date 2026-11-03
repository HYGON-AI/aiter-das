// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Standalone HIP regression: see README.md for compile/run commands.
#include "hip_kernel_launch.h"
#include <cstdint>

__global__ void launch_contract_kernel(int64_t* output, int64_t rows, double value)
{
    extern __shared__ int64_t scratch[];
    scratch[threadIdx.x] = rows + static_cast<int64_t>(value) + threadIdx.x;
    __syncthreads();
    if(threadIdx.x == 0)
        output[blockIdx.x] = scratch[blockDim.x - 1];
}

template <typename F>
void expect_error(F&& fn, const char* message)
{
    try { fn(); }
    catch(const std::exception& e)
    {
        if(std::string(e.what()).find(message) != std::string::npos)
            return;
        throw;
    }
    throw std::runtime_error(std::string("expected error: ") + message);
}

int main()
{
    try
    {
        const auto& info = ck_tile::get_hip_device_info();
        const bool multi = AITER_ENABLE_MULTIDIE && info.arch_name == "gfx946" && info.num_xcds >= 2;
        unsetenv("GPU_CHIP"); // topology was initialized; routing must use the device.
        hipStream_t stream;
        HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        int64_t* output = nullptr;
        HIP_CHECK_ERROR(hipMalloc(&output, 5 * sizeof(int64_t)));
        const int32_t rows = 0x12345678;
        const float value = 7.0f;
        int checks = 0;
        for(const auto& p : {aiter::launch_policy::silu, aiter::launch_policy::quant,
                            aiter::launch_policy::moe_topk, aiter::launch_policy::moe_sum,
                            aiter::launch_policy::normal_gemm})
        {
            auto launch = [&](unsigned groups) {
                aiter::launch_kernel(p, launch_contract_kernel, dim3(groups), dim3(64),
                                     64 * sizeof(int64_t), stream, output, rows, value);
            };
            auto verify = [&](unsigned groups) {
                launch(groups);
                HIP_CHECK_ERROR(hipStreamSynchronize(stream));
                int64_t result[5]{};
                HIP_CHECK_ERROR(hipMemcpy(result, output, groups * sizeof(int64_t), hipMemcpyDeviceToHost));
                for(unsigned i = 0; i < groups; ++i)
                    if(result[i] != int64_t(rows) + 7 + 63)
                        throw std::runtime_error("kernel ABI/dynamic LDS mismatch");
                ++checks;
            };
            setenv(p.mode_env, "auto", 1);
            setenv(p.chunk_env, "1", 1);
            setenv(p.mask_env, "0", 1);
            setenv(p.log_env, "1", 1);
            verify(5);
            // auto single-TG and off must ignore unused chunk/mask values.
            setenv(p.chunk_env, "invalid", 1);
            setenv(p.mask_env, "invalid", 1);
            verify(1);
            setenv(p.mode_env, "off", 1);
            verify(5);
            setenv(p.mode_env, "invalid", 1);
            expect_error([&] { launch(5); }, "must be auto"); ++checks;
            if(p.skip_empty_grid)
            {
                expect_error([&] { launch(0); }, "must be auto"); ++checks;
                setenv(p.mode_env, "off", 1);
                launch(0); ++checks;
            }
            setenv(p.mode_env, "linear", 1);
            if(multi)
            {
                setenv(p.chunk_env, "0", 1);
                expect_error([&] { launch(5); }, "must be positive"); ++checks;
                setenv(p.chunk_env, "2", 1);
                setenv(p.mask_env, "1", 1);
                expect_error([&] { launch(5); }, "all dies"); ++checks;
                setenv(p.mask_env, "0", 1);
                verify(5);
            }
            else
            {
                expect_error([&] { launch(5); }, "requires"); ++checks;
            }
            unsetenv(p.mode_env); unsetenv(p.chunk_env); unsetenv(p.mask_env); unsetenv(p.log_env);
        }
        HIP_CHECK_ERROR(hipFree(output));
        HIP_CHECK_ERROR(hipStreamDestroy(stream));
        std::cout << "PASS checks=" << checks << " arch=" << info.arch_name
                  << " xcds=" << info.num_xcds << " multidie_build=" << AITER_ENABLE_MULTIDIE << '\n';
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
