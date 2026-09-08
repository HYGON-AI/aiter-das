// SPDX-License-Identifier: MIT

#include "aiter_c_ops.h"

#include <hip/hip_runtime.h>
#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Shape = std::tuple<int64_t, int64_t, int64_t>;

struct DTypeCase
{
    const char* name;
    at::ScalarType dtype;
    double rtol;
    double atol;
};

const std::vector<DTypeCase> kDTypes = {
    {"fp16", at::kHalf, 5e-2, 5e-2},
    {"bf16", at::kBFloat16, 5e-2, 5e-2},
    {"fp8", at::kFloat8_e4m3fn, 2e-1, 2e-1},
    {"int8", at::kChar, 0.0, 0.0},
};

void require(bool ok, const std::string& message)
{
    if(!ok)
    {
        throw std::runtime_error(message);
    }
}

at::ScalarType output_dtype(at::ScalarType dtype)
{
    if(dtype == at::kChar)
    {
        return at::kInt;
    }
    if(dtype == at::kFloat8_e4m3fn)
    {
        return at::kFloat;
    }
    return dtype;
}

torch::Tensor make_a(int64_t m, int64_t k, at::ScalarType dtype)
{
    const auto options = torch::TensorOptions().device(torch::kCUDA);
    if(dtype == at::kChar)
    {
        auto values =
            torch::remainder(torch::arange(k, options.dtype(torch::kInt16)), 5).to(torch::kInt8);
        return values.view({1, k}).repeat({m, 1}).contiguous();
    }
    if(dtype == at::kFloat8_e4m3fn)
    {
        auto values =
            torch::remainder(torch::arange(k, options.dtype(torch::kFloat16)), 5).to(dtype);
        return values.view({1, k}).repeat({m, 1}).contiguous();
    }
    return (torch::randn({m, k}, options.dtype(torch::kFloat16)) / 10)
        .to(dtype)
        .contiguous();
}

torch::Tensor make_b(int64_t n, int64_t k, at::ScalarType dtype)
{
    if(dtype == at::kChar || dtype == at::kFloat8_e4m3fn)
    {
        auto b = torch::zeros(
            {n, k}, torch::TensorOptions().device(torch::kCUDA).dtype(dtype));
        if(dtype == at::kChar)
        {
            b[0].fill_(1);
        }
        else
        {
            b[0].copy_(torch::ones(
                           {k},
                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16))
                           .to(dtype));
        }
        return b.contiguous();
    }
    return make_a(n, k, dtype);
}

torch::Tensor reference_rc(const torch::Tensor& a, const torch::Tensor& b)
{
    return torch::matmul(a.to(torch::kFloat), b.to(torch::kFloat).transpose(0, 1));
}

void check_outputs(const char* case_name,
                   const DTypeCase& dtype_case,
                   const std::vector<torch::Tensor>& outputs,
                   const std::vector<torch::Tensor>& refs)
{
    require(outputs.size() == refs.size(),
            std::string(case_name) + " output count mismatch");
    for(std::size_t i = 0; i < outputs.size(); ++i)
    {
        auto out = outputs[i].to(torch::kFloat);
        auto ref = refs[i].to(torch::kFloat);
        require(out.sizes() == ref.sizes(),
                std::string(case_name) + " output shape mismatch at group " +
                    std::to_string(i));
        if(!torch::allclose(out, ref, dtype_case.rtol, dtype_case.atol, true))
        {
            const double max_abs = (out - ref).abs().max().item<double>();
            throw std::runtime_error(std::string(case_name) + " " + dtype_case.name +
                                     " mismatch at group " + std::to_string(i) +
                                     ", max_abs=" + std::to_string(max_abs));
        }
    }
}

std::vector<Shape> heterogeneous_shapes(at::ScalarType dtype)
{
    if(dtype == at::kHalf || dtype == at::kBFloat16)
    {
        return {{128, 128, 128}, {192, 256, 128}, {256, 128, 128}};
    }
    if(dtype == at::kFloat8_e4m3fn)
    {
        return {{128, 128, 128}, {256, 128, 128}, {128, 256, 128}};
    }
    return {{32, 32, 128}, {64, 64, 128}, {96, 32, 128}};
}

void make_inputs(const std::vector<Shape>& shapes,
                 at::ScalarType dtype,
                 std::vector<torch::Tensor>& a_tensors,
                 std::vector<torch::Tensor>& b_tensors,
                 std::vector<torch::Tensor>& refs)
{
    for(const auto& [m, n, k] : shapes)
    {
        auto a = make_a(m, k, dtype);
        auto b = make_b(n, k, dtype);
        refs.push_back(reference_rc(a, b));
        a_tensors.push_back(std::move(a));
        b_tensors.push_back(std::move(b));
    }
}

void run_heterogeneous_cases()
{
    std::cout << "=== heterogeneous grouped GEMM cases ===\n";
    for(const auto& dtype_case : kDTypes)
    {
        std::vector<torch::Tensor> a_tensors;
        std::vector<torch::Tensor> b_tensors;
        std::vector<torch::Tensor> refs;
        make_inputs(heterogeneous_shapes(dtype_case.dtype),
                    dtype_case.dtype,
                    a_tensors,
                    b_tensors,
                    refs);

        auto outputs = aiter::native::ck_grouped_gemm(a_tensors, b_tensors);
        check_outputs("heterogeneous", dtype_case, outputs, refs);
        std::cout << "  passed dtype=" << dtype_case.name << "\n";
    }
}

void run_moe_case(const DTypeCase& dtype_case)
{
    constexpr int64_t n = 128;
    constexpr int64_t k = 128;
    const std::vector<int64_t> m_values = {1, 17, 33, 63, 65, 100};
    std::vector<Shape> shapes;
    for(int64_t m : m_values)
    {
        shapes.emplace_back(m, n, k);
    }

    std::vector<torch::Tensor> a_tensors;
    std::vector<torch::Tensor> b_tensors;
    std::vector<torch::Tensor> refs;
    make_inputs(shapes, dtype_case.dtype, a_tensors, b_tensors, refs);

    auto outputs = aiter::native::ck_grouped_gemm(a_tensors, b_tensors);
    check_outputs("moe alloc", dtype_case, outputs, refs);

    std::vector<torch::Tensor> c_tensors;
    for(int64_t m : m_values)
    {
        c_tensors.push_back(torch::empty(
            {m, n},
            torch::TensorOptions().device(torch::kCUDA).dtype(output_dtype(dtype_case.dtype))));
    }
    auto outputs_out =
        aiter::native::ck_grouped_gemm_out(a_tensors, b_tensors, c_tensors);
    check_outputs("moe out", dtype_case, outputs_out, refs);
    require(outputs_out.size() == c_tensors.size(), "moe out count mismatch");
    for(std::size_t i = 0; i < outputs_out.size(); ++i)
    {
        require(outputs_out[i].data_ptr() == c_tensors[i].data_ptr(),
                "ck_grouped_gemm_out did not return caller-provided tensor");
    }
}

void run_moe_cases()
{
    std::cout << "=== MOE dynamic-M grouped GEMM cases ===\n";
    for(const auto& dtype_case : kDTypes)
    {
        if(dtype_case.dtype == at::kChar)
        {
            continue;
        }
        run_moe_case(dtype_case);
        std::cout << "  passed dtype=" << dtype_case.name << " alloc/out\n";
    }
}

} // namespace

int main()
{
    if(!torch::cuda::is_available())
    {
        std::cerr << "CUDA/HIP device is not available\n";
        return 1;
    }

    torch::manual_seed(20260624);
    run_heterogeneous_cases();
    run_moe_cases();
    require(hipDeviceSynchronize() == hipSuccess, "hipDeviceSynchronize failed");
    std::cout << "all libtorch ck_grouped_gemm tests passed\n";
    return 0;
}
