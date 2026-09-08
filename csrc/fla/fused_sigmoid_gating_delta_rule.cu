// SPDX-License-Identifier: MIT

#include <ATen/Dispatch.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "fla.h"

#ifndef AITER_HIP_KERNEL_LAUNCH_CHECK
#if defined(C10_CUDA_KERNEL_LAUNCH_CHECK)
#define AITER_HIP_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#elif defined(C10_HIP_KERNEL_LAUNCH_CHECK)
#define AITER_HIP_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()
#else
#define AITER_HIP_KERNEL_LAUNCH_CHECK() C10_HIP_CHECK(hipGetLastError())
#endif
#endif

namespace {

constexpr int kSupportedK = 128;
constexpr int kThreads = 64;
constexpr int kWaveSize = 64;
constexpr int kTileThreads = 256;
constexpr int kDefaultKernelVariant = 16;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t x)
{
    return static_cast<float>(x);
}

__device__ __forceinline__ float sigmoidf_plain(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus_threshold(float x, float beta, float threshold)
{
    const float bx = beta * x;
    return bx <= threshold ? log1pf(expf(bx)) / beta : x;
}

__device__ __forceinline__ float wave_sum_64(float value)
{
#pragma unroll
    for (int mask = kWaveSize / 2; mask > 0; mask >>= 1) {
        value += __shfl_xor(value, mask, kWaveSize);
    }
    return value;
}

template <int GroupSize>
__device__ __forceinline__ float group_sum_8(float value)
{
#pragma unroll
    for (int mask = GroupSize / 2; mask > 0; mask >>= 1) {
        value += __shfl_xor(value, mask, GroupSize);
    }
    return value;
}

template <int GroupSize>
__device__ __forceinline__ float group_broadcast_8(float value, int src_lane)
{
    return src_lane == 0 ? group_sum_8<GroupSize>(value) : value;
}

template <typename scalar_t, int GroupSize, int TileRows, int TileThreads>
__global__ __launch_bounds__(TileThreads)
void fused_sigmoid_gating_delta_rule_tiled_kernel(
    const scalar_t* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ dt_bias,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ o,
    float* __restrict__ state,
    int64_t stride_state_slot,
    int64_t stride_state_hv,
    int64_t stride_state_v,
    int64_t stride_state_k,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ ssm_state_indices,
    const int32_t* __restrict__ num_accepted_tokens,
    float beta,
    float threshold,
    float scale,
    int B,
    int T_total,
    int N,
    int H,
    int HV,
    int V,
    int stride_indices_seq,
    bool is_varlen,
    bool is_spec_decoding,
    bool use_qk_l2norm)
{
    constexpr int elems_per_thread = kSupportedK / GroupSize;
    static_assert(kSupportedK % GroupSize == 0, "K must be divisible by GroupSize");
    static_assert(kWaveSize % GroupSize == 0, "wave64 must be divisible by GroupSize");
    static_assert(TileRows == TileThreads / GroupSize, "TileRows must match threads/group");

    const int n = blockIdx.x;
    const int hv = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave = tid / kWaveSize;
    const int lane = tid & (kWaveSize - 1);
    const int group_in_wave = lane / GroupSize;
    const int group_lane = lane & (GroupSize - 1);
    const int row = wave * (kWaveSize / GroupSize) + group_in_wave;
    const int v_col = blockIdx.z * TileRows + row;
    const bool valid_v = (n < N) && (hv < HV) && (v_col < V);

    int bos = n * T_total;
    int T_cur = T_total;
    if (is_varlen) {
        bos = static_cast<int>(cu_seqlens[n]);
        const int eos = static_cast<int>(cu_seqlens[n + 1]);
        T_cur = eos - bos;
    }

    if (T_cur <= 0) {
        return;
    }

    const int h = hv / (HV / H);
    const int init_t = is_spec_decoding
        ? static_cast<int>(num_accepted_tokens[n]) - 1
        : 0;
    const int init_state_idx =
        static_cast<int>(ssm_state_indices[n * stride_indices_seq + init_t]);

    if (init_state_idx < 0) {
        return;
    }

    float h_frag[elems_per_thread];
    const int64_t init_state_base =
        static_cast<int64_t>(init_state_idx) * stride_state_slot +
        static_cast<int64_t>(hv) * stride_state_hv +
        static_cast<int64_t>(v_col) * stride_state_v;
#pragma unroll
    for (int i = 0; i < elems_per_thread; ++i) {
        const int kk = i * GroupSize + group_lane;
        h_frag[i] = valid_v
            ? state[init_state_base + static_cast<int64_t>(kk) * stride_state_k]
            : 0.0f;
    }

    __shared__ float q_s[kSupportedK];
    __shared__ float k_s[kSupportedK];
    __shared__ float q_norm_part[2];
    __shared__ float k_norm_part[2];
    __shared__ float inv_q_s;
    __shared__ float inv_k_s;
    __shared__ float gate_s;
    __shared__ float beta_gate_s;

    const float A = expf(to_float(A_log[hv]));
    const float dt_bias_value = to_float(dt_bias[hv]);

    for (int t = 0; t < T_cur; ++t) {
        const int token = bos + t;

        if (tid < kSupportedK) {
            const int64_t qk_off =
                (static_cast<int64_t>(token) * H + h) * kSupportedK + tid;
            const float q_val = to_float(q[qk_off]);
            const float k_val = to_float(k[qk_off]);
            q_s[tid] = q_val;
            k_s[tid] = k_val;

            float q_part = q_val * q_val;
            float k_part = k_val * k_val;
            q_part = wave_sum_64(q_part);
            k_part = wave_sum_64(k_part);
            if ((lane == 0) && (wave < 2)) {
                q_norm_part[wave] = q_part;
                k_norm_part[wave] = k_part;
            }
        }

        if (tid == 0) {
            const float a_value =
                to_float(a[static_cast<int64_t>(token) * HV + hv]);
            const float softplus_x =
                softplus_threshold(a_value + dt_bias_value, beta, threshold);
            gate_s = expf(-A * softplus_x);
            beta_gate_s =
                sigmoidf_plain(to_float(b[static_cast<int64_t>(token) * HV + hv]));
        }

        __syncthreads();

        if (tid == 0) {
            if (use_qk_l2norm) {
                inv_q_s = rsqrtf(q_norm_part[0] + q_norm_part[1] + 1.0e-6f);
                inv_k_s = rsqrtf(k_norm_part[0] + k_norm_part[1] + 1.0e-6f);
            } else {
                inv_q_s = 1.0f;
                inv_k_s = 1.0f;
            }
        }

        __syncthreads();

        const float gate = gate_s;
        const float beta_gate = beta_gate_s;
        const float inv_q = inv_q_s;
        const float inv_k = inv_k_s;

        float hk_part = 0.0f;
#pragma unroll
        for (int i = 0; i < elems_per_thread; ++i) {
            const int kk = i * GroupSize + group_lane;
            const float k_value = k_s[kk] * inv_k;
            h_frag[i] *= gate;
            hk_part += h_frag[i] * k_value;
        }
        const float hk = group_sum_8<GroupSize>(hk_part);

        const int64_t v_off =
            (static_cast<int64_t>(token) * HV + hv) * V + v_col;
        float v_value = 0.0f;
        if (valid_v && group_lane == 0) {
            v_value = to_float(v[v_off]);
        }
        v_value = group_broadcast_8<GroupSize>(v_value, 0);
        const float v_delta =
            (valid_v ? (v_value - hk) * beta_gate : 0.0f);

        float out_part = 0.0f;
#pragma unroll
        for (int i = 0; i < elems_per_thread; ++i) {
            const int kk = i * GroupSize + group_lane;
            const float k_value = k_s[kk] * inv_k;
            const float q_value = q_s[kk] * inv_q * scale;
            h_frag[i] += v_delta * k_value;
            out_part += h_frag[i] * q_value;
        }
        const float out_value = group_sum_8<GroupSize>(out_part);

        if (valid_v && group_lane == 0) {
            o[v_off] = static_cast<scalar_t>(out_value);
        }

        const int final_state_idx =
            static_cast<int>(ssm_state_indices[n * stride_indices_seq + t]);
        if (valid_v && final_state_idx >= 0) {
            const int64_t final_state_base =
                static_cast<int64_t>(final_state_idx) * stride_state_slot +
                static_cast<int64_t>(hv) * stride_state_hv +
                static_cast<int64_t>(v_col) * stride_state_v;
#pragma unroll
            for (int i = 0; i < elems_per_thread; ++i) {
                const int kk = i * GroupSize + group_lane;
                state[final_state_base + static_cast<int64_t>(kk) * stride_state_k] =
                    h_frag[i];
            }
        }

        __syncthreads();
    }
}

template <typename scalar_t, int kK>
__global__ __launch_bounds__(kThreads)
void fused_sigmoid_gating_delta_rule_kernel(
    const scalar_t* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ dt_bias,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ o,
    float* __restrict__ state,
    int64_t stride_state_slot,
    int64_t stride_state_hv,
    int64_t stride_state_v,
    int64_t stride_state_k,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ ssm_state_indices,
    const int32_t* __restrict__ num_accepted_tokens,
    float beta,
    float threshold,
    float scale,
    int B,
    int T_total,
    int N,
    int H,
    int HV,
    int V,
    int stride_indices_seq,
    bool is_varlen,
    bool is_spec_decoding,
    bool use_qk_l2norm)
{
    const int n = blockIdx.x;
    const int hv = blockIdx.y;
    const int v_col = blockIdx.z * blockDim.x + threadIdx.x;

    if (n >= N || hv >= HV || v_col >= V) {
        return;
    }

    int bos = n * T_total;
    int T_cur = T_total;
    if (is_varlen) {
        bos = static_cast<int>(cu_seqlens[n]);
        const int eos = static_cast<int>(cu_seqlens[n + 1]);
        T_cur = eos - bos;
    }

    if (T_cur <= 0) {
        return;
    }

    const int h = hv / (HV / H);
    const int init_t = is_spec_decoding
        ? static_cast<int>(num_accepted_tokens[n]) - 1
        : 0;
    const int init_state_idx =
        static_cast<int>(ssm_state_indices[n * stride_indices_seq + init_t]);

    if (init_state_idx < 0) {
        return;
    }

    float h_vec[kK];
    const int64_t init_state_base =
        static_cast<int64_t>(init_state_idx) * stride_state_slot +
        static_cast<int64_t>(hv) * stride_state_hv +
        static_cast<int64_t>(v_col) * stride_state_v;
#pragma unroll
    for (int kk = 0; kk < kK; ++kk) {
        h_vec[kk] =
            state[init_state_base + static_cast<int64_t>(kk) * stride_state_k];
    }

    const float A = expf(to_float(A_log[hv]));
    const float dt_bias_value = to_float(dt_bias[hv]);

    for (int t = 0; t < T_cur; ++t) {
        const int token = bos + t;

        float k_norm = 0.0f;
        float q_norm = 0.0f;
#pragma unroll
        for (int kk = 0; kk < kK; ++kk) {
            const int64_t qk_off =
                (static_cast<int64_t>(token) * H + h) * kK + kk;
            const float kval = to_float(k[qk_off]);
            const float qval = to_float(q[qk_off]);
            k_norm += kval * kval;
            q_norm += qval * qval;
        }

        float inv_k = 1.0f;
        float inv_q = 1.0f;
        if (use_qk_l2norm) {
            inv_k = rsqrtf(k_norm + 1.0e-6f);
            inv_q = rsqrtf(q_norm + 1.0e-6f);
        }

        const float a_value =
            to_float(a[static_cast<int64_t>(token) * HV + hv]);
        const float softplus_x =
            softplus_threshold(a_value + dt_bias_value, beta, threshold);
        const float gate = expf(-A * softplus_x);
        const float beta_gate =
            sigmoidf_plain(to_float(b[static_cast<int64_t>(token) * HV + hv]));

        float hk = 0.0f;
#pragma unroll
        for (int kk = 0; kk < kK; ++kk) {
            const int64_t k_off =
                (static_cast<int64_t>(token) * H + h) * kK + kk;
            const float k_value = to_float(k[k_off]) * inv_k;
            h_vec[kk] *= gate;
            hk += h_vec[kk] * k_value;
        }

        const int64_t v_off =
            (static_cast<int64_t>(token) * HV + hv) * V + v_col;
        const float v_delta = (to_float(v[v_off]) - hk) * beta_gate;

        float out_value = 0.0f;
#pragma unroll
        for (int kk = 0; kk < kK; ++kk) {
            const int64_t qk_off =
                (static_cast<int64_t>(token) * H + h) * kK + kk;
            const float k_value = to_float(k[qk_off]) * inv_k;
            const float q_value = to_float(q[qk_off]) * inv_q * scale;
            h_vec[kk] += v_delta * k_value;
            out_value += h_vec[kk] * q_value;
        }

        o[v_off] = static_cast<scalar_t>(out_value);

        const int final_state_idx =
            static_cast<int>(ssm_state_indices[n * stride_indices_seq + t]);
        if (final_state_idx >= 0) {
            const int64_t final_state_base =
                static_cast<int64_t>(final_state_idx) * stride_state_slot +
                static_cast<int64_t>(hv) * stride_state_hv +
                static_cast<int64_t>(v_col) * stride_state_v;
#pragma unroll
            for (int kk = 0; kk < kK; ++kk) {
                state[final_state_base + static_cast<int64_t>(kk) * stride_state_k] =
                    h_vec[kk];
            }
        }
    }
}

void check_same_dtype(
    const torch::Tensor& ref,
    const torch::Tensor& tensor,
    const char* name)
{
    TORCH_CHECK(
        tensor.scalar_type() == ref.scalar_type(),
        "vllm_fused_sigmoid_gating_delta_rule_update: ",
        name,
        " must have the same dtype as q");
}

void check_fp32_dtype(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "vllm_fused_sigmoid_gating_delta_rule_update: ",
        name,
        " must be fp32");
}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.is_cuda(),
        "vllm_fused_sigmoid_gating_delta_rule_update: ",
        name,
        " must be a CUDA/HIP tensor");
}

} // namespace

std::vector<at::Tensor>
vllm_fused_sigmoid_gating_delta_rule_update(
    at::Tensor const &A_log,
    at::Tensor const &a,
    at::Tensor const &b,
    at::Tensor const &dt_bias,
    at::Tensor const &q,
    at::Tensor const &k,
    at::Tensor const &v,
    float beta,
    float threshold,
    std::optional<float> const &scale,
    std::optional<at::Tensor> const &initial_state_opt,
    bool inplace_final_state,
    std::optional<at::Tensor> const &cu_seqlens,
    std::optional<at::Tensor> const &ssm_state_indices,
    std::optional<at::Tensor> const &num_accepted_tokens,
    bool use_qk_l2norm_in_kernel,
    bool is_kda)
{
    check_cuda_tensor(A_log, "A_log");
    check_cuda_tensor(a, "a");
    check_cuda_tensor(b, "b");
    check_cuda_tensor(dt_bias, "dt_bias");
    check_cuda_tensor(q, "q");
    check_cuda_tensor(k, "k");
    check_cuda_tensor(v, "v");
    TORCH_CHECK(
        initial_state_opt.has_value(),
        "vllm_fused_sigmoid_gating_delta_rule_update: initial_state is required");
    torch::Tensor initial_state = *initial_state_opt;
    check_cuda_tensor(initial_state, "initial_state");

    TORCH_CHECK(
        !is_kda,
        "vllm_fused_sigmoid_gating_delta_rule_update: is_kda=True is not supported yet");
    TORCH_CHECK(
        inplace_final_state,
        "vllm_fused_sigmoid_gating_delta_rule_update: only inplace_final_state=True is supported");
    TORCH_CHECK(
        ssm_state_indices.has_value(),
        "vllm_fused_sigmoid_gating_delta_rule_update: ssm_state_indices is required");

    TORCH_CHECK(
        q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
        "vllm_fused_sigmoid_gating_delta_rule_update: q/k/v must be 4D");
    TORCH_CHECK(
        initial_state.dim() == 4,
        "vllm_fused_sigmoid_gating_delta_rule_update: initial_state must be 4D [slots, HV, V, K]");
    TORCH_CHECK(
        A_log.dim() == 1 && dt_bias.dim() == 1,
        "vllm_fused_sigmoid_gating_delta_rule_update: A_log and dt_bias must be 1D");
    TORCH_CHECK(
        a.dim() == 2 && b.dim() == 2,
        "vllm_fused_sigmoid_gating_delta_rule_update: only non-KDA a/b shapes [tokens, HV] are supported");

    TORCH_CHECK(
        q.scalar_type() == at::ScalarType::Half ||
            q.scalar_type() == at::ScalarType::BFloat16 ||
            q.scalar_type() == at::ScalarType::Float,
        "vllm_fused_sigmoid_gating_delta_rule_update: q must be fp16, bf16, or fp32");

    check_same_dtype(q, A_log, "A_log");
    check_same_dtype(q, a, "a");
    check_same_dtype(q, b, "b");
    check_same_dtype(q, dt_bias, "dt_bias");
    check_same_dtype(q, k, "k");
    check_same_dtype(q, v, "v");
    // State cache stays fp32 even when activations run in fp16/bf16.
    check_fp32_dtype(initial_state, "initial_state");

    const int64_t B_i64 = k.size(0);
    const int64_t T_i64 = k.size(1);
    const int64_t H_i64 = k.size(2);
    const int64_t K_i64 = k.size(3);
    const int64_t HV_i64 = v.size(2);
    const int64_t V_i64 = v.size(3);

    TORCH_CHECK(
        q.size(0) == B_i64 && q.size(1) == T_i64 &&
            q.size(2) == H_i64 && q.size(3) == K_i64,
        "vllm_fused_sigmoid_gating_delta_rule_update: q and k shapes must match");
    TORCH_CHECK(
        v.size(0) == B_i64 && v.size(1) == T_i64,
        "vllm_fused_sigmoid_gating_delta_rule_update: v batch/token dimensions must match k");
    TORCH_CHECK(
        K_i64 == kSupportedK,
        "vllm_fused_sigmoid_gating_delta_rule_update: only K=128 is supported");
    TORCH_CHECK(
        V_i64 == 128 || V_i64 == 256,
        "vllm_fused_sigmoid_gating_delta_rule_update: only V=128 or V=256 is supported");
    TORCH_CHECK(
        HV_i64 % H_i64 == 0,
        "vllm_fused_sigmoid_gating_delta_rule_update: HV must be divisible by H");
    TORCH_CHECK(
        A_log.size(0) == HV_i64 && dt_bias.size(0) == HV_i64 &&
            a.size(1) == HV_i64 && b.size(1) == HV_i64,
        "vllm_fused_sigmoid_gating_delta_rule_update: gating tensors must match HV");
    TORCH_CHECK(
        initial_state.size(1) == HV_i64 &&
            initial_state.size(2) == V_i64 &&
            initial_state.size(3) == K_i64,
        "vllm_fused_sigmoid_gating_delta_rule_update: initial_state shape must be [slots, HV, V, K]");
    TORCH_CHECK(
        B_i64 <= std::numeric_limits<int>::max() &&
            T_i64 <= std::numeric_limits<int>::max() &&
            H_i64 <= std::numeric_limits<int>::max() &&
            HV_i64 <= std::numeric_limits<int>::max() &&
            V_i64 <= std::numeric_limits<int>::max(),
        "vllm_fused_sigmoid_gating_delta_rule_update: tensor dimensions are too large");

    if (cu_seqlens.has_value()) {
        check_cuda_tensor(*cu_seqlens, "cu_seqlens");
        TORCH_CHECK(
            cu_seqlens->scalar_type() == at::ScalarType::Int,
            "vllm_fused_sigmoid_gating_delta_rule_update: cu_seqlens must be int32");
        TORCH_CHECK(
            q.size(0) == 1,
            "vllm_fused_sigmoid_gating_delta_rule_update: q.shape[0] must be 1 when cu_seqlens is provided");
    }

    check_cuda_tensor(*ssm_state_indices, "ssm_state_indices");
    TORCH_CHECK(
        ssm_state_indices->scalar_type() == at::ScalarType::Int,
        "vllm_fused_sigmoid_gating_delta_rule_update: ssm_state_indices must be int32");
    TORCH_CHECK(
        ssm_state_indices->dim() == 1 || ssm_state_indices->dim() == 2,
        "vllm_fused_sigmoid_gating_delta_rule_update: ssm_state_indices must be 1D or 2D");

    if (num_accepted_tokens.has_value()) {
        check_cuda_tensor(*num_accepted_tokens, "num_accepted_tokens");
        TORCH_CHECK(
            num_accepted_tokens->scalar_type() == at::ScalarType::Int ||
                num_accepted_tokens->scalar_type() == at::ScalarType::Long,
            "vllm_fused_sigmoid_gating_delta_rule_update: num_accepted_tokens must be int32 or int64");
    }

    auto A_log_contig = A_log.contiguous();
    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();
    auto dt_bias_contig = dt_bias.contiguous();
    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();
    auto indices_contig = ssm_state_indices->contiguous();
    auto cu_contig = cu_seqlens.has_value() ? cu_seqlens->contiguous() : torch::Tensor();
    auto accepted_contig = num_accepted_tokens.has_value()
        ? num_accepted_tokens->to(at::ScalarType::Int).contiguous()
        : torch::Tensor();

    const int B = static_cast<int>(B_i64);
    const int T_total = static_cast<int>(T_i64);
    const int H = static_cast<int>(H_i64);
    const int HV = static_cast<int>(HV_i64);
    const int V = static_cast<int>(V_i64);
    const bool is_varlen = cu_seqlens.has_value();
    const bool is_spec_decoding = num_accepted_tokens.has_value();
    const int N = is_varlen ? static_cast<int>(cu_contig.size(0) - 1) : B;
    const int stride_indices_seq =
        indices_contig.dim() == 2 ? static_cast<int>(indices_contig.stride(0)) : 1;
    const int64_t stride_state_slot = initial_state.stride(0);
    const int64_t stride_state_hv = initial_state.stride(1);
    const int64_t stride_state_v = initial_state.stride(2);
    const int64_t stride_state_k = initial_state.stride(3);
    const float scale_f =
        static_cast<float>(scale.has_value() ? *scale : std::pow(static_cast<double>(K_i64), -0.5));

    TORCH_CHECK(
        !is_spec_decoding || indices_contig.dim() == 2,
        "vllm_fused_sigmoid_gating_delta_rule_update: speculative decoding requires 2D ssm_state_indices");
    if (is_spec_decoding) {
        TORCH_CHECK(
            accepted_contig.dim() == 1,
            "vllm_fused_sigmoid_gating_delta_rule_update: num_accepted_tokens must be 1D");
        TORCH_CHECK(
            accepted_contig.size(0) >= N,
            "vllm_fused_sigmoid_gating_delta_rule_update: num_accepted_tokens length must cover number of sequences");
    }

    auto output = torch::empty_like(v_contig);
    if (output.numel() == 0) {
        return {output, initial_state};
    }

    auto stream = at::hip::getCurrentHIPStream();
    const char* variant_env = std::getenv("AITER_FSG_KERNEL_VARIANT");
    const int kernel_variant =
        variant_env != nullptr ? std::atoi(variant_env) : kDefaultKernelVariant;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q_contig.scalar_type(),
        "vllm_fused_sigmoid_gating_delta_rule_update",
        [&] {
            const dim3 block(kTileThreads);
            if (kernel_variant == 16) {
                const dim3 grid(
                    static_cast<unsigned int>(N),
                    static_cast<unsigned int>(HV),
                    static_cast<unsigned int>((V + 16 - 1) / 16));
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        fused_sigmoid_gating_delta_rule_tiled_kernel<scalar_t, 16, 16, kTileThreads>),
                    grid,
                    block,
                    0,
                    stream.stream(),
                    A_log_contig.data_ptr<scalar_t>(),
                    a_contig.data_ptr<scalar_t>(),
                    b_contig.data_ptr<scalar_t>(),
                    dt_bias_contig.data_ptr<scalar_t>(),
                    q_contig.data_ptr<scalar_t>(),
                    k_contig.data_ptr<scalar_t>(),
                    v_contig.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    initial_state.data_ptr<float>(),
                    stride_state_slot,
                    stride_state_hv,
                    stride_state_v,
                    stride_state_k,
                    is_varlen ? cu_contig.data_ptr<int32_t>() : nullptr,
                    indices_contig.data_ptr<int32_t>(),
                    is_spec_decoding ? accepted_contig.data_ptr<int32_t>() : nullptr,
                    static_cast<float>(beta),
                    static_cast<float>(threshold),
                    scale_f,
                    B,
                    T_total,
                    N,
                    H,
                    HV,
                    V,
                    stride_indices_seq,
                    is_varlen,
                    is_spec_decoding,
                    use_qk_l2norm_in_kernel);
            } else if (kernel_variant == 8) {
                const dim3 grid(
                    static_cast<unsigned int>(N),
                    static_cast<unsigned int>(HV),
                    static_cast<unsigned int>((V + 32 - 1) / 32));
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        fused_sigmoid_gating_delta_rule_tiled_kernel<scalar_t, 8, 32, kTileThreads>),
                    grid,
                    block,
                    0,
                    stream.stream(),
                    A_log_contig.data_ptr<scalar_t>(),
                    a_contig.data_ptr<scalar_t>(),
                    b_contig.data_ptr<scalar_t>(),
                    dt_bias_contig.data_ptr<scalar_t>(),
                    q_contig.data_ptr<scalar_t>(),
                    k_contig.data_ptr<scalar_t>(),
                    v_contig.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    initial_state.data_ptr<float>(),
                    stride_state_slot,
                    stride_state_hv,
                    stride_state_v,
                    stride_state_k,
                    is_varlen ? cu_contig.data_ptr<int32_t>() : nullptr,
                    indices_contig.data_ptr<int32_t>(),
                    is_spec_decoding ? accepted_contig.data_ptr<int32_t>() : nullptr,
                    static_cast<float>(beta),
                    static_cast<float>(threshold),
                    scale_f,
                    B,
                    T_total,
                    N,
                    H,
                    HV,
                    V,
                    stride_indices_seq,
                    is_varlen,
                    is_spec_decoding,
                    use_qk_l2norm_in_kernel);
            } else {
                const dim3 grid(
                    static_cast<unsigned int>(N),
                    static_cast<unsigned int>(HV),
                    static_cast<unsigned int>((V + 64 - 1) / 64));
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        fused_sigmoid_gating_delta_rule_tiled_kernel<scalar_t, 4, 64, kTileThreads>),
                    grid,
                    block,
                    0,
                    stream.stream(),
                    A_log_contig.data_ptr<scalar_t>(),
                    a_contig.data_ptr<scalar_t>(),
                    b_contig.data_ptr<scalar_t>(),
                    dt_bias_contig.data_ptr<scalar_t>(),
                    q_contig.data_ptr<scalar_t>(),
                    k_contig.data_ptr<scalar_t>(),
                    v_contig.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    initial_state.data_ptr<float>(),
                    stride_state_slot,
                    stride_state_hv,
                    stride_state_v,
                    stride_state_k,
                    is_varlen ? cu_contig.data_ptr<int32_t>() : nullptr,
                    indices_contig.data_ptr<int32_t>(),
                    is_spec_decoding ? accepted_contig.data_ptr<int32_t>() : nullptr,
                    static_cast<float>(beta),
                    static_cast<float>(threshold),
                    scale_f,
                    B,
                    T_total,
                    N,
                    H,
                    HV,
                    V,
                    stride_indices_seq,
                    is_varlen,
                    is_spec_decoding,
                    use_qk_l2norm_in_kernel);
            }
        });

    AITER_HIP_KERNEL_LAUNCH_CHECK();

    return {output, initial_state};
}
