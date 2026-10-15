// SPDX-License-Identifier: MIT

#include <ATen/Dispatch.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <type_traits>
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

typedef float f32x4 __attribute__((ext_vector_type(4)));

constexpr int kVec = 4;
constexpr int kMaxSupportedK = 128;
constexpr int kBlockThreads = 256;

template <typename T>
struct alignas(sizeof(T) * kVec) Vec4 {
    T v[kVec];
};

__device__ __forceinline__ float fast_exp(float x)
{
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
    constexpr float kLog2e = 1.44269504088896340736f;
    return __builtin_amdgcn_exp2f(x * kLog2e);
#else
    return __expf(x);
#endif
}

__device__ __forceinline__ float softplus_th(float x, float beta, float inv_beta, float th)
{
    const float bx = beta * x;
    return bx <= th ? __logf(1.0f + fast_exp(bx)) * inv_beta : x;
}

// exp(g) where g is the (possibly lower-bounded) sigmoid gate log-value.
// UseLowerBound mirrors triton's USE_LOWER_BOUND constexpr: when set, the KDA
// "safe gate" lower_bound * sigmoid(exp(A_log) * x) replaces the softplus gate
// -exp(A_log) * softplus(x). `A` is already exp(A_log) (computed once per hv).
template <bool UseLowerBound>
__device__ __forceinline__ float gate_exp(
    float x, float A, float lower_bound, float beta, float inv_beta, float th)
{
    if constexpr (UseLowerBound) {
        const float sig = 1.0f / (1.0f + fast_exp(-A * x));
        return fast_exp(lower_bound * sig);
    } else {
        return fast_exp(-A * softplus_th(x, beta, inv_beta, th));
    }
}

__device__ __forceinline__ f32x4 ld_nt(const f32x4* p)
{
    return __builtin_nontemporal_load(p);
}

__device__ __forceinline__ void st_nt(f32x4* p, f32x4 x)
{
    __builtin_nontemporal_store(x, p);
}

template <typename scalar_t>
struct GdnParams {
    // A_log/a/b/dt_bias are the sigmoid-gating inputs. They are converted to
    // fp32 host-side (same treatment as A_log) instead of being forced to
    // match q's dtype -- triton loads each of these independently and
    // upcasts to fp32 on load, with no dtype relationship to q/k/v.
    const float* __restrict__ A_log;
    const float* __restrict__ a;
    const float* __restrict__ b;
    const float* __restrict__ dt_bias;
    const scalar_t* __restrict__ q;
    const scalar_t* __restrict__ k;
    const scalar_t* __restrict__ v;
    scalar_t* __restrict__ o;

    // Recurrent state / intermediate-state cache. Element type is given by
    // state_dtype / cache_dtype (kDtypeF32 / kDtypeBF16 / kDtypeF16); strides
    // below are in elements of that type.
    void* __restrict__ state;
    void* __restrict__ cache;
    int state_dtype;
    int cache_dtype;

    const int32_t* __restrict__ cu_seqlens;
    const int32_t* __restrict__ state_idx;
    const int32_t* __restrict__ cache_idx;
    const int32_t* __restrict__ parent_tok;

    int64_t s_slot, s_hv, s_v;
    int64_t c_slot, c_step, c_hv, c_v;
    int64_t p_seq, p_tok;

    float sp_beta, sp_inv_beta, sp_th, scale;
    float lower_bound;
    int T_total;
    int H;
    int HV;
    int K;
    int V;
    int h_ratio;
    int write_state;
    int has_tree;
};

template <typename scalar_t, int NumChunk, int LPR, bool FullVec>
__device__ __forceinline__ void load_vec_frag(
    Vec4<scalar_t> (&dst)[NumChunk],
    const scalar_t* __restrict__ base,
    int lane,
    int K)
{
#pragma unroll
    for (int j = 0; j < NumChunk; ++j) {
        const int vi = j * LPR + lane;
        if constexpr (FullVec) {
            dst[j] = reinterpret_cast<const Vec4<scalar_t>*>(base)[vi];
        } else {
#pragma unroll
            for (int e = 0; e < kVec; ++e) {
                const int kk = vi * kVec + e;
                dst[j].v[e] = (kk < K) ? base[kk] : static_cast<scalar_t>(0);
            }
        }
    }
}

// State / intermediate-cache element types. The recurrent math always runs in
// fp32 (hs is f32x4); like triton, the buffers are loaded with .to(fp32) and
// stored back with .to(buffer dtype), so the buffers themselves may be fp32,
// bf16 or fp16. The dtype is a runtime code (not a template parameter) so the
// state/cache dtypes don't multiply with scalar_t in the instantiation count.
constexpr int kDtypeF32 = 0;
constexpr int kDtypeBF16 = 1;
constexpr int kDtypeF16 = 2;

template <typename T>
__device__ __forceinline__ f32x4 ld_state4(const T* p, bool nt)
{
    if constexpr (std::is_same<T, float>::value) {
        const f32x4* q = reinterpret_cast<const f32x4*>(p);
        return nt ? ld_nt(q) : *q;
    } else {
        const Vec4<T> raw = *reinterpret_cast<const Vec4<T>*>(p);
        f32x4 r;
#pragma unroll
        for (int e = 0; e < kVec; ++e) {
            r[e] = static_cast<float>(raw.v[e]);
        }
        return r;
    }
}

template <typename T>
__device__ __forceinline__ void st_state4(T* p, f32x4 x)
{
    if constexpr (std::is_same<T, float>::value) {
        st_nt(reinterpret_cast<f32x4*>(p), x);
    } else {
        Vec4<T> raw;
#pragma unroll
        for (int e = 0; e < kVec; ++e) {
            raw.v[e] = static_cast<T>(x[e]);
        }
        *reinterpret_cast<Vec4<T>*>(p) = raw;
    }
}

template <typename T, int NumChunk, int LPR, bool FullVec>
__device__ __forceinline__ void load_state_row_t(
    f32x4 (&dst)[NumChunk],
    const T* __restrict__ base,
    int lane,
    int K,
    bool nt)
{
#pragma unroll
    for (int j = 0; j < NumChunk; ++j) {
        const int vi = j * LPR + lane;
        if constexpr (FullVec) {
            dst[j] = ld_state4<T>(base + vi * kVec, nt);
        } else {
#pragma unroll
            for (int e = 0; e < kVec; ++e) {
                const int kk = vi * kVec + e;
                dst[j][e] = (kk < K) ? static_cast<float>(base[kk]) : 0.0f;
            }
        }
    }
}

template <typename T, int NumChunk, int LPR, bool FullVec>
__device__ __forceinline__ void store_state_row_t(
    const f32x4 (&src)[NumChunk],
    T* __restrict__ base,
    int lane,
    int K)
{
#pragma unroll
    for (int j = 0; j < NumChunk; ++j) {
        const int vi = j * LPR + lane;
        if constexpr (FullVec) {
            st_state4<T>(base + vi * kVec, src[j]);
        } else {
#pragma unroll
            for (int e = 0; e < kVec; ++e) {
                const int kk = vi * kVec + e;
                if (kk < K) {
                    base[kk] = static_cast<T>(src[j][e]);
                }
            }
        }
    }
}

// Runtime-dtype wrappers. `off` is an element offset into `base`.
template <int NumChunk, int LPR, bool FullVec>
__device__ __forceinline__ void load_state_row(
    f32x4 (&dst)[NumChunk],
    const void* __restrict__ base,
    int64_t off,
    int dtype,
    int lane,
    int K,
    bool nt)
{
    if (dtype == kDtypeF32) {
        load_state_row_t<float, NumChunk, LPR, FullVec>(
            dst, static_cast<const float*>(base) + off, lane, K, nt);
    } else if (dtype == kDtypeBF16) {
        load_state_row_t<c10::BFloat16, NumChunk, LPR, FullVec>(
            dst, static_cast<const c10::BFloat16*>(base) + off, lane, K, nt);
    } else {
        load_state_row_t<c10::Half, NumChunk, LPR, FullVec>(
            dst, static_cast<const c10::Half*>(base) + off, lane, K, nt);
    }
}

template <int NumChunk, int LPR, bool FullVec>
__device__ __forceinline__ void store_state_row(
    const f32x4 (&src)[NumChunk],
    void* __restrict__ base,
    int64_t off,
    int dtype,
    int lane,
    int K)
{
    if (dtype == kDtypeF32) {
        store_state_row_t<float, NumChunk, LPR, FullVec>(
            src, static_cast<float*>(base) + off, lane, K);
    } else if (dtype == kDtypeBF16) {
        store_state_row_t<c10::BFloat16, NumChunk, LPR, FullVec>(
            src, static_cast<c10::BFloat16*>(base) + off, lane, K);
    } else {
        store_state_row_t<c10::Half, NumChunk, LPR, FullVec>(
            src, static_cast<c10::Half*>(base) + off, lane, K);
    }
}

template <
    typename scalar_t,
    int NumChunk,
    int LPR,
    int VRows,
    int BlockThreads,
    bool FullVec,
    bool IsKda,
    bool HasCache,
    bool UseLowerBound>
__global__ __launch_bounds__(BlockThreads)
void sglang_fused_sigmoid_gating_delta_rule_update_kernel(GdnParams<scalar_t> p)
{
    constexpr int Groups = BlockThreads / LPR;
    constexpr int RowsPerBlock = Groups * VRows;

    static_assert(BlockThreads % LPR == 0, "block must split into lane groups");
    static_assert(LPR == 16 || LPR == 32, "LPR must be 16 or 32");
    static_assert(NumChunk == 1 || NumChunk == 2, "NumChunk must be 1 or 2");

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (LPR - 1);
    const int gid = tid / LPR;

    const int v0 = static_cast<int>(blockIdx.x) * RowsPerBlock + gid * VRows;
    if constexpr (!IsKda) {
        if (v0 >= p.V) {
            return;
        }
    }
    const int hv = static_cast<int>(blockIdx.y);
    const int n = static_cast<int>(blockIdx.z);

    const int K = FullVec ? (NumChunk * LPR * kVec) : p.K;

    int bos;
    int T_cur;
    if (p.cu_seqlens != nullptr) {
        bos = p.cu_seqlens[n];
        T_cur = p.cu_seqlens[n + 1] - bos;
    } else {
        bos = n * p.T_total;
        T_cur = p.T_total;
    }
    if (T_cur <= 0) {
        return;
    }

    const bool use_state = (p.state != nullptr);
    int slot = 0;
    if (use_state) {
        slot = p.state_idx[n];
        if (slot < 0) {
            return;
        }
    }

    int cslot = -1;
    if constexpr (HasCache) {
        cslot = p.cache_idx[n];
    }

    bool row_ok[VRows];
#pragma unroll
    for (int r = 0; r < VRows; ++r) {
        row_ok[r] = (!IsKda && VRows == 1) ? true : ((v0 + r) < p.V);
    }

    const int h = hv / p.h_ratio;

    f32x4 hs[VRows][NumChunk];
    const int64_t s_base =
        use_state ? (static_cast<int64_t>(slot) * p.s_slot +
                     static_cast<int64_t>(hv) * p.s_hv)
                  : 0;
#pragma unroll
    for (int r = 0; r < VRows; ++r) {
        if (use_state && row_ok[r]) {
            load_state_row<NumChunk, LPR, FullVec>(
                hs[r], p.state,
                s_base + static_cast<int64_t>(v0 + r) * p.s_v,
                p.state_dtype, lane, K, true);
        } else {
#pragma unroll
            for (int j = 0; j < NumChunk; ++j) {
                hs[r][j] = f32x4(0.0f);
            }
        }
    }

    const float A = fast_exp(p.A_log[hv]);

    __shared__ float s_gate[IsKda ? 2 * kMaxSupportedK : 1];
    const bool gate_lane = IsKda && (tid < kMaxSupportedK);
    int gbuf = 0;

    float dts = 0.0f;
    float dt_t = 0.0f;
    float a_cur = 0.0f;
    float a_nxt = 0.0f;
    if constexpr (IsKda) {
        if (gate_lane && tid < K) {
            dt_t = static_cast<float>(p.dt_bias[static_cast<int64_t>(hv) * K + tid]);
            a_cur = static_cast<float>(
                p.a[(static_cast<int64_t>(bos) * p.HV + hv) * K + tid]);
        }
    } else {
        dts = static_cast<float>(p.dt_bias[hv]);
    }

    const scalar_t* qp = p.q + (static_cast<int64_t>(bos) * p.H + h) * K;
    const scalar_t* kp = p.k + (static_cast<int64_t>(bos) * p.H + h) * K;
    const int64_t qk_step = static_cast<int64_t>(p.H) * K;

    const float* ap = p.a + static_cast<int64_t>(bos) * p.HV + hv;
    const float* bp = p.b + static_cast<int64_t>(bos) * p.HV + hv;
    const int64_t ab_step = p.HV;

    int64_t vo = (static_cast<int64_t>(bos) * p.HV + hv) * p.V + v0;
    const int64_t vo_step = static_cast<int64_t>(p.HV) * p.V;

    int64_t c_base = 0;
    int64_t c_cur = 0;
    if constexpr (HasCache) {
        if (cslot >= 0) {
            c_base = static_cast<int64_t>(cslot) * p.c_slot +
                     static_cast<int64_t>(hv) * p.c_hv;
            c_cur = c_base;
        }
    }

    Vec4<scalar_t> qc[NumChunk], kc[NumChunk], qx[NumChunk], kx[NumChunk];
    load_vec_frag<scalar_t, NumChunk, LPR, FullVec>(qc, qp, lane, K);
    load_vec_frag<scalar_t, NumChunk, LPR, FullVec>(kc, kp, lane, K);

    float ac = 0.0f;
    float bc = static_cast<float>(bp[0]);
    float an = 0.0f;
    float bn = 0.0f;
    if constexpr (!IsKda) {
        ac = static_cast<float>(ap[0]);
    }

    for (int t = 0; t < T_cur; ++t) {
        if constexpr (IsKda) {
            if (gate_lane) {
                s_gate[gbuf * kMaxSupportedK + tid] =
                    (tid < K) ? gate_exp<UseLowerBound>(
                                    a_cur + dt_t, A, p.lower_bound, p.sp_beta,
                                    p.sp_inv_beta, p.sp_th)
                              : 1.0f;
                if (t + 1 < T_cur && tid < K) {
                    a_nxt = static_cast<float>(
                        p.a[(static_cast<int64_t>(bos + t + 1) * p.HV + hv) * K + tid]);
                }
            }
            __syncthreads();
        }

        if constexpr (HasCache) {
            if (p.has_tree != 0 && t != 0 && cslot >= 0) {
                const int parent = p.parent_tok[static_cast<int64_t>(n) * p.p_seq +
                                                static_cast<int64_t>(t) * p.p_tok];
                if (parent != t - 1) {
                    const int64_t pb = c_base + static_cast<int64_t>(parent) * p.c_step;
#pragma unroll
                    for (int r = 0; r < VRows; ++r) {
                        if (row_ok[r]) {
                            load_state_row<NumChunk, LPR, FullVec>(
                                hs[r], p.cache,
                                pb + static_cast<int64_t>(v0 + r) * p.c_v,
                                p.cache_dtype, lane, K, false);
                        }
                    }
                }
            }
        }

        if (t + 1 < T_cur) {
            const int64_t off = static_cast<int64_t>(t + 1) * qk_step;
            load_vec_frag<scalar_t, NumChunk, LPR, FullVec>(qx, qp + off, lane, K);
            load_vec_frag<scalar_t, NumChunk, LPR, FullVec>(kx, kp + off, lane, K);
            bn = static_cast<float>(bp[static_cast<int64_t>(t + 1) * ab_step]);
            if constexpr (!IsKda) {
                an = static_cast<float>(ap[static_cast<int64_t>(t + 1) * ab_step]);
            }
        }

        const float beta_gate = 1.0f / (1.0f + fast_exp(-bc));
        float gate = 1.0f;
        if constexpr (!IsKda) {
            gate = gate_exp<UseLowerBound>(
                ac + dts, A, p.lower_bound, p.sp_beta, p.sp_inv_beta, p.sp_th);
        }

        float hk[VRows];
#pragma unroll
        for (int r = 0; r < VRows; ++r) {
            hk[r] = 0.0f;
        }
        float qsq = 0.0f;
        float ksq = 0.0f;

#pragma unroll
        for (int j = 0; j < NumChunk; ++j) {
            float gv[kVec];
            if constexpr (IsKda) {
                const int gb = gbuf * kMaxSupportedK + (j * LPR + lane) * kVec;
#pragma unroll
                for (int e = 0; e < kVec; ++e) {
                    gv[e] = s_gate[gb + e];
                }
            }
#pragma unroll
            for (int e = 0; e < kVec; ++e) {
                const float kv = static_cast<float>(kc[j].v[e]);
                const float qv = static_cast<float>(qc[j].v[e]);
                const float g = IsKda ? gv[e] : gate;
                qsq += qv * qv;
                ksq += kv * kv;
#pragma unroll
                for (int r = 0; r < VRows; ++r) {
                    const float hg = hs[r][j][e] * g;
                    hs[r][j][e] = hg;
                    hk[r] += hg * kv;
                }
            }
        }

#pragma unroll
        for (int m = LPR / 2; m > 0; m >>= 1) {
#pragma unroll
            for (int r = 0; r < VRows; ++r) {
                hk[r] += __shfl_xor(hk[r], m, LPR);
            }
            qsq += __shfl_xor(qsq, m, LPR);
            ksq += __shfl_xor(ksq, m, LPR);
        }

        const float inv_q = (p.scale < 0.0f) ? 1.0f : rsqrtf(qsq + 1.0e-6f);
        const float inv_k = (p.scale < 0.0f) ? 1.0f : rsqrtf(ksq + 1.0e-6f);
        const float abs_scale = (p.scale < 0.0f) ? -p.scale : p.scale;

        float coef[VRows];
#pragma unroll
        for (int r = 0; r < VRows; ++r) {
            const float vv = row_ok[r] ? static_cast<float>(p.v[vo + r]) : 0.0f;
            coef[r] = (vv - hk[r] * inv_k) * beta_gate * inv_k;
        }

        float out[VRows];
#pragma unroll
        for (int r = 0; r < VRows; ++r) {
            out[r] = 0.0f;
        }

#pragma unroll
        for (int j = 0; j < NumChunk; ++j) {
#pragma unroll
            for (int e = 0; e < kVec; ++e) {
                const float kv = static_cast<float>(kc[j].v[e]);
                const float qv = static_cast<float>(qc[j].v[e]);
#pragma unroll
                for (int r = 0; r < VRows; ++r) {
                    const float hn = hs[r][j][e] + coef[r] * kv;
                    hs[r][j][e] = hn;
                    out[r] += hn * qv;
                }
            }
        }

#pragma unroll
        for (int m = LPR / 2; m > 0; m >>= 1) {
#pragma unroll
            for (int r = 0; r < VRows; ++r) {
                out[r] += __shfl_xor(out[r], m, LPR);
            }
        }

        if (lane == 0) {
            const float os = inv_q * abs_scale;
#pragma unroll
            for (int r = 0; r < VRows; ++r) {
                if (row_ok[r]) {
                    p.o[vo + r] = static_cast<scalar_t>(out[r] * os);
                }
            }
        }

        if constexpr (HasCache) {
            if (cslot >= 0) {
#pragma unroll
                for (int r = 0; r < VRows; ++r) {
                    if (row_ok[r]) {
                        store_state_row<NumChunk, LPR, FullVec>(
                            hs[r], p.cache,
                            c_cur + static_cast<int64_t>(v0 + r) * p.c_v,
                            p.cache_dtype, lane, K);
                    }
                }
                c_cur += p.c_step;
            }
        }

#pragma unroll
        for (int j = 0; j < NumChunk; ++j) {
            qc[j] = qx[j];
            kc[j] = kx[j];
        }
        ac = an;
        bc = bn;
        vo += vo_step;
        if constexpr (IsKda) {
            a_cur = a_nxt;
            gbuf ^= 1;
        }
    }

    if (use_state && p.write_state != 0) {
#pragma unroll
        for (int r = 0; r < VRows; ++r) {
            if (row_ok[r]) {
                store_state_row<NumChunk, LPR, FullVec>(
                    hs[r], p.state,
                    s_base + static_cast<int64_t>(v0 + r) * p.s_v,
                    p.state_dtype, lane, K);
            }
        }
    }
}

int gdn_num_cu()
{
    static const int cached = [] {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) {
            return 64;
        }
        int cu = 0;
        if (hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev) !=
            hipSuccess) {
            return 64;
        }
        return cu > 0 ? cu : 64;
    }();
    return cached;
}

int gdn_env_vrows()
{
    static const int cached = [] {
        const char* e = std::getenv("SGLANG_GDN_VROWS");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    return cached;
}

int pick_vrows(int N, int HV, int V, int LPR)
{
    const int groups = kBlockThreads / LPR;
    const long target = 4L * gdn_num_cu();
    const int candidates[3] = {4, 2, 1};
    for (int i = 0; i < 3; ++i) {
        const int vr = candidates[i];
        const int rows_per_block = groups * vr;
        const int nvt = (V + rows_per_block - 1) / rows_per_block;
        const long ctas = static_cast<long>(N) * HV * nvt;
        if (ctas >= target) {
            return vr;
        }
    }
    return 1;
}

template <
    typename scalar_t,
    int NumChunk,
    int LPR,
    int VRows,
    bool FullVec,
    bool IsKda,
    bool HasCache,
    bool UseLowerBound>
void launch_leaf(const GdnParams<scalar_t>& p, int N, hipStream_t stream)
{
    constexpr int Groups = kBlockThreads / LPR;
    constexpr int RowsPerBlock = Groups * VRows;
    const dim3 grid(static_cast<unsigned>((p.V + RowsPerBlock - 1) / RowsPerBlock),
                    static_cast<unsigned>(p.HV),
                    static_cast<unsigned>(N));
    sglang_fused_sigmoid_gating_delta_rule_update_kernel<
        scalar_t, NumChunk, LPR, VRows, kBlockThreads, FullVec, IsKda, HasCache, UseLowerBound>
        <<<grid, dim3(kBlockThreads), 0, stream>>>(p);
}

template <typename scalar_t, int NumChunk, int LPR, bool FullVec, bool IsKda, bool HasCache, bool UseLowerBound>
void launch_vrows(const GdnParams<scalar_t>& p, int N, int vrows, hipStream_t stream)
{
    switch (vrows) {
    case 4:
        launch_leaf<scalar_t, NumChunk, LPR, 4, FullVec, IsKda, HasCache, UseLowerBound>(p, N, stream);
        return;
    case 2:
        launch_leaf<scalar_t, NumChunk, LPR, 2, FullVec, IsKda, HasCache, UseLowerBound>(p, N, stream);
        return;
    default:
        launch_leaf<scalar_t, NumChunk, LPR, 1, FullVec, IsKda, HasCache, UseLowerBound>(p, N, stream);
        return;
    }
}

template <typename scalar_t, int NumChunk, int LPR, bool FullVec>
void launch_flags(const GdnParams<scalar_t>& p, int N, int vrows,
                  bool is_kda, bool has_cache, bool use_lower_bound, hipStream_t stream)
{
    if (is_kda) {
        if (has_cache) {
            if (use_lower_bound) {
                launch_vrows<scalar_t, NumChunk, LPR, FullVec, true, true, true>(p, N, vrows, stream);
            } else {
                launch_vrows<scalar_t, NumChunk, LPR, FullVec, true, true, false>(p, N, vrows, stream);
            }
        } else {
            if (use_lower_bound) {
                launch_vrows<scalar_t, NumChunk, LPR, FullVec, true, false, true>(p, N, vrows, stream);
            } else {
                launch_vrows<scalar_t, NumChunk, LPR, FullVec, true, false, false>(p, N, vrows, stream);
            }
        }
    } else {
        if (has_cache) {
            if (use_lower_bound) {
                launch_vrows<scalar_t, NumChunk, LPR, FullVec, false, true, true>(p, N, vrows, stream);
            } else {
                launch_vrows<scalar_t, NumChunk, LPR, FullVec, false, true, false>(p, N, vrows, stream);
            }
        } else {
            if (use_lower_bound) {
                launch_vrows<scalar_t, NumChunk, LPR, FullVec, false, false, true>(p, N, vrows, stream);
            } else {
                launch_vrows<scalar_t, NumChunk, LPR, FullVec, false, false, false>(p, N, vrows, stream);
            }
        }
    }
}

inline int state_elem_size(int dtype)
{
    return dtype == kDtypeF32 ? 4 : 2;
}

// The vector path loads/stores kVec elements at a time, so the base pointer
// must be aligned to kVec * sizeof(elem) (16B for fp32, 8B for bf16/fp16) and
// every stride must be a multiple of kVec elements.
template <typename scalar_t>
bool gdn_can_vectorize(const GdnParams<scalar_t>& p, bool has_cache)
{
    if ((p.K % (16 * kVec)) != 0) {
        return false;
    }
    if (p.state != nullptr) {
        if ((p.s_slot & 3) || (p.s_hv & 3) || (p.s_v & 3)) {
            return false;
        }
        const uintptr_t align = kVec * state_elem_size(p.state_dtype);
        if (reinterpret_cast<uintptr_t>(p.state) & (align - 1)) {
            return false;
        }
    }
    if (has_cache) {
        if ((p.c_slot & 3) || (p.c_step & 3) || (p.c_hv & 3) || (p.c_v & 3)) {
            return false;
        }
        const uintptr_t align = kVec * state_elem_size(p.cache_dtype);
        if (reinterpret_cast<uintptr_t>(p.cache) & (align - 1)) {
            return false;
        }
    }
    return true;
}

template <typename scalar_t>
void gdn_dispatch(const GdnParams<scalar_t>& p, int N, bool is_kda, bool has_cache,
                  bool use_lower_bound, int vrows_req, hipStream_t stream)
{
    if (gdn_can_vectorize(p, has_cache)) {
        const int vrows = vrows_req > 0 ? vrows_req : pick_vrows(N, p.HV, p.V, 16);
        if (p.K / (16 * kVec) == 2) {
            launch_flags<scalar_t, 2, 16, true>(p, N, vrows, is_kda, has_cache, use_lower_bound, stream);
        } else {
            launch_flags<scalar_t, 1, 16, true>(p, N, vrows, is_kda, has_cache, use_lower_bound, stream);
        }
        return;
    }
    const int vrows = vrows_req > 0 ? vrows_req : pick_vrows(N, p.HV, p.V, 32);
    launch_flags<scalar_t, 1, 32, false>(p, N, vrows, is_kda, has_cache, use_lower_bound, stream);
}

void check_cuda_tensor_sglang(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.is_cuda(),
        "sglang_fused_sigmoid_gating_delta_rule_update: ",
        name,
        " must be a CUDA/HIP tensor");
}

void check_same_dtype_sglang(
    const torch::Tensor& ref,
    const torch::Tensor& tensor,
    const char* name)
{
    TORCH_CHECK(
        tensor.scalar_type() == ref.scalar_type(),
        "sglang_fused_sigmoid_gating_delta_rule_update: ",
        name,
        " must have the same dtype as q");
}

void check_fp32_dtype_sglang(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "sglang_fused_sigmoid_gating_delta_rule_update: ",
        name,
        " must be fp32");
}

int state_dtype_sglang(const torch::Tensor& tensor, const char* name)
{
    switch (tensor.scalar_type()) {
    case at::ScalarType::Float:
        return kDtypeF32;
    case at::ScalarType::BFloat16:
        return kDtypeBF16;
    case at::ScalarType::Half:
        return kDtypeF16;
    default:
        TORCH_CHECK(
            false,
            "sglang_fused_sigmoid_gating_delta_rule_update: ",
            name,
            " must be fp32, bf16, or fp16");
        return kDtypeF32;
    }
}

}  // namespace

std::vector<at::Tensor>
sglang_fused_sigmoid_gating_delta_rule_update(
    at::Tensor const &A_log,
    at::Tensor const &a,
    at::Tensor const &dt_bias,
    float softplus_beta,
    float softplus_threshold_value,
    at::Tensor const &q,
    at::Tensor const &k,
    at::Tensor const &v,
    at::Tensor const &b,
    std::optional<at::Tensor> const &initial_state_source_opt,
    std::optional<at::Tensor> const &initial_state_indices_opt,
    std::optional<float> const &scale,
    bool use_qk_l2norm_in_kernel,
    std::optional<at::Tensor> const &cu_seqlens,
    bool is_kda,
    std::optional<float> const &lower_bound,
    bool disable_state_update,
    std::optional<at::Tensor> const &intermediate_states_buffer,
    std::optional<at::Tensor> const &intermediate_state_indices,
    int64_t cache_steps,
    std::optional<at::Tensor> const &retrieve_parent_token)
{
    check_cuda_tensor_sglang(A_log, "A_log");
    check_cuda_tensor_sglang(a, "a");
    check_cuda_tensor_sglang(dt_bias, "dt_bias");
    check_cuda_tensor_sglang(q, "q");
    check_cuda_tensor_sglang(k, "k");
    check_cuda_tensor_sglang(v, "v");
    check_cuda_tensor_sglang(b, "b");

    TORCH_CHECK(
        q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
        "sglang_fused_sigmoid_gating_delta_rule_update: q/k/v must be 4D");
    TORCH_CHECK(
        q.scalar_type() == at::ScalarType::Half ||
            q.scalar_type() == at::ScalarType::BFloat16 ||
            q.scalar_type() == at::ScalarType::Float,
        "sglang_fused_sigmoid_gating_delta_rule_update: q must be fp16, bf16, or fp32");
    check_same_dtype_sglang(q, k, "k");
    check_same_dtype_sglang(q, v, "v");

    TORCH_CHECK(
        softplus_beta > 0.0f,
        "sglang_fused_sigmoid_gating_delta_rule_update: softplus_beta must be positive");

    const int64_t B_i64 = k.size(0);
    const int64_t T_i64 = k.size(1);
    const int64_t H_i64 = k.size(2);
    const int64_t K_i64 = k.size(3);
    const int64_t HV_i64 = v.size(2);
    const int64_t V_i64 = v.size(3);

    TORCH_CHECK(
        q.size(0) == B_i64 && q.size(1) == T_i64 &&
            q.size(2) == H_i64 && q.size(3) == K_i64,
        "sglang_fused_sigmoid_gating_delta_rule_update: q and k shapes must match");
    TORCH_CHECK(
        v.size(0) == B_i64 && v.size(1) == T_i64,
        "sglang_fused_sigmoid_gating_delta_rule_update: v batch/token dimensions must match k");
    TORCH_CHECK(
        HV_i64 % H_i64 == 0,
        "sglang_fused_sigmoid_gating_delta_rule_update: HV must be divisible by H");
    TORCH_CHECK(
        K_i64 > 0 && K_i64 <= kMaxSupportedK,
        "sglang_fused_sigmoid_gating_delta_rule_update: only 1 <= K <= 128 is supported");
    TORCH_CHECK(
        A_log.numel() >= HV_i64,
        "sglang_fused_sigmoid_gating_delta_rule_update: A_log must cover HV");
    TORCH_CHECK(
        B_i64 <= std::numeric_limits<int>::max() &&
            T_i64 <= std::numeric_limits<int>::max() &&
            H_i64 <= std::numeric_limits<int>::max() &&
            HV_i64 <= std::numeric_limits<int>::max() &&
            K_i64 <= std::numeric_limits<int>::max() &&
            V_i64 <= std::numeric_limits<int>::max(),
        "sglang_fused_sigmoid_gating_delta_rule_update: tensor dimensions are too large");

    const int64_t total_tokens = B_i64 * T_i64;
    if (is_kda) {
        TORCH_CHECK(
            a.numel() >= total_tokens * HV_i64 * K_i64,
            "sglang_fused_sigmoid_gating_delta_rule_update: KDA a must cover [tokens, HV, K]");
        TORCH_CHECK(
            dt_bias.numel() >= HV_i64 * K_i64,
            "sglang_fused_sigmoid_gating_delta_rule_update: KDA dt_bias must cover [HV, K]");
    } else {
        TORCH_CHECK(
            a.numel() >= total_tokens * HV_i64,
            "sglang_fused_sigmoid_gating_delta_rule_update: a must cover [tokens, HV]");
        TORCH_CHECK(
            dt_bias.numel() >= HV_i64,
            "sglang_fused_sigmoid_gating_delta_rule_update: dt_bias must cover [HV]");
    }
    TORCH_CHECK(
        b.numel() >= total_tokens * HV_i64,
        "sglang_fused_sigmoid_gating_delta_rule_update: b must cover [tokens, HV]");

    int state_dtype = kDtypeF32;
    int cache_dtype = kDtypeF32;

    const bool use_initial_state = initial_state_source_opt.has_value();
    torch::Tensor initial_state;
    torch::Tensor state_indices;
    if (use_initial_state) {
        initial_state = *initial_state_source_opt;
        check_cuda_tensor_sglang(initial_state, "initial_state_source");
        state_dtype = state_dtype_sglang(initial_state, "initial_state_source");
        TORCH_CHECK(
            initial_state.dim() == 4 &&
                initial_state.size(1) == HV_i64 &&
                initial_state.size(2) == V_i64 &&
                initial_state.size(3) == K_i64,
            "sglang_fused_sigmoid_gating_delta_rule_update: initial_state_source must be [slots, HV, V, K]");
        TORCH_CHECK(
            initial_state.stride(3) == 1,
            "sglang_fused_sigmoid_gating_delta_rule_update: initial_state_source must be K-contiguous");
        TORCH_CHECK(
            initial_state_indices_opt.has_value(),
            "sglang_fused_sigmoid_gating_delta_rule_update: initial_state_indices is required when initial_state_source is provided");
        state_indices = *initial_state_indices_opt;
        check_cuda_tensor_sglang(state_indices, "initial_state_indices");
        TORCH_CHECK(
            (state_indices.scalar_type() == at::ScalarType::Int ||
             state_indices.scalar_type() == at::ScalarType::Long) &&
                state_indices.dim() == 1,
            "sglang_fused_sigmoid_gating_delta_rule_update: initial_state_indices must be 1D int32 or int64");
        state_indices = state_indices.to(at::ScalarType::Int).contiguous();
    }

    torch::Tensor cu_contig;
    if (cu_seqlens.has_value()) {
        check_cuda_tensor_sglang(*cu_seqlens, "cu_seqlens");
        TORCH_CHECK(
            cu_seqlens->scalar_type() == at::ScalarType::Int ||
                cu_seqlens->scalar_type() == at::ScalarType::Long,
            "sglang_fused_sigmoid_gating_delta_rule_update: cu_seqlens must be int32 or int64");
        TORCH_CHECK(
            q.size(0) == 1,
            "sglang_fused_sigmoid_gating_delta_rule_update: q.shape[0] must be 1 when cu_seqlens is provided");
        cu_contig = cu_seqlens->to(at::ScalarType::Int).contiguous();
    }
    const bool is_varlen = cu_seqlens.has_value();
    const int N = is_varlen ? static_cast<int>(cu_contig.size(0) - 1)
                            : static_cast<int>(B_i64);
    if (use_initial_state) {
        TORCH_CHECK(
            state_indices.size(0) >= N,
            "sglang_fused_sigmoid_gating_delta_rule_update: initial_state_indices length must cover N");
    }

    const bool cache_intermediate = intermediate_states_buffer.has_value();
    torch::Tensor intermediate;
    torch::Tensor intermediate_indices;
    if (cache_intermediate) {
        intermediate = *intermediate_states_buffer;
        check_cuda_tensor_sglang(intermediate, "intermediate_states_buffer");
        cache_dtype = state_dtype_sglang(intermediate, "intermediate_states_buffer");
        TORCH_CHECK(
            intermediate.dim() == 5 &&
                intermediate.size(1) >= cache_steps &&
                intermediate.size(2) == HV_i64 &&
                intermediate.size(3) == V_i64 &&
                intermediate.size(4) == K_i64,
            "sglang_fused_sigmoid_gating_delta_rule_update: intermediate_states_buffer must be [slots, cache_steps, HV, V, K]");
        TORCH_CHECK(
            intermediate.stride(4) == 1,
            "sglang_fused_sigmoid_gating_delta_rule_update: intermediate_states_buffer must be K-contiguous");
        TORCH_CHECK(
            intermediate_state_indices.has_value(),
            "sglang_fused_sigmoid_gating_delta_rule_update: intermediate_state_indices is required with intermediate_states_buffer");
        TORCH_CHECK(
            cache_steps > 0,
            "sglang_fused_sigmoid_gating_delta_rule_update: cache_steps must be positive with intermediate_states_buffer");
        intermediate_indices = *intermediate_state_indices;
        check_cuda_tensor_sglang(intermediate_indices, "intermediate_state_indices");
        TORCH_CHECK(
            (intermediate_indices.scalar_type() == at::ScalarType::Int ||
             intermediate_indices.scalar_type() == at::ScalarType::Long) &&
                intermediate_indices.dim() == 1 &&
                intermediate_indices.size(0) >= N,
            "sglang_fused_sigmoid_gating_delta_rule_update: intermediate_state_indices must be 1D int32 or int64 and cover N");
        intermediate_indices = intermediate_indices.to(at::ScalarType::Int).contiguous();
    }

    torch::Tensor retrieve_parent;
    if (retrieve_parent_token.has_value()) {
        check_cuda_tensor_sglang(*retrieve_parent_token, "retrieve_parent_token");
        TORCH_CHECK(
            cache_intermediate,
            "sglang_fused_sigmoid_gating_delta_rule_update: retrieve_parent_token requires intermediate_states_buffer");
        TORCH_CHECK(
            (retrieve_parent_token->scalar_type() == at::ScalarType::Int ||
             retrieve_parent_token->scalar_type() == at::ScalarType::Long) &&
                retrieve_parent_token->dim() == 2 &&
                retrieve_parent_token->size(0) >= N &&
                retrieve_parent_token->size(1) >= cache_steps,
            "sglang_fused_sigmoid_gating_delta_rule_update: retrieve_parent_token must be 2D int32 or int64 and cover [N, cache_steps]");
        retrieve_parent = retrieve_parent_token->to(at::ScalarType::Int).contiguous();
    }

    auto A_log_contig = A_log.to(at::ScalarType::Float).contiguous();
    // a/b/dt_bias: upcast to fp32 regardless of input dtype, same treatment
    // as A_log -- these are read-only per-step gating inputs (not updated
    // in place), so a dtype conversion here carries no write-back concerns.
    auto a_contig = a.to(at::ScalarType::Float).contiguous();
    auto dt_bias_contig = dt_bias.to(at::ScalarType::Float).contiguous();
    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();
    auto b_contig = b.to(at::ScalarType::Float).contiguous();

    const float scale_f = static_cast<float>(
        scale.has_value() ? *scale : std::pow(static_cast<double>(K_i64), -0.5));
    TORCH_CHECK(
        scale_f > 0.0f,
        "sglang_fused_sigmoid_gating_delta_rule_update: scale must be positive");

    const bool use_lower_bound = lower_bound.has_value();

    auto output = torch::empty_like(v_contig);
    if (output.numel() == 0) {
        return {output};
    }

    int vrows_req = 0;
    if (gdn_env_vrows() > 0) {
        vrows_req = gdn_env_vrows();
    }

    auto stream = at::hip::getCurrentHIPStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q_contig.scalar_type(),
        "sglang_fused_sigmoid_gating_delta_rule_update",
        [&] {
            GdnParams<scalar_t> p{};
            p.A_log = A_log_contig.data_ptr<float>();
            p.a = a_contig.data_ptr<float>();
            p.b = b_contig.data_ptr<float>();
            p.dt_bias = dt_bias_contig.data_ptr<float>();
            p.q = q_contig.data_ptr<scalar_t>();
            p.k = k_contig.data_ptr<scalar_t>();
            p.v = v_contig.data_ptr<scalar_t>();
            p.o = output.data_ptr<scalar_t>();

            p.state = use_initial_state ? initial_state.data_ptr() : nullptr;
            p.cache = cache_intermediate ? intermediate.data_ptr() : nullptr;
            p.state_dtype = state_dtype;
            p.cache_dtype = cache_dtype;

            p.cu_seqlens = is_varlen ? cu_contig.data_ptr<int32_t>() : nullptr;
            p.state_idx =
                use_initial_state ? state_indices.data_ptr<int32_t>() : nullptr;
            p.cache_idx = cache_intermediate
                              ? intermediate_indices.data_ptr<int32_t>()
                              : nullptr;
            p.parent_tok = retrieve_parent_token.has_value()
                               ? retrieve_parent.data_ptr<int32_t>()
                               : nullptr;

            p.s_slot = use_initial_state ? initial_state.stride(0) : 0;
            p.s_hv = use_initial_state ? initial_state.stride(1) : 0;
            p.s_v = use_initial_state ? initial_state.stride(2) : 0;
            p.c_slot = cache_intermediate ? intermediate.stride(0) : 0;
            p.c_step = cache_intermediate ? intermediate.stride(1) : 0;
            p.c_hv = cache_intermediate ? intermediate.stride(2) : 0;
            p.c_v = cache_intermediate ? intermediate.stride(3) : 0;
            p.p_seq = retrieve_parent_token.has_value() ? retrieve_parent.stride(0) : 0;
            p.p_tok = retrieve_parent_token.has_value() ? retrieve_parent.stride(1) : 0;

            p.sp_beta = softplus_beta;
            p.sp_inv_beta = 1.0f / softplus_beta;
            p.sp_th = softplus_threshold_value;
            p.scale = use_qk_l2norm_in_kernel ? scale_f : -scale_f;
            p.lower_bound = use_lower_bound ? *lower_bound : 0.0f;;
            p.T_total = static_cast<int>(T_i64);
            p.H = static_cast<int>(H_i64);
            p.HV = static_cast<int>(HV_i64);
            p.K = static_cast<int>(K_i64);
            p.V = static_cast<int>(V_i64);
            p.h_ratio = static_cast<int>(HV_i64 / H_i64);
            p.write_state = disable_state_update ? 0 : 1;
            p.has_tree = retrieve_parent_token.has_value() ? 1 : 0;

            gdn_dispatch<scalar_t>(p, N, is_kda, cache_intermediate, use_lower_bound,
                                   vrows_req, stream.stream());
        });

    AITER_HIP_KERNEL_LAUNCH_CHECK();
    return {output};
}
