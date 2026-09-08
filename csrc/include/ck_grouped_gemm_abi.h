// SPDX-License-Identifier: MIT
// Pure-C public header for the CK grouped-GEMM C ABI.
//
// This header is intentionally free of C++, CK template, and torch/extension.h
// dependencies so that external projects can include it without pulling in the
// full CK or PyTorch headers.
//
// The same struct and enumerators are also defined inside
// 3rdparty/composable_kernel/example_hcu/ck_tile/19_grouped_gemm/grouped_gemm.hpp
// (inside an extern "C" block).  When both headers are visible in a single
// translation unit the guard macro CK_GROUPED_GEMM_ABI_DEFINED prevents
// duplicate definitions.

#pragma once

#ifndef CK_GROUPED_GEMM_ABI_DEFINED
#define CK_GROUPED_GEMM_ABI_DEFINED

#include <stddef.h>

#ifdef __cplusplus
#include <hip/hip_runtime.h>
extern "C" {
#else
#include <hip/hip_runtime.h>
#endif

enum ck_tile_hcu_grouped_gemm_dtype
{
    CK_TILE_HCU_GROUPED_GEMM_FP16 = 0,
    CK_TILE_HCU_GROUPED_GEMM_FP8  = 1,
    CK_TILE_HCU_GROUPED_GEMM_INT8 = 2,
    CK_TILE_HCU_GROUPED_GEMM_BF8  = 3,
    CK_TILE_HCU_GROUPED_GEMM_BF16 = 4,
    CK_TILE_HCU_GROUPED_GEMM_INT4 = 5
};

// Per-group descriptor passed to ck_tile_hcu_grouped_gemm_run.
//
// Memory layout convention:
//   A: [M, K]  row-major  (stride_A = K)
//   B: [N, K]  row-major stored, but interpreted as column-major by the kernel
//              when b_layout='C' (stride_B = K), yielding C = A @ B^T
//   C: [M, N]  row-major  (stride_C = N)
//
// d_ptrs / stride_Ds are optional bias tensors (set num_d_tensors=0 and both
// pointers to NULL for a plain GEMM).
struct ck_tile_hcu_grouped_gemm_desc
{
    const void* a_ptr;
    const void* b_ptr;
    void*       c_ptr;
    int         k_batch;    // SplitK factor; use 1 for standard GEMM
    int         M;
    int         N;
    int         K;
    int         stride_A;
    int         stride_B;
    int         stride_C;
    int         num_d_tensors;
    const void* const* d_ptrs;
    const int*  stride_Ds;
};

// Returns the number of bytes of device workspace required for group_count groups.
size_t ck_tile_hcu_grouped_gemm_workspace_size(int group_count, int num_d_tensors);

// Launches the grouped GEMM kernel.
//
// descs       - array of group_count descriptors (device-visible pointers inside)
// group_count - number of GEMM groups
// dtype       - element dtype (ck_tile_hcu_grouped_gemm_dtype)
// a_layout    - 'R' (row-major) or 'C' (column-major) for A
// b_layout    - 'R' or 'C' for B; use 'C' when B is stored as [N,K] row-major
// workspace   - device buffer of at least ck_tile_hcu_grouped_gemm_workspace_size bytes
// stream      - HIP stream to submit the kernel on
//
// Returns 0 on success, negative on error.
int ck_tile_hcu_grouped_gemm_run(const struct ck_tile_hcu_grouped_gemm_desc* descs,
                                 int        group_count,
                                 int        dtype,
                                 char       a_layout,
                                 char       b_layout,
                                 void*      workspace,
                                 hipStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CK_GROUPED_GEMM_ABI_DEFINED
