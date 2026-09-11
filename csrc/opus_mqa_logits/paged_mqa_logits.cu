// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "paged_mqa_logits.h"
#include "gfx938/paged_mqa_logits.cuh"
#include "gfx946/paged_mqa_logits.cuh"
#include <climits>
#include <cstring>

void paged_mqa_logits_opus(const torch::Tensor& q, const torch::Tensor& cache,
                          const torch::Tensor& weights, const torch::Tensor& context,
                          const torch::Tensor& tables, torch::Tensor& output,
                          int64_t max_len, int64_t kernel_id)
{
    TORCH_CHECK(q.is_cuda(), "paged_mqa_logits: q must be on GPU");
    for(const auto& t : {q,cache,weights,context,tables,output}) {
        TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.is_contiguous(),
                    "paged_mqa_logits: tensors must be contiguous on the same GPU");
    }
    TORCH_CHECK(q.dim()==4 && q.scalar_type()==at::ScalarType::Float8_e4m3fn,
                "paged_mqa_logits: q must be E4M3FN [B,R,H,128]");
    const auto batch=q.size(0), next=q.size(1), heads=q.size(2);
    TORCH_CHECK(batch>0 && (next==1 || next==2 || next==4) &&
                (heads==32 || heads==64) && q.size(3)==128 && batch*next<=65535,
                "paged_mqa_logits: requires B>0, B*R<=65535, R=1/2/4, H=32/64, D=128");
    TORCH_CHECK(cache.scalar_type()==at::kByte && cache.dim()==4 && cache.size(0)>0 &&
                cache.size(0)<=INT_MAX/132 && cache.size(1)==1 && cache.size(2)==1 && cache.size(3)==132,
                "paged_mqa_logits: cache must be uint8 [P,1,1,132], P*132<=INT_MAX");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr())%4==0 &&
                reinterpret_cast<uintptr_t>(cache.data_ptr())%4==0,
                "paged_mqa_logits: Q/cache storage must be 4-byte aligned");
    TORCH_CHECK(weights.scalar_type()==at::kFloat && weights.dim()==2 &&
                weights.size(0)==batch*next && weights.size(1)==heads,
                "paged_mqa_logits: weights must be FP32 [B*R,H]");
    TORCH_CHECK(context.scalar_type()==at::kInt && context.dim()==1 && context.size(0)==batch,
                "paged_mqa_logits: context must be int32 [B]");
    TORCH_CHECK(tables.scalar_type()==at::kInt && tables.dim()==2 && tables.size(0)==batch &&
                tables.size(1)>0 && tables.size(1)<=INT_MAX,
                "paged_mqa_logits: tables must be int32 [B,T]");
    TORCH_CHECK(max_len>0 && max_len<=INT_MAX && output.scalar_type()==at::kFloat &&
                output.dim()==2 && output.size(0)==batch*next && output.size(1)==max_len,
                "paged_mqa_logits: output must be FP32 [B*R,max_len], max_len>0");
    for(const auto& t : {q,cache,weights,context,tables})
        TORCH_CHECK(!output.is_alias_of(t), "paged_mqa_logits: output must not alias inputs");
    TORCH_CHECK(kernel_id>=0 && kernel_id<=5, "paged_mqa_logits: kernelId must be in [0,5]");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(q));
    // Cache per calling thread/device; never infer a tensor's architecture from
    // GPU_ARCHS or from a different current device.
    static thread_local int cached_device=-1;
    static thread_local bool supported=false;
    static thread_local bool is_gfx946=false;
    const int device=q.get_device();
    if(cached_device!=device) {
        hipDeviceProp_t prop{};
        TORCH_CHECK(hipGetDeviceProperties(&prop,device)==hipSuccess,"cannot query HIP device");
        is_gfx946=std::strncmp(prop.gcnArchName,"gfx946",6)==0;
        supported=is_gfx946 || std::strncmp(prop.gcnArchName,"gfx938",6)==0;
        cached_device=device;
    }
    TORCH_CHECK(supported,"paged_mqa_logits: only gfx938/gfx946 FP8 is implemented");
    TORCH_CHECK(is_gfx946 ? kernel_id>=4 : kernel_id<=3,
                "paged_mqa_logits: gfx938 requires ID0-3; gfx946 requires ID4-5");
    const dim3 grid(static_cast<unsigned>((max_len+63)/64),static_cast<unsigned>(batch*next));
    const auto stream=at::hip::getCurrentHIPStream();
    if(is_gfx946) {
#define LAUNCH_GFX946(H,P) hipLaunchKernelGGL((opus_paged_mqa::paged_mqa_gfx946<H,P>),grid,dim3(256),0,stream, \
        static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(), \
        weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(), \
        output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)))
        if(kernel_id==4) {
            if(heads==32) { LAUNCH_GFX946(32,false); } else { LAUNCH_GFX946(64,false); }
        } else {
            if(heads==32) { LAUNCH_GFX946(32,true); } else { LAUNCH_GFX946(64,true); }
        }
#undef LAUNCH_GFX946
    }
    else if(kernel_id>=2) {
#define LAUNCH_DPP(H,P) hipLaunchKernelGGL((opus_paged_mqa::paged_mqa_dpp<H,P>),grid,dim3(256),0,stream, \
        static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(), \
        weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(), \
        output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)))
        if(kernel_id==2) {
            if(heads==32) { LAUNCH_DPP(32,false); } else { LAUNCH_DPP(64,false); }
        } else {
            if(heads==32) { LAUNCH_DPP(32,true); } else { LAUNCH_DPP(64,true); }
        }
#undef LAUNCH_DPP
    }
    else if(kernel_id==1 && heads==32)
        hipLaunchKernelGGL((opus_paged_mqa::paged_mqa_id0<32,true>),grid,dim3(256),0,stream,
            static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(),
            weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),
            output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)));
    else if(kernel_id==1 && heads==64)
        hipLaunchKernelGGL((opus_paged_mqa::paged_mqa_id0<64,true>),grid,dim3(256),0,stream,
            static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(),
            weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),
            output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)));
    else if(heads==32)
        hipLaunchKernelGGL(opus_paged_mqa::paged_mqa_id0<32>,grid,dim3(256),0,stream,
            static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(),
            weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),
            output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)));
    else
        hipLaunchKernelGGL(opus_paged_mqa::paged_mqa_id0<64>,grid,dim3(256),0,stream,
            static_cast<const unsigned char*>(q.data_ptr()),cache.data_ptr<unsigned char>(),
            weights.data_ptr<float>(),context.data_ptr<int>(),tables.data_ptr<int>(),
            output.data_ptr<float>(),static_cast<int>(next),static_cast<int>(max_len),static_cast<int>(tables.size(1)));
    const auto err=hipGetLastError();
    TORCH_CHECK(err==hipSuccess,"paged_mqa_logits launch: ",hipGetErrorString(err));
}
