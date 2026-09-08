# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 Hygon Info Technologies Ltd.
import torch
from aiter.test_common import checkAllclose, perftest
from aiter import dtypes
from aiter.fused_moe import torch_moe, fused_topk
from aiter.ops.triton.fused_moe import fused_experts_impl
#from aiter.fused_moe_bf16_asm import asm_moe
from aiter.fused_moe_asm import asm_moe
from aiter.ops.shuffle import shuffle_weight,ck_shuffle_weight
from aiter import pertoken_quant, ck_moe, ck_shuffle_moe
from aiter.int4_utils import *
from aiter import ActivationType

BLOCK_SIZE_M = 16

def permute_weight_a(x: torch.Tensor) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    BK = 128
    BN = 128
    x_ = x
    x_ = x_.view(
        x.shape[0], x.shape[1] // BN, BN // 16, 16, x.shape[2] // BK, BK // 32, 4, 8
    )
    x_ = x_.permute(0, 1, 5, 2, 6, 4, 3, 7)
    x_ = x_.contiguous()
    x_ = x_.view(x.shape[0], x.shape[1], x.shape[2])
    return x_


@perftest(num_warmup=1, num_iters=2)
def torch_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    activation=ActivationType.Silu,
):
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        None,
        activation,
    )

#@perftest(testGraph=True)
@perftest(num_warmup=1, num_iters=2)
def asm_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    a16=False,
    activation=ActivationType.Silu,
):
    return asm_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        a16,
        None,
        None,
        None,
        activation,
    )

#@perftest(testGraph=True)
@perftest(num_warmup=1, num_iters=2)
def ck_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
):
    return ck_shuffle_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        block_m = BLOCK_SIZE_M
    ).to(dtypes.fp16)

#@perftest(testGraph=True)
@perftest(num_warmup=1, num_iters=2)
def triton_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
):
   return fused_experts_impl(hidden_states, w1, w2, topk_weight, topk_ids)

# @perftest()
# def vllm_moe(hidden_states, w1, w2, topk_weight, topk_ids):
#     return fused_experts(hidden_states, w1, w2, topk_weight, topk_ids, inplace=False)


quant_algo = [
    "No",  # g1u0/ck(g1ux) support
    "int8quant",  # g1u1 support
    "fp8quant",  # g1u1 support
    "int8smoothquant",  # g1u1/g1u0 support
    "fp8smoothquant",  # g1u1 support
    "wint4afp8smoothquant",  # g1u1 support
]


def test_fmoe(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    quant="No",
    use_g1u1=False,
    shared_E=0,
    activation=ActivationType.Silu,
):
    quantAlgoId = quant_algo.index(quant)
    if quantAlgoId not in [0, 3] and not use_g1u1:
        print("g1u0 only could test no quant and int8smoothquant")
        return
    # torch.manual_seed(0)
    quantstr = quant_algo[quantAlgoId]
    use_int4 = "wint4" in quantstr
    quant_dtype = dtypes.i8 if use_int4 or quantstr.startswith("int8") else dtypes.fp8
    use_smooth = "smooth" in quantstr
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")

    if use_g1u1:
        w1 = (
            torch.randn(
                (E + shared_E, inter_dim * 2, model_dim), dtype=dtype, device="cuda"
            )
            / 10.0
        )
    else:
        w1 = torch.randn(
            (E + shared_E, inter_dim, model_dim), dtype=dtype, device="cuda"
        )
    w2 = torch.randn((E + shared_E, model_dim, inter_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    if shared_E > 0:
        shared_E_score = 0.5
        s_topk_weights = torch.tensor(
            [
                [shared_E_score, shared_E_score],
            ]
            * token,
            dtype=dtypes.fp32,
            device=input.device,
        )
        topk_weights = torch.cat((topk_weights, s_topk_weights), dim=1)
        s_topk_ids = torch.tensor(
            [
                [E, E + 1],
            ]
            * token,
            dtype=dtypes.i32,
            device=input.device,
        )
        topk_ids = torch.cat((topk_ids, s_topk_ids), dim=1)

    # ref implement
    # w1a = permute_weight_a(w1)
    # w2a = permute_weight_a(w2)
    # w1a = w1
    # w2a = w2
    # avg_a = 1
    # ref1, avg_a = vllm_moe(input,
    #                        w1a,
    #                        w2a,
    #                        topk_weights,
    #                        topk_ids)
    # print(f'{ref1=}')

    if quantAlgoId == 0:
        # ref2 implement
        ref2, avg_c = torch_moe_test(input, w1, w2, topk_weights, topk_ids)

        # b implement
        # TODO: Support shuffle weight
        # w1b = shuffle_weight(w1)
        # w2b = shuffle_weight(w2)

        if use_g1u1:
            #out_b = ref2
            #avg_b = 9999
            #print("asm g1u1 only support quant/smoothquant Now")
            out_b, avg_b = asm_moe_test(input, w1, w2, topk_weights, topk_ids)
        else:
            assert False, "not support asm yet"
            # out_b, avg_b = asm_moe_test(input, w1b, w2b, topk_weights, topk_ids)

        out_triton, avg_triton = triton_moe_test(input, w1, w2, topk_weights, topk_ids)

        # test ck moe
        # out_ck, avg_ck = ck_moe_test(
        #     input, w1b, w2b, topk_weights, topk_ids, None, None, None, None
        # )
        # print(w1.shape)
        w1a = ck_shuffle_weight(w1)
        print(w1a.shape)
        out_ck, avg_ck = ck_moe_test(
            input, w1a, w2, topk_weights, topk_ids, None, None, None, None
        )

        # print(out_ck.shape)
        torch.set_printoptions(profile="full")

        msg = f"[perf] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, triton_avg: {avg_triton:>8.2f} us, ck_avg: {avg_ck:>8.2f} us, uplift: {avg_triton/avg_ck-1:.1%} blas_avg: {avg_b:>8.2f} us, uplift: {avg_triton/avg_b-1:.1%}"
        checkAllclose(ref2, out_b, rtol=0.01, atol=100, msg=msg)
        checkAllclose(ref2, out_ck, rtol=0.01, atol=100, msg="ck check", printNum=300)
    else:
        assert False, "quant method not support yet"
"""
        dtypeMax = 7 if use_int4 else None
        w1, fc1_scale = pertoken_quant(w1, quant_dtype=quant_dtype, dtypeMax=dtypeMax)
        w2, fc2_scale = pertoken_quant(w2, quant_dtype=quant_dtype, dtypeMax=dtypeMax)

        sp1 = (E + shared_E, inter_dim)
        sp2 = (E + shared_E, model_dim)

        if not use_smooth:
            fc1_smooth_scale = None
            fc2_smooth_scale = None
        else:
            if use_int4:
                # fixme @felix: hack here, int4 kernel need this buffer but not used, so ones.
                # [expert, 1, model_dim]
                fc1_smooth_scale = torch.ones(sp2, dtype=dtypes.fp32, device="cuda")
                # [expert, 1, inter_dim]
                fc2_smooth_scale = torch.ones(sp1, dtype=dtypes.fp32, device="cuda")
            else:
                # [expert, 1, model_dim]
                fc1_smooth_scale = torch.randn(sp2, dtype=dtypes.fp32, device="cuda")
                # [expert, 1, inter_dim]
                fc2_smooth_scale = torch.randn(sp1, dtype=dtypes.fp32, device="cuda")

        # ref2 implement
        ref2, avg_c = torch_moe_test(
            input,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            activation,
        )

        # b implement
        if use_int4:
            w1 = rearrange_4bit_elements(convert_int8_to_uint32_int4(w1))
            w2 = rearrange_4bit_elements(convert_int8_to_uint32_int4(w2))
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)
        out_b, avg_b = asm_moe_test(
            input,
            w1b,
            w2b,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            a16=False,
            activation=activation,
        )

        def calculateTensorsSize(*args):
            num_btype = 0
            for el in args:
                if isinstance(el, torch.Tensor):
                    num_btype += el.element_size() * el.numel()
            return num_btype

        num_tb = calculateTensorsSize(
            input,
            input,
            w1b,
            w2b,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
        ) / (1024 * 1024 * 1024 * 1024.0)
        bw = num_tb * 1e6 / avg_b
        print(
            f"[BW  ] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, dtype: {dtype}, asm_bandwidth: {bw:>8.2f}TB/s"
        )

        if use_smooth and (
            (
                (inter_dim % 512 == 0 or inter_dim % 320 == 0)
                and (w1b.dtype == dtypes.fp8 and inter_dim * 2 == w1b.shape[1])
            )
            or (
                (inter_dim % 320 == 0)
                and (w1b.dtype == dtypes.i8 and inter_dim * 2 == w1b.shape[1])
            )
            or (
                (inter_dim % 512 == 0)
                and (w1b.dtype == dtypes.i8 and inter_dim == w1b.shape[1])
            )
        ):
            out_b2, avg_b2 = asm_moe_test(
                input,
                w1b,
                w2b,
                topk_weights,
                topk_ids,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
                a16=True,
                activation=activation,
            )
            msg = f"[perf] a8w8 asm: {avg_b:>8.2f} vs a16w8 asm: {avg_b2:>8.2f} ......"
            checkAllclose(ref2, out_b2, atol=100, msg=msg)

        # # test ck moe, not support now
        # out_ck, avg_ck = ck_moe_test(input, w1b, w2b, topk_weights, topk_ids,
        #                              fc1_scale, fc2_scale,
        #                              fc1_smooth_scale, fc2_smooth_scale)
        msg = f"[perf] {use_g1u1=} {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:>8.2f} us ...... uplift: {avg_c/avg_b-1:.1%}"
        checkAllclose(ref2, out_b, rtol=0.01, atol=100, msg=msg)
        # checkAllclose(ref2, avg_ck, rtol=0.01, atol=100)
"""

# print("test test_fmoe 16 bit")
# print("\ng1u0 no quant")
# # for dtype in [dtypes.fp16, dtypes.bf16]:
# for dtype in [dtypes.fp16]:
#     for m in [128, 256]:
#         for dim in [4096, 8192]:
#             for hdim in [1024]:
#                 # test_fmoe(dtype, m, dim, hdim, 32, 5)
#                 test_fmoe(dtype, m, dim, hdim, 32, 5, quant="No")

print("\ng1u1 no quant")
# for dtype in [dtypes.fp16, dtypes.bf16]:
for dtype in [dtypes.fp16]:
    for m in [1]:
    #for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,64,96,128,256,512,1024]:
    # for m in [1, 2,  4,  8, 16 ,32,64, 96, 128,256,512,1024,2048,4096,8192,16384,32768]:
        for dim in [7168]:
            for hdim in [256]:
                # test_fmoe(dtype, m, dim, hdim, 32, 5)
                # test_fmoe(dtype, m, dim, hdim, 256, 8, quant="No", use_g1u1=True)
                test_fmoe(dtype, m, dim, hdim, 256, 8, quant="No", use_g1u1=True)

# print("\ng1u1 int8quant")
# for dtype in [dtypes.bf16]:
#     for m in [128, 256]:
#         for dim in [4096, 8192]:
#             for hdim in [1024]:
#                 test_fmoe(
#                     dtype,
#                     m,
#                     dim,
#                     hdim,
#                     32,
#                     5,
#                     #   quant='int8quant', use_g1u1=True, shared_E=0, activation=ActivationType.Gelu)
#                     quant="int8quant",
#                     use_g1u1=True,
#                 )

# print("\ng1u1 fp8quant")
# for dtype in [dtypes.bf16]:
#     for m in [128, 256]:
#         for dim in [4096, 8192]:
#             for hdim in [1024]:
#                 test_fmoe(
#                     dtype,
#                     m,
#                     dim,
#                     hdim,
#                     32,
#                     5,
#                     quant="fp8quant",
#                     use_g1u1=True,
#                     shared_E=0,
#                     activation=ActivationType.Gelu,
#                 )
#                 #   quant='fp8quant', use_g1u1=True)


# print("\ng1u0 int8smoothquant")
# for dtype in [dtypes.bf16]:
#     for m in [128]:
#         for dim in [4096, 6144, 8192]:
#             for hdim in [512, 1024]:
#                 test_fmoe(
#                     dtype, m, dim, hdim, 32, 5, quant="int8smoothquant", use_g1u1=False
#                 )

# print("\ng1u1 int8smoothquant")
# for dtype in [dtypes.bf16]:
#     for m in [128]:
#         for dim in [4096, 6144, 8192]:
#             for hdim in [512, 1024, 1280]:
#                 test_fmoe(
#                     dtype, m, dim, hdim, 32, 5, quant="int8smoothquant", use_g1u1=True
#                 )

# print("\ng1u1 fp8smoothquant")
# for dtype in [dtypes.bf16]:
#     for m in [128]:
#         for dim in [4096, 6144, 8192]:
#             for hdim in [512, 1024, 1280]:
#                 test_fmoe(
#                     dtype, m, dim, hdim, 32, 5, quant="fp8smoothquant", use_g1u1=True
#                 )

# print("\ng1u1 int4")
# for dtype in [dtypes.bf16]:
#     for m in [32, 128]:
#         # for dim in [1024]:
#         for dim in [4096, 6144]:
#             for hdim in [1024, 4096]:
#                 test_fmoe(
#                     dtype,
#                     m,
#                     dim,
#                     hdim,
#                     8,
#                     3,
#                     quant="wint4afp8smoothquant",
#                     use_g1u1=True,
#                 )
