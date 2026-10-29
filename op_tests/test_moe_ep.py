# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import torch
from aiter.test_common import checkAllclose, perftest
from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter.fused_moe_layout import fused_moe_prepare_weights
from aiter import ActivationType, QuantType, pertoken_quant, dtypes

BLOCK_SIZE_M = 32
MAX_TOKENS = 4096 * 4


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
    expert_mask=None,
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
        expert_mask,
    )


@perftest(num_warmup=5, num_iters=20)
def fused_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for w8a8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    quant_type=QuantType.No,
    expert_mask=None,
    weight_layout=None,
):
    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask=expert_mask,
        activation=ActivationType.Silu,
        quant_type=quant_type,
        w1_scale=fc1_scale,
        w2_scale=fc2_scale,
        weight_layout=weight_layout,
    )


quant_algo = [
    "No",  # no quant -> triton backend
    "int8quant",  # w8a8 per token -> triton backend (EP)
    "fp8quant",  # w8a8 per token -> triton backend (EP)
    "int8smoothquant",  # w8a8 per token, smooth folded into weights
    "fp8smoothquant",  # w8a8 per token, smooth folded into weights
]


def test_fmoe_ep(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    quant="No",
    use_g1u1=False,
    shared_E=2,
    ep=8,
):
    # This gpu id in EP, this example use the last id
    ep_id = ep - 1
    # total_expert = unshared_expert + shared_expert + fake_expert(only use this fake expert id to mask)
    # expert_mask = torch.randint(0, 2, (E+shared_E+1,), dtype=dtypes.i32, device="cuda")
    expert_mask = torch.zeros((E + shared_E + 1,), dtype=dtypes.i32, device="cuda")
    expert_mask[ep_id * (E // ep) : (ep_id + 1) * E // ep] = 1
    # The last expert
    fake_expertid = expert_mask.numel() - 1
    # Ensure fake expert to be masked
    expert_mask[-1] = 0
    # Ensure shared expert not to be masked
    expert_mask[E:-1] = 1
    # # Get local expert Number in this gpu
    # local_E = torch.sum(expert_mask).item()

    quantAlgoId = quant_algo.index(quant)
    if quantAlgoId not in [0, 3] and not use_g1u1:
        print("g1u0 only could test no quant and int8smoothquant")
        return

    quantstr = quant_algo[quantAlgoId]
    quant_dtype = dtypes.i8 if quantstr.startswith("int8") else dtypes.fp8
    use_smooth = "smooth" in quantstr

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    if use_g1u1:
        w1 = (
            torch.randn(
                (E // ep + shared_E, inter_dim * 2, model_dim),
                dtype=dtype,
                device="cuda",
            )
            / 10
        )
    else:
        w1 = (
            torch.randn(
                (E // ep + shared_E, inter_dim, model_dim), dtype=dtype, device="cuda"
            )
            / 10
        )
    w2 = (
        torch.randn(
            (E // ep + shared_E, model_dim, inter_dim), dtype=dtype, device="cuda"
        )
        / 10
    )
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    # if shared_E > 0:
    shared_E_score = 0.1
    # init total_topk_ids, inference time you just need to fill ns_topk_ids in total_topk_ids
    total_topk_ids = torch.empty(
        (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.i32, device=input.device
    )
    ns_topk_ids, s_topk_ids = total_topk_ids.split([topk, shared_E + 1], dim=1)
    shared_expert_ids = [E + i for i in range(shared_E + 1)]
    s_topk_ids_list = [[fake_expertid] * (shared_E + 1)] * MAX_TOKENS
    for i in range(ep_id, MAX_TOKENS, ep):
        s_topk_ids_list[i] = shared_expert_ids
    s_topk_ids[:] = torch.tensor(s_topk_ids_list, dtype=dtypes.i32, device=input.device)

    # init total_topk_weights, inference time you just need to fill ns_topk_weights in total_topk_weights
    total_topk_weights = torch.empty(
        (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.fp32, device=input.device
    )
    ns_topk_weights, s_topk_weights = total_topk_weights.split(
        [topk, shared_E + 1], dim=1
    )
    s_topk_weights[:] = shared_E_score

    # inference time, use fused_topk to fill ns_topk_ids and ns_topk_weights
    fused_topk(input, score, topk, True, ns_topk_ids, ns_topk_weights)
    # inference time, topk_ids simply slices total_topk_ids into the number of input tokens, same for topk_weights
    topk_ids = total_topk_ids[:token]
    topk_weights = total_topk_weights[:token]

    # else:
    #     topk_ids, topk_weights = fused_topk(input, score, topk, True)

    if quantAlgoId == 0:
        # ref2 implement
        ref2, avg_c = torch_moe_test(
            input, w1, w2, topk_weights, topk_ids, expert_mask=expert_mask
        )

        # fused_moe implement (no quant -> triton backend, which converts the
        # 0/1 expert_mask to an expert map internally)
        # 加载期一次性准备（is_ep 场景决策为 raw）
        w1_prep, w2_prep, layout = fused_moe_prepare_weights(
            w1, w2, quant_type=QuantType.No, dtype=dtype, topk=topk,
            activation=ActivationType.Silu, expert_mask=expert_mask)
        out_fused, avg_fused = fused_moe_test(
            input, w1_prep, w2_prep, topk_weights, topk_ids,
            expert_mask=expert_mask, weight_layout=layout
        )

        msg = f"[perf] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, fused_avg: {avg_fused:>8.2f} us, uplift: {avg_c/avg_fused-1:.1%}"
        checkAllclose(ref2, out_fused, rtol=0.01, atol=10, msg=msg)

    else:
        sp1 = (E + shared_E, 1, inter_dim)
        sp2 = (E + shared_E, 1, model_dim)

        if not use_smooth:
            fc1_smooth_scale = None
            fc2_smooth_scale = None
        else:
            # [expert, 1, model_dim]
            fc1_smooth_scale = torch.randn(sp2, dtype=dtypes.fp32, device="cuda")
            # [expert, 1, inter_dim]
            fc2_smooth_scale = torch.randn(sp1, dtype=dtypes.fp32, device="cuda")

            # fused_moe has no argument for per-expert smooth
            # scales; fold them into the weights before quantization (standard
            # smoothquant deployment) so ref and fused_moe see the same math.
            fc1_smooth_scale = fc1_smooth_scale[: w1.shape[0]]
            fc2_smooth_scale = fc2_smooth_scale[: w2.shape[0]]
            w1 = w1 * fc1_smooth_scale
            w2 = w2 * fc2_smooth_scale

        w1, fc1_scale = pertoken_quant(w1, quant_dtype=quant_dtype)
        w2, fc2_scale = pertoken_quant(w2, quant_dtype=quant_dtype)

        # ref2 implement
        ref2, avg_c = torch_moe_test(
            input,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            expert_mask=expert_mask,
        )

        # fused_moe implement (w8a8 per token -> triton backend for EP; the
        # backend quantizes activations per token internally)
        # 加载期一次性准备（is_ep 场景决策为 raw）
        w1_prep, w2_prep, layout = fused_moe_prepare_weights(
            w1, w2, quant_type=QuantType.per_Token, w1_scale=fc1_scale,
            w2_scale=fc2_scale, dtype=dtype, topk=topk,
            activation=ActivationType.Silu, expert_mask=expert_mask)
        out_fused, avg_fused = fused_moe_test(
            input,
            w1_prep,
            w2_prep,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            quant_type=QuantType.per_Token,
            expert_mask=expert_mask,
            weight_layout=layout,
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
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
        ) / (1024 * 1024 * 1024 * 1024.0)
        bw = num_tb * 1e6 / avg_fused
        print(
            f"[BW  ] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, {topk=}, dtype: {dtype}, fused_bandwidth: {bw:>8.2f}TB/s"
        )

        msg = f"[perf] {use_g1u1=} {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, {topk=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, fused_avg: {avg_fused:>8.2f} us ...... uplift: {avg_c/avg_fused-1:.1%}"
        checkAllclose(ref2, out_fused, rtol=0.01, atol=10, msg=msg)


print("test test_fmoe 16 bit")
print("\ng1u0 no quant")
for dtype in [dtypes.fp16, dtypes.bf16]:
    for m in [7, 128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(
                        dtype, m, dim, hdim, 32, 5, quant="No", shared_E=2, ep=ep
                    )

print("\ng1u1 no quant")
for dtype in [dtypes.fp16, dtypes.bf16]:
    for m in [7, 128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(
                        dtype,
                        m,
                        dim,
                        hdim,
                        32,
                        5,
                        quant="No",
                        use_g1u1=True,
                        shared_E=2,
                        ep=ep,
                    )

print("\ng1u1 int8quant")
for dtype in [dtypes.bf16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(
                        dtype,
                        m,
                        dim,
                        hdim,
                        32,
                        5,
                        quant="int8quant",
                        use_g1u1=True,
                        shared_E=2,
                        ep=ep,
                    )

print("\ng1u1 fp8quant")
for dtype in [dtypes.bf16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(
                        dtype,
                        m,
                        dim,
                        hdim,
                        32,
                        5,
                        quant="fp8quant",
                        use_g1u1=True,
                        shared_E=2,
                        ep=ep,
                    )


print("\ng1u0 int8smoothquant")
for dtype in [dtypes.bf16]:
    for m in [128]:
        for dim in [4096, 6144, 8192]:
            for hdim in [512, 1024]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(
                        dtype,
                        m,
                        dim,
                        hdim,
                        32,
                        5,
                        quant="int8smoothquant",
                        use_g1u1=False,
                        shared_E=2,
                        ep=ep,
                    )

print("\ng1u1 int8smoothquant")
for dtype in [dtypes.bf16]:
    for m in [128]:
        for dim in [4096]:
            for hdim in [1280]:
                for ep in [8]:
                    test_fmoe_ep(
                        dtype,
                        m,
                        dim,
                        hdim,
                        128,
                        6,
                        quant="int8smoothquant",
                        use_g1u1=True,
                        shared_E=2,
                        ep=ep,
                    )

print("\ng1u1 fp8smoothquant")
for dtype in [dtypes.bf16]:
    for m in [128]:
        for dim in [4096, 6144, 8192]:
            for hdim in [512, 1024, 1280]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(
                        dtype,
                        m,
                        dim,
                        hdim,
                        32,
                        5,
                        quant="fp8smoothquant",
                        use_g1u1=True,
                        shared_E=2,
                        ep=ep,
                    )
