# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
"""CK GDN native entry points (module_gdn_ck).

Covers three levels:

1. workspace sizing / argument validation of the native surface;
2. CK self-consistency -- the chunked prefill pipeline and the fused recurrent
   decode compute the same recurrence, so their output and final state must
   agree on identical inputs;
3. cross-implementation triangulation -- CK decode vs HIPC
   (``aiter_fused_recurrent_gated_delta_rule_packed_decode``) vs Triton
   (``aiter.ops.triton.fla.fused_recurrent``), all fed from the same
   ``mixed_qkv`` / ``a`` / ``b`` / ``A_log`` / ``dt_bias`` tensors.

Run from the repository root:

    PYTHONPATH=. python -m pytest -q op_tests/test_ck_gdn_native.py

The first run JIT-compiles ``module_gdn_ck`` (CK GDN + adapter), which is slow;
pre-build it with ``PREBUILD_KERNELS=1`` or a dummy call before benchmarking.
"""

from __future__ import annotations

import pytest
import torch

import aiter
from aiter.ops.triton.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as triton_packed_decode,
)

HEAD_DIM = 128
CHUNK = 64


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    # CK's own correctness script uses 3e-2 (bf16) / 2e-2 (fp16) against an
    # independent GPU reference; cross-implementation runs use the same order.
    return (3e-2, 3e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str, rtol: float, atol: float):
    diff = (actual.float() - expected.float()).abs()
    tol = atol + rtol * expected.float().abs()
    bad = (diff > tol).sum().item()
    assert bad == 0, (
        f"{name}: {bad} elements out of tolerance "
        f"(max_abs_diff={diff.max().item():.4e}, max_tol={tol.max().item():.4e})"
    )


def _prefill_workspace(q, v, **flags) -> torch.Tensor:
    need = aiter.ck_gdn_prefill_workspace_bytes(q, v, **flags)
    assert need > 0
    return torch.empty(need, dtype=torch.uint8, device=q.device)


# --------------------------------------------------------------------------- #
# 1. sizing / validation
# --------------------------------------------------------------------------- #
def test_workspace_bytes_matches_layout():
    torch.manual_seed(0)
    q = torch.randn(256, 2, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(256, 8, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    base = aiter.ck_gdn_prefill_workspace_bytes(q, v)
    # Enable every optional region and check the workspace grows accordingly.
    bigger = aiter.ck_gdn_prefill_workspace_bytes(
        q,
        v,
        use_gk=True,
        transpose_state=True,
        qk_l2norm=True,
        beta_sigmoid_in_kernel=True,
        cp_context=True,
        cp_world_size=2,
    )
    assert bigger > base

    ws = aiter.ck_gdn_prefill_alloc_workspace(q, v)
    assert ws.numel() == base and ws.dtype == torch.uint8
    assert ws.device == q.device


def test_workspace_too_small_is_rejected():
    torch.manual_seed(0)
    q = torch.randn(64, 2, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(64, 8, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    g = -torch.rand(64, 8, dtype=torch.float32, device="cuda")
    beta = torch.rand(64, 8, dtype=torch.float32, device="cuda")

    need = aiter.ck_gdn_prefill_workspace_bytes(q, v)
    with pytest.raises(RuntimeError, match="workspace is too small"):
        aiter.ck_gdn_prefill(q, q, v, g, beta, torch.empty(need // 2, dtype=torch.uint8, device="cuda"))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"use_g": True, "g": None}, "g is required"),
        ({"use_gk": True, "gk": None}, "gk is required"),
        ({"gate_in_kernel": True, "a_log": None}, "a_log is required"),
    ],
)
def test_missing_optional_inputs_are_rejected(kwargs, match):
    torch.manual_seed(0)
    q = torch.randn(64, 2, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(64, 8, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    beta = torch.rand(64, 8, dtype=torch.float32, device="cuda")
    g = -torch.rand(64, 8, dtype=torch.float32, device="cuda")
    ws = aiter.ck_gdn_prefill_alloc_workspace(q, v, **{k: v2 for k, v2 in kwargs.items() if v2 is not None})

    call = {"g": g}
    call.update(kwargs)
    with pytest.raises(RuntimeError, match=match):
        aiter.ck_gdn_prefill(q, q, v, beta=beta, workspace=ws, **call)


def test_wrong_head_dim_is_rejected():
    torch.manual_seed(0)
    q = torch.randn(64, 2, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(64, 8, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match=r"\[tokens, heads, 128\]"):
        aiter.ck_gdn_prefill_workspace_bytes(q, v)


# --------------------------------------------------------------------------- #
# 2. CK prefill vs CK decode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
@pytest.mark.parametrize("heads,value_heads", [(2, 8), (2, 4), (2, 2)])
@pytest.mark.parametrize("tokens", [192, 1024])
def test_ck_prefill_matches_ck_decode(dtype_name, heads, value_heads, tokens):
    torch.manual_seed(0)
    dtype = _dtype(dtype_name)
    # 192 = 3 chunks (logical H); 1024 additionally exercises CK's packed
    # ("preshuffled") H layout and the state_cp_eligible branch.

    q = (torch.randn(tokens, heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.1)
    k = (torch.randn(tokens, heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.1)
    v = (torch.randn(tokens, value_heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.1)
    g = -torch.rand(tokens, value_heads, dtype=torch.float32, device="cuda")
    beta = torch.rand(tokens, value_heads, dtype=torch.float32, device="cuda")
    state = torch.randn(1, value_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda") * 0.02

    # The size query and the launch must agree on flags; both modes are covered
    # here (explicit buffer + adapter-side allocation).
    flags = dict(use_g=True, use_gk=False, use_exp2=False, transpose_state=True)
    ws = aiter.ck_gdn_prefill_alloc_workspace(q, v, **flags)
    out_prefill, state_prefill = aiter.ck_gdn_prefill(
        q,
        k,
        v,
        g,
        beta,
        ws,
        initial_state=state.clone(),
        **flags,
    )
    out_auto, state_auto = aiter.ck_gdn_prefill(
        q,
        k,
        v,
        g,
        beta,
        None,
        initial_state=state.clone(),
        **flags,
    )
    assert torch.equal(out_auto, out_prefill) and torch.equal(state_auto, state_prefill), (
        "explicit workspace and adapter-allocated workspace must produce identical results"
    )
    out_decode, state_decode = aiter.ck_gdn_decode(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state.clone(),
        use_exp2=False,
        transpose_state=True,
    )

    rtol, atol = _tolerances(dtype)
    _assert_close(out_prefill, out_decode, "prefill vs decode output", rtol, atol)
    _assert_close(state_prefill, state_decode, "prefill vs decode final state", rtol, atol)


# --------------------------------------------------------------------------- #
# 3. CK vs HIPC vs Triton (packed decode, one token per sequence)
# --------------------------------------------------------------------------- #
def _packed_decode_inputs(dtype, batch, heads, value_heads, state_pool):
    torch.manual_seed(0)
    q = (torch.randn(batch, heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.2)
    k = (torch.randn(batch, heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.2)
    v = (torch.randn(batch, value_heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.2)
    mixed_qkv = torch.cat(
        (q.reshape(batch, heads * HEAD_DIM),
         k.reshape(batch, heads * HEAD_DIM),
         v.reshape(batch, value_heads * HEAD_DIM)),
        dim=-1,
    ).contiguous()

    a = torch.randn(batch, value_heads, dtype=torch.float32, device="cuda") * 0.2
    b = torch.randn(batch, value_heads, dtype=torch.float32, device="cuda") * 0.2
    a_log = -torch.rand(value_heads, dtype=torch.float32, device="cuda") * 2.0
    dt_bias = torch.randn(value_heads, dtype=torch.float32, device="cuda") * 0.1
    state_pool_t = torch.randn(
        state_pool, value_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda"
    ) * 0.02
    # HIPC/Triton packed decode indexes a state pool with int32 indices and
    # updates the pool in place.
    indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    return dict(
        q=q, k=k, v=v, mixed_qkv=mixed_qkv, a=a, b=b,
        a_log=a_log, dt_bias=dt_bias, state_pool=state_pool_t, indices=indices,
    )


@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
@pytest.mark.parametrize("heads,value_heads", [(2, 8), (2, 4), (2, 2)])
def test_ck_decode_matches_hipc_and_triton(dtype_name, heads, value_heads):
    dtype = _dtype(dtype_name)
    batch, state_pool = 3, 5
    data = _packed_decode_inputs(dtype, batch, heads, value_heads, state_pool)
    scale = HEAD_DIM ** -0.5
    rtol, atol = _tolerances(dtype)

    # --- HIPC + Triton reference ------------------------------------------ #
    # Both update the state pool in place and return (out, initial_state), with
    # out shaped (batch, 1, HV, V).
    refs = {}
    for name, fn in (
        ("hipc", aiter.aiter_fused_recurrent_gated_delta_rule_packed_decode),
        ("triton", triton_packed_decode),
    ):
        pool = data["state_pool"].clone()
        out = torch.empty(batch, 1, value_heads, HEAD_DIM, dtype=dtype, device="cuda")
        fn(
            mixed_qkv=data["mixed_qkv"],
            a=data["a"],
            b=data["b"],
            A_log=data["a_log"],
            dt_bias=data["dt_bias"],
            scale=scale,
            initial_state=pool,
            out=out,
            ssm_state_indices=data["indices"],
            use_qk_l2norm_in_kernel=False,
        )
        refs[name] = (
            out.reshape(batch, value_heads, HEAD_DIM).clone(),
            pool.index_select(0, data["indices"].to(torch.int64)).clone(),
        )

    # --- CK --------------------------------------------------------------- #
    # CK has no state pool: gather the slots the framework would have used and
    # hand CK a dense [sequences, HV, V, K] state, matching the Triton [V, K]
    # layout through transpose_state=True.
    ck_state = data["state_pool"].index_select(0, data["indices"].to(torch.int64)).clone()
    ck_out, ck_final = aiter.ck_gdn_decode(
        data["q"],
        data["k"],
        data["v"],
        data["a"],
        torch.sigmoid(data["b"]),
        a_log=data["a_log"],
        dt_bias=data["dt_bias"],
        initial_state=ck_state,
        cu_seqlens=torch.arange(batch + 1, dtype=torch.int64, device="cuda"),
        scale=scale,
        use_g=True,
        gate_in_kernel=True,
        has_dt_bias=True,
        transpose_state=True,
        store_final_state=True,
    )
    assert ck_final is not None

    _assert_close(ck_out, refs["triton"][0], "CK vs Triton output", rtol, atol)
    _assert_close(ck_final, refs["triton"][1], "CK vs Triton final state", rtol, atol)
    _assert_close(ck_out, refs["hipc"][0], "CK vs HIPC output", rtol, atol)
    _assert_close(ck_final, refs["hipc"][1], "CK vs HIPC final state", rtol, atol)
