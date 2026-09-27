# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Benchmark the native CK GDN entry points (module_gdn_ck) against HIPC GDN.

Modes:

* ``--mode decode``  -- packed decode (one token per sequence) timed for CK and
  HIPC from identical ``mixed_qkv`` / ``a`` / ``b`` / ``A_log`` / ``dt_bias`` /
  state-pool inputs, with an optional numerics check;
* ``--mode update``  -- sigmoid-gating sequence update (``a``/``b``/``A_log`` /
  ``dt_bias``, non-spec) timed for CK and HIPC. ``seq_len == 1`` is the
  operator's primary contract; ``seq_len > 1`` requires per-token state slots;
* ``--mode prefill`` -- CK prefill pipeline (preprocess + cumsum + kkt +
  recompute_w_u + state + output) timed on its own. HIPC has no single-operator
  counterpart at this granularity, so it is a single-row benchmark.

Examples
--------
    PYTHONPATH=. python op_tests/op_benchmarks/bench_ck_gdn_native.py --mode decode
    PYTHONPATH=. python op_tests/op_benchmarks/bench_ck_gdn_native.py \
        --mode update --batch 8 --seqlen 8 --heads 4 --value-heads 16
    PYTHONPATH=. python op_tests/op_benchmarks/bench_ck_gdn_native.py \
        --mode prefill --seqlen 4096 --heads 16 --value-heads 64 --dtype bf16
    PYTHONPATH=. python op_tests/op_benchmarks/bench_ck_gdn_native.py \
        --mode decode --out-json bench_gdn_results/ck_decode.json

The first call JIT-compiles ``module_gdn_ck``; run ``bench_gdn.sh compile`` (or
any correctness test) once beforehand so the timing numbers are meaningful.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

import aiter

HEAD_DIM = 128


def parse_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def timed(fn: Callable[[], None], warmup: int, rep: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rep)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in events]
    return statistics.median(samples) * 1000.0, min(samples) * 1000.0  # microseconds


def build_packed_decode(batch: int, heads: int, value_heads: int, dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.2
    k = torch.randn(batch, heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.2
    v = torch.randn(batch, value_heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.2
    mixed_qkv = torch.cat(
        (
            q.reshape(batch, heads * HEAD_DIM),
            k.reshape(batch, heads * HEAD_DIM),
            v.reshape(batch, value_heads * HEAD_DIM),
        ),
        dim=-1,
    ).contiguous()
    return dict(
        q=q,
        k=k,
        v=v,
        mixed_qkv=mixed_qkv,
        a=torch.randn(batch, value_heads, dtype=torch.float32, device="cuda") * 0.2,
        b=torch.randn(batch, value_heads, dtype=torch.float32, device="cuda") * 0.2,
        a_log=-torch.rand(value_heads, dtype=torch.float32, device="cuda") * 2.0,
        dt_bias=torch.randn(value_heads, dtype=torch.float32, device="cuda") * 0.1,
        state_pool=torch.randn(batch, value_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda") * 0.02,
        indices=torch.arange(batch, dtype=torch.int32, device="cuda"),
    )


def bench_decode(args) -> dict:
    dtype = parse_dtype(args.dtype)
    scale = args.scale if args.scale is not None else HEAD_DIM**-0.5
    data = build_packed_decode(args.batch, args.heads, args.value_heads, dtype, args.seed)

    # CK's tuned raw-gate path requires the gate and dt_bias tensors to carry the
    # input dtype while beta/a_log stay fp32. The HIPC kernel keeps fp32 gates,
    # so its reference row always uses the fp32 ``a``.
    gate_dtype = parse_dtype(args.gate_dtype)
    ck_gate = data["a"] if args.gate_dtype == "f32" else data["a"].to(gate_dtype)
    ck_dt_bias = data["dt_bias"] if args.gate_dtype == "f32" else data["dt_bias"].to(gate_dtype)

    base_pool = data["state_pool"].clone()
    pool_hipc = base_pool.clone()
    ck_state0 = base_pool.index_select(0, data["indices"].to(torch.int64)).clone()
    ck_state = ck_state0.clone()
    cu_seqlens = torch.arange(args.batch + 1, dtype=torch.int64, device="cuda")
    beta_sigmoid = torch.sigmoid(data["b"])
    out_hipc = torch.empty(args.batch, 1, args.value_heads, HEAD_DIM, dtype=dtype, device="cuda")

    def run_hipc():
        pool_hipc.copy_(base_pool)
        aiter.aiter_fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=data["mixed_qkv"],
            a=data["a"],
            b=data["b"],
            A_log=data["a_log"],
            dt_bias=data["dt_bias"],
            scale=scale,
            initial_state=pool_hipc,
            out=out_hipc,
            ssm_state_indices=data["indices"],
            use_qk_l2norm_in_kernel=False,
        )

    def ck_call():
        return aiter.ck_gdn_decode(
            data["q"],
            data["k"],
            data["v"],
            ck_gate,
            beta_sigmoid,
            a_log=data["a_log"],
            dt_bias=ck_dt_bias,
            initial_state=ck_state,
            cu_seqlens=cu_seqlens,
            scale=scale,
            use_g=True,
            gate_in_kernel=True,
            has_dt_bias=True,
            transpose_state=True,
            store_final_state=True,
        )

    def run_ck():
        ck_state.copy_(ck_state0)
        ck_call()

    results = {}
    for name, fn in (("hipc", run_hipc), ("ck", run_ck)):
        median_us, best_us = timed(fn, args.warmup, args.rep)
        results[name] = {"median_us": median_us, "best_us": best_us}
        print(f"decode {name:<6} median {median_us:9.2f} us  best {best_us:9.2f} us")

    if not args.no_verify:
        run_hipc()
        ck_state.copy_(ck_state0)
        ck_out, _ = ck_call()
        torch.cuda.synchronize()
        rtol, atol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)
        diff = (ck_out.reshape_as(out_hipc).float() - out_hipc.float()).abs()
        tol = atol + rtol * out_hipc.float().abs()
        bad = int((diff > tol).sum().item())
        print(f"  CK vs hipc: mismatches={bad} max_abs={diff.max().item():.4e}")
        results["verify_vs_hipc"] = {"mismatches": bad, "max_abs": diff.max().item()}
        results["speedup_hipc_over_ck"] = results["hipc"]["median_us"] / results["ck"]["median_us"]

    results["shape"] = {
        "batch": args.batch,
        "heads": args.heads,
        "value_heads": args.value_heads,
        "dtype": args.dtype,
        "gate_dtype": args.gate_dtype,
    }
    return results


def _randn(shape, dtype, scale):
    return (torch.randn(shape, device="cuda", dtype=dtype) * scale).contiguous()


def bench_update(args) -> dict:
    """CK decode vs HIPC sigmoid-gating sequence update (non-spec).

    Same tensors feed both: a/b carry the input dtype, A_log and the state pool
    are fp32, and the state pool is updated in place. CK has no state pool, so
    its slots are gathered once and its dense final state is compared back
    against the pool the HIPC kernel wrote.
    """
    dtype = parse_dtype(args.dtype)
    num_reqs, seq_len = args.batch, args.seqlen
    heads, value_heads = args.heads, args.value_heads
    num_tokens = num_reqs * seq_len
    # The HIPC kernel indexes the state pool per TOKEN
    # (ssm_state_indices[i_n * stride_seq + i_t]), so T > 1 needs a
    # (num_reqs, seq_len) index tensor and one distinct slot per token.
    state_pool = max(4 * num_reqs, num_tokens)
    scale = args.scale if args.scale is not None else HEAD_DIM**-0.5

    torch.manual_seed(args.seed)
    q = _randn((1, num_tokens, heads, HEAD_DIM), dtype, 0.2)
    k = _randn((1, num_tokens, heads, HEAD_DIM), dtype, 0.2)
    v = _randn((1, num_tokens, value_heads, HEAD_DIM), dtype, 0.2)
    A_log = _randn((value_heads,), torch.float32, 0.2)
    a = _randn((num_tokens, value_heads), dtype, 0.2)
    b = _randn((num_tokens, value_heads), dtype, 0.2)
    dt_bias = _randn((value_heads,), dtype, 0.2)
    base_pool = _randn((state_pool, value_heads, HEAD_DIM, HEAD_DIM), torch.float32, 0.02)

    slots = torch.randperm(state_pool, device="cuda", dtype=torch.int64)[:num_tokens]
    indices = slots.to(torch.int32).reshape(num_reqs, seq_len).contiguous()
    first_slots = indices[:, 0].to(torch.int64)
    last_slots = indices[:, -1].to(torch.int64)
    cu_seqlens = torch.arange(0, num_tokens + 1, seq_len, device="cuda", dtype=torch.int32)
    if seq_len == 1:
        # A single token per request is the kernel's primary contract; keep the
        # 1-D index form the production decode path uses.
        indices = indices.reshape(-1).contiguous()

    pools = {"hipc": base_pool.clone()}
    returned: dict[str, tuple] = {}
    ck_state0 = base_pool.index_select(0, first_slots).clone()
    ck_state = ck_state0.clone()
    cu64 = cu_seqlens.to(torch.int64).contiguous()
    beta_sigmoid = torch.sigmoid(b)

    def run_hipc():
        pools["hipc"].copy_(base_pool)
        returned["hipc"] = aiter.vllm_fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
            beta=1.0, threshold=20.0, scale=scale,
            initial_state=pools["hipc"], inplace_final_state=True,
            cu_seqlens=cu_seqlens, ssm_state_indices=indices,
            num_accepted_tokens=None, use_qk_l2norm_in_kernel=False, is_kda=False,
        )

    def ck_call():
        return aiter.ck_gdn_decode(
            q.reshape(num_tokens, heads, HEAD_DIM),
            k.reshape(num_tokens, heads, HEAD_DIM),
            v.reshape(num_tokens, value_heads, HEAD_DIM),
            a,
            beta_sigmoid,
            a_log=A_log,
            dt_bias=dt_bias,
            initial_state=ck_state,
            cu_seqlens=cu64,
            scale=scale,
            use_g=True,
            gate_in_kernel=True,
            has_dt_bias=True,
            transpose_state=True,
            store_final_state=True,
        )

    def run_ck():
        ck_state.copy_(ck_state0)
        ck_call()

    results = {}
    for name, fn in (("hipc", run_hipc), ("ck", run_ck)):
        median_us, best_us = timed(fn, args.warmup, args.rep)
        results[name] = {"median_us": median_us, "best_us": best_us}
        print(f"update {name:<6} median {median_us:9.2f} us  best {best_us:9.2f} us")

    if not args.no_verify:
        run_hipc()
        ck_state.copy_(ck_state0)
        ck_out, ck_final = ck_call()
        torch.cuda.synchronize()
        rtol, atol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (2e-2, 2e-2)
        for name in ("hipc",):
            ref = returned[name][0]
            ref = ref.reshape(-1, value_heads, HEAD_DIM)[:num_tokens]
            diff = (ck_out.float() - ref.float()).abs()
            tol = atol + rtol * ref.float().abs()
            bad = int((diff > tol).sum().item())
            # The kernel writes the state after every token; the request's final
            # state lives in the slot of its last token.
            ref_state = pools[name].index_select(0, last_slots)
            sdiff = (ck_final - ref_state).abs()
            stol = atol + rtol * ref_state.abs()
            sbad = int((sdiff > stol).sum().item())
            print(
                f"  CK vs {name}: out mismatches={bad} max_abs={diff.max().item():.3e} | "
                f"state mismatches={sbad} max_abs={sdiff.max().item():.3e}"
            )
            results[f"verify_vs_{name}"] = {
                "out_mismatches": bad,
                "out_max_abs": diff.max().item(),
                "state_mismatches": sbad,
                "state_max_abs": sdiff.max().item(),
            }

    results["speedup_hipc_over_ck"] = results["hipc"]["median_us"] / results["ck"]["median_us"]
    results["shape"] = {
        "num_reqs": num_reqs,
        "seq_len": seq_len,
        "heads": heads,
        "value_heads": value_heads,
        "dtype": args.dtype,
    }
    return results


def bench_prefill(args) -> dict:
    dtype = parse_dtype(args.dtype)
    tokens = args.seqlen
    heads, value_heads = args.heads, args.value_heads
    torch.manual_seed(args.seed)
    q = torch.randn(tokens, heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    k = torch.randn(tokens, heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    v = torch.randn(tokens, value_heads, HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    g = -torch.rand(tokens, value_heads, dtype=torch.float32, device="cuda")
    beta = torch.rand(tokens, value_heads, dtype=torch.float32, device="cuda")

    flags = dict(use_g=True, use_gk=False, use_exp2=False, transpose_state=True)
    ws = aiter.ck_gdn_prefill_alloc_workspace(q, v, **flags)
    print(f"prefill workspace: {ws.numel() / 2**20:.1f} MiB")

    def run():
        aiter.ck_gdn_prefill(
            q, k, v, g, beta, ws, output_final_state=True, **flags
        )

    median_us, best_us = timed(run, args.warmup, args.rep)
    print(f"prefill ck  median {median_us:9.2f} us  best {best_us:9.2f} us")
    out, final_state = aiter.ck_gdn_prefill(
        q, k, v, g, beta, ws, output_final_state=True, **flags
    )
    assert torch.isfinite(out.float()).all()
    assert final_state is not None and torch.isfinite(final_state).all()
    return {
        "ck": {"median_us": median_us, "best_us": best_us},
        "workspace_mib": ws.numel() / 2**20,
        "shape": {
            "seqlen": tokens,
            "heads": heads,
            "value_heads": value_heads,
            "dtype": args.dtype,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser("Benchmark native CK GDN (module_gdn_ck)")
    parser.add_argument("--mode", choices=("decode", "prefill", "update"), default="decode")
    parser.add_argument("--batch", type=int, default=32, help="decode: packed sequences (1 token each)")
    parser.add_argument("--seqlen", type=int, default=4096, help="prefill: packed tokens")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--value-heads", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--gate-dtype",
        choices=("f32", "bf16", "fp16"),
        default="f32",
        help="decode: dtype of the CK gate/dt_bias tensors (bf16/fp16 hits CK's raw-gate fast path)",
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    if args.mode == "decode":
        results = bench_decode(args)
    elif args.mode == "update":
        results = bench_update(args)
    else:
        results = bench_prefill(args)
    results["mode"] = args.mode

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"json written to {path}")


if __name__ == "__main__":
    main()
