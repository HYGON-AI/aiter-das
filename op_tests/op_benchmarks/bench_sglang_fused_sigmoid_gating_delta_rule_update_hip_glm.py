# SPDX-License-Identifier: MIT
"""Benchmark the fixed SGLang target-verify KDA shape against Triton.

Run from the repository root:

    PYTHONPATH=. python \
      op_tests/op_benchmarks/bench_sglang_fused_sigmoid_gating_delta_rule_update_target_verify_8x6_h64.py

The fixed logical shape is:

    A_log                         [1, 1, 64, 1]
    a/raw_gate                    [1, 48, 64, 128]
    dt_bias                       [8192]
    q/k/v/o                       [1, 48, 64, 128]
    b/beta_raw                    [1, 48, 64]
    h0_indices                    [8]
    cu_seqlens                    [9] = [0, 6, ..., 48]
    intermediate_state_indices    [8]
    retrieve_parent_token         [8, 6]

The first dimension of state and intermediate cache is --num-cache-lines
(default: 8).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import triton

import aiter
from aiter.ops.triton.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as triton_sglang_fused_sigmoid_gating_delta_rule_update,
)

NUM_REQUESTS = 8
CACHE_STEPS = 6
NUM_TOKENS = NUM_REQUESTS * CACHE_STEPS
NUM_HEADS = 64
HEAD_DIM = 128


def parse_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError("unsupported dtype={}".format(name))


def randn(shape: Tuple[int, ...], dtype: torch.dtype, scale: float) -> torch.Tensor:
    return (torch.randn(shape, device="cuda", dtype=dtype) * scale).contiguous()


def make_inputs(args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    if args.num_cache_lines < NUM_REQUESTS:
        raise ValueError("--num-cache-lines must be at least {}".format(NUM_REQUESTS))

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    state_indices = torch.arange(NUM_REQUESTS, device="cuda", dtype=torch.int32)
    parent = torch.zeros((NUM_REQUESTS, CACHE_STEPS), device="cuda", dtype=torch.int32)
    parent[:, 1:] = torch.arange(CACHE_STEPS - 1, device="cuda", dtype=torch.int32)

    return {
        "A_log": randn((1, 1, NUM_HEADS, 1), torch.float32, args.input_scale),
        "a": randn((1, NUM_TOKENS, NUM_HEADS, HEAD_DIM), dtype, args.input_scale),
        "dt_bias": randn((NUM_HEADS * HEAD_DIM,), dtype, args.input_scale),
        "q": randn((1, NUM_TOKENS, NUM_HEADS, HEAD_DIM), dtype, args.input_scale),
        "k": randn((1, NUM_TOKENS, NUM_HEADS, HEAD_DIM), dtype, args.input_scale),
        "v": randn((1, NUM_TOKENS, NUM_HEADS, HEAD_DIM), dtype, args.input_scale),
        "b": randn((1, NUM_TOKENS, NUM_HEADS), dtype, args.input_scale),
        "initial_state": randn(
            (args.num_cache_lines, NUM_HEADS, HEAD_DIM, HEAD_DIM),
            torch.float32,
            args.state_scale,
        ),
        "initial_state_indices": state_indices,
        "cu_seqlens": torch.arange(
            0,
            NUM_TOKENS + 1,
            CACHE_STEPS,
            device="cuda",
            dtype=torch.int32,
        ).contiguous(),
        "intermediate_states_buffer": torch.empty(
            (args.num_cache_lines, CACHE_STEPS, NUM_HEADS, HEAD_DIM, HEAD_DIM),
            device="cuda",
            dtype=torch.float32,
        ),
        "intermediate_state_indices": state_indices.clone(),
        "retrieve_parent_token": parent.contiguous(),
    }


def clone_for_run(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cloned = dict(tensors)
    cloned["initial_state"] = tensors["initial_state"].clone()
    cloned["intermediate_states_buffer"] = torch.empty_like(
        tensors["intermediate_states_buffer"]
    )
    return cloned


def call_hip(args: argparse.Namespace, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    return aiter.sglang_fused_sigmoid_gating_delta_rule_update(
        A_log=tensors["A_log"],
        a=tensors["a"],
        dt_bias=tensors["dt_bias"],
        softplus_beta=args.softplus_beta,
        softplus_threshold=args.softplus_threshold,
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        b=tensors["b"],
        initial_state_source=tensors["initial_state"],
        initial_state_indices=tensors["initial_state_indices"],
        scale=args.scale,
        use_qk_l2norm_in_kernel=args.qk_l2norm,
        cu_seqlens=tensors["cu_seqlens"],
        is_kda=True,
        # lower_bound=args.lower_bound,
        disable_state_update=True,
        intermediate_states_buffer=tensors["intermediate_states_buffer"],
        intermediate_state_indices=tensors["intermediate_state_indices"],
        cache_steps=CACHE_STEPS,
        retrieve_parent_token=tensors["retrieve_parent_token"],
    )


def call_triton(args: argparse.Namespace, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    return triton_sglang_fused_sigmoid_gating_delta_rule_update(
        A_log=tensors["A_log"],
        a=tensors["a"],
        dt_bias=tensors["dt_bias"],
        softplus_beta=args.softplus_beta,
        softplus_threshold=args.softplus_threshold,
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        b=tensors["b"],
        initial_state_source=tensors["initial_state"],
        initial_state_indices=tensors["initial_state_indices"],
        scale=args.scale,
        use_qk_l2norm_in_kernel=args.qk_l2norm,
        cu_seqlens=tensors["cu_seqlens"],
        is_kda=True,
        disable_state_update=True,
        intermediate_states_buffer=tensors["intermediate_states_buffer"],
        intermediate_state_indices=tensors["intermediate_state_indices"],
        cache_steps=CACHE_STEPS,
        retrieve_parent_token=tensors["retrieve_parent_token"],
    )


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def kernel_io_payload_bytes(tensors: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Estimate the explicit global-memory I/O payload of one Triton call.

    This is not a profiler-derived HBM byte count.  It counts the tensor payload
    passed to the kernel (reads) plus the output and selected cache rows written
    by the kernel, then divides that total by the Triton benchmark time.
    """
    read_names = (
        "A_log",
        "a",
        "dt_bias",
        "q",
        "k",
        "v",
        "b",
        "initial_state",
        "initial_state_indices",
        "cu_seqlens",
        "intermediate_state_indices",
        "retrieve_parent_token",
    )
    read_bytes = sum(tensor_nbytes(tensors[name]) for name in read_names)

    # target-verify writes the cache rows selected by intermediate_state_indices.
    # The sequential parent chain [0, 0, 1, 2, 3, 4] does not add a separate
    # parent-state payload term to this interface-level estimate.
    cache_rows = selected_cache_rows(tensors)
    cache_row_bytes = (
        CACHE_STEPS * NUM_HEADS * HEAD_DIM * HEAD_DIM
        * tensors["intermediate_states_buffer"].element_size()
    )
    cache_write_bytes = cache_rows.numel() * cache_row_bytes
    output_write_bytes = tensor_nbytes(tensors["v"])
    write_bytes = output_write_bytes + cache_write_bytes
    return {
        "read": read_bytes,
        "write": write_bytes,
        "total": read_bytes + write_bytes,
        "output_write": output_write_bytes,
        "cache_write": cache_write_bytes,
    }


def gbps(num_bytes: int, ms: float) -> float:
    return num_bytes / (ms * 1e-3) / 1e9


def selected_cache_rows(tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.unique(tensors["intermediate_state_indices"].to(torch.long))


def verify(args: argparse.Namespace, tensors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    hip_tensors = clone_for_run(tensors)
    triton_tensors = clone_for_run(tensors)
    hip_out = call_hip(args, hip_tensors)
    triton_out = call_triton(args, triton_tensors)
    torch.cuda.synchronize()

    rows = selected_cache_rows(tensors)
    torch.testing.assert_close(
        hip_out.float(),
        triton_out.float(),
        rtol=args.rtol,
        atol=args.atol,
        equal_nan=True,
    )
    torch.testing.assert_close(hip_tensors["initial_state"], tensors["initial_state"], rtol=0, atol=0)
    torch.testing.assert_close(
        triton_tensors["initial_state"], tensors["initial_state"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        hip_tensors["intermediate_states_buffer"].index_select(0, rows),
        triton_tensors["intermediate_states_buffer"].index_select(0, rows),
        rtol=args.rtol,
        atol=args.atol,
        equal_nan=True,
    )
    return {
        "verify_pass": True,
        "out_max_abs": float((hip_out.float() - triton_out.float()).abs().max().item()),
    }


def shape_dict(tensors: Dict[str, torch.Tensor]) -> Dict[str, list]:
    shapes = {name: list(tensor.shape) for name, tensor in tensors.items()}
    shapes["o"] = list(tensors["v"].shape)
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(
        "Benchmark fixed SGLang target-verify KDA HIP path against Triton"
    )
    parser.add_argument("--num-cache-lines", type=int, default=8)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--input-scale", type=float, default=0.2)
    parser.add_argument("--state-scale", type=float, default=0.02)
    parser.add_argument("--softplus-beta", type=float, default=1.0)
    parser.add_argument("--softplus-threshold", type=float, default=20.0)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--qk-l2norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--kernel-bv",
        choices=(0, 16, 32),
        type=int,
        default=0,
        help="HIP BV selector: 0=auto host dispatch, 16/32=fixed launch family.",
    )
    parser.add_argument(
        "--kernel-num-warps",
        type=int,
        default=1,
        help="Accepted for API parity; HIP launch selection is currently controlled by --kernel-bv.",
    )
    parser.add_argument("--triton-bv", choices=(16, 32, 64), type=int, default=32)
    parser.add_argument("--triton-num-warps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-triton", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--rtol", type=float, default=6e-2)
    parser.add_argument("--atol", type=float, default=6e-2)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/HIP CUDA-compatible device is required")

    tensors = make_inputs(args)
    correctness = None
    if not args.no_verify and not args.skip_triton:
        correctness = verify(args, tensors)

    hip_tensors = clone_for_run(tensors)
    triton_tensors = clone_for_run(tensors)
    call_hip(args, hip_tensors)
    if not args.skip_triton:
        call_triton(args, triton_tensors)
    torch.cuda.synchronize()

    hip_ms = float(
        triton.testing.do_bench(
            lambda: call_hip(args, hip_tensors), warmup=args.warmup, rep=args.rep
        )
    )
    triton_ms = None
    if not args.skip_triton:
        triton_ms = float(
            triton.testing.do_bench(
                lambda: call_triton(args, triton_tensors),
                warmup=args.warmup,
                rep=args.rep,
            )
        )

    io_payload = kernel_io_payload_bytes(tensors)
    result = {
        "case": "sglang_target_verify_kda_8x6_h64_k128",
        "shapes": shape_dict(tensors),
        "config": {
            "dtype": args.dtype,
            "num_cache_lines": args.num_cache_lines,
            "is_kda": True,
            "target_verify": True,
            "disable_state_update": True,
            "kernel_bv": args.kernel_bv,
            "kernel_num_warps": args.kernel_num_warps,
            "triton_bv": args.triton_bv,
            "triton_num_warps": args.triton_num_warps,
        },
        "bench": {
            "hip_ms": hip_ms,
            "triton_ms": triton_ms,
            "speedup_vs_triton": triton_ms / hip_ms if triton_ms is not None else None,
            "kernel_io_payload_read_bytes": io_payload["read"],
            "kernel_io_payload_write_bytes": io_payload["write"],
            "kernel_io_payload_total_bytes": io_payload["total"],
            "kernel_io_payload_output_write_bytes": io_payload["output_write"],
            "kernel_io_payload_cache_write_bytes": io_payload["cache_write"],
            "hip_io_payload_gbps": gbps(io_payload["total"], hip_ms),
            "triton_io_payload_gbps": (
                gbps(io_payload["total"], triton_ms) if triton_ms is not None else None
            ),
        },
        "correctness": correctness,
    }

    print("=" * 110)
    print(
        "SGLang target-verify KDA | N=8 T=6 H=HV=64 K=V=128 "
        "vs Triton | cache_lines={} dtype={} HIP_BV={} HIP_warps={} "
        "Triton_BV={} Triton_warps={}".format(
            args.num_cache_lines,
            args.dtype,
            args.kernel_bv,
            args.kernel_num_warps,
            args.triton_bv,
            args.triton_num_warps,
        )
    )
    print("A_log={} a={} dt_bias={} q={} o={} state={} cache={} parent={}".format(
        tuple(tensors["A_log"].shape),
        tuple(tensors["a"].shape),
        tuple(tensors["dt_bias"].shape),
        tuple(tensors["q"].shape),
        tuple(tensors["v"].shape),
        tuple(tensors["initial_state"].shape),
        tuple(tensors["intermediate_states_buffer"].shape),
        tuple(tensors["retrieve_parent_token"].shape),
    ))
    print("-" * 110)
    print(
        "Kernel I/O payload: read={} bytes, write={} bytes (o={}, cache={}), total={} bytes".format(
            io_payload["read"],
            io_payload["write"],
            io_payload["output_write"],
            io_payload["cache_write"],
            io_payload["total"],
        )
    )
    print("HIP    : {:.4f} ms, {:.2f} us, {:.2f} I/O-payload GB/s".format(
        hip_ms, hip_ms * 1000.0, gbps(io_payload["total"], hip_ms)
    ))
    if triton_ms is not None:
        print(
            "Triton : {:.4f} ms, {:.2f} us, {:.2f} I/O-payload GB/s, HIP speedup {:.4f}x".format(
                triton_ms,
                triton_ms * 1000.0,
                gbps(io_payload["total"], triton_ms),
                triton_ms / hip_ms,
            )
        )
    if correctness is not None:
        print("Verify : {}".format(correctness))
    print("=" * 110)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
