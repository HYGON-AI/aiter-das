# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""8-bit native ReduceScatter: CPU numeric oracle, changing-input Graph, RCCL.

Run via torchrun. Correctness does not initialize RCCL; --bench adds it.
INT8 uses INT32/modulo-256; FP8 uses ordered FP32 SUM then non-saturating RNE.
"""
import argparse
from datetime import timedelta
import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import statistics
import time


@lru_cache(None)
def fp8_table(dtype):
    """Independent format oracle, unaffected by patched Torch saturating casts."""
    name = str(dtype)
    if name not in ('torch.float8_e4m3fn', 'torch.float8_e5m2'):
        raise ValueError(f'unsupported Hygon FP8 dtype: {dtype}')
    mb = 3 if name == 'torch.float8_e4m3fn' else 2
    bias = 7 if mb == 3 else 15
    last = 126 if mb == 3 else 123
    values = []
    for code in range(256):
        magnitude = code & 127
        exponent, mantissa = divmod(magnitude, 1 << mb)
        if mb == 3 and magnitude == 127:
            value = float('nan')
        elif mb == 2 and exponent == 31:
            value = float('nan') if mantissa else float('inf')
        else:
            value = math.ldexp(mantissa if exponent == 0 else (1 << mb) + mantissa,
                               (1 if exponent == 0 else exponent) - bias - mb)
        values.append(-value if code & 128 else value)
    table = torch.tensor(values, dtype=torch.float32)
    finite = table[:last + 1]
    # Include the next (overflow) rounding bin, then resolve exact ties to even.
    next_value = finite[-1] + (finite[-1] - finite[-2])
    boundaries = (finite + torch.cat((finite[1:], next_value.reshape(1)))) / 2
    return table, boundaries, last


def fp8_encode_oracle(value, dtype):
    _, boundaries, last = fp8_table(dtype)
    magnitude = value.abs().contiguous()
    code = torch.bucketize(magnitude, boundaries)
    at_tie = magnitude == boundaries[code.clamp(max=last)]
    code += (at_tie & (code % 2 == 1)).to(code.dtype)
    sign = torch.signbit(value).to(torch.int64) * 128
    overflow_code = 124 if dtype == torch.float8_e5m2 else 127
    encoded = torch.where(code > last, overflow_code, code)
    encoded |= sign
    encoded = torch.where(value.isnan(), 127, encoded)
    return encoded.to(torch.uint8).view(dtype)


def pattern(shape, dtype, rank, iteration=0, pairs=False):
    index = torch.arange(math.prod(shape), dtype=torch.int64)
    if pairs:
        codes = index // 256 if rank == 0 else index % 256 if rank == 1 else torch.zeros_like(index)
    else:
        codes = (index * (2 * iteration + 1) + rank * 37 + iteration * 19) % 256
    return codes.to(torch.uint8).reshape(shape).view(dtype)


def reference(shape, dtype, rank, world, dim, iteration=0, pairs=False):
    acc = torch.zeros(shape, dtype=torch.int32 if dtype == torch.int8 else torch.float32)
    for source in range(world):
        value = pattern(shape, dtype, source, iteration, pairs)
        acc.add_(value.to(acc.dtype) if dtype == torch.int8 else fp8_table(dtype)[0][value.view(torch.uint8).long()])
    shard = acc.chunk(world, dim=dim)[rank].contiguous()
    return shard.to(dtype) if dtype == torch.int8 else fp8_encode_oracle(shard, dtype)


def exact(actual, expected):
    actual = actual.cpu()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.dtype == torch.int8:
        mask = torch.ones_like(actual, dtype=torch.bool)
    else:
        nan_a, nan_b = actual.float().isnan(), expected.float().isnan()
        bad_nan = nan_a != nan_b
        assert not bad_nan.any(), (f'NaN classification mismatch count={bad_nan.sum().item()} '
                                   f'actual={actual.view(torch.uint8)[bad_nan][:16]} '
                                   f'expected={expected.view(torch.uint8)[bad_nan][:16]}')
        mask = ~nan_a
    a, b = actual.view(torch.uint8), expected.view(torch.uint8)
    if actual.element_size() > 1:
        mask = mask.repeat_interleave(actual.element_size(), dim=-1)
    bad = mask & (a != b)
    assert not bad.any(), f'byte mismatches={bad.sum().item()} actual={a[bad][:16]} expected={b[bad][:16]}'


def expect_error(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('expected validation failure')


def correctness(ca, args, rank, world, device):
    from aiter.ops import custom_all_reduce as ops
    from aiter.dist.device_communicators.communicator_cuda import CudaCommunicator
    public = object.__new__(CudaCommunicator)
    public.ca_comm, public.world_size = ca, world
    cases = 0
    shapes = [((world * 16,), 0), ((world * 4, 64), 0),
              ((16, world * 32), -1), ((16, world * 3), -1),
              ((2, world * 3, 32), 1), ((16, world * 3, 3), 1),
              ((2, 2, world * 3, 16), -2),
              ((16, 512), 0), ((16, 512), -1)]
    for dtype in [torch.int8, *FP8_DTYPES, torch.float16, torch.bfloat16, torch.float32]:
        for shape, dim in shapes:
            # Legacy floats use finite integer values for exact reduction.
            byte_dtype = dtype == torch.int8 or dtype in FP8_DTYPES
            def value(source, iteration):
                if byte_dtype:
                    return pattern(shape, dtype, source, iteration)
                return torch.full(shape, float(source + iteration), dtype=dtype)
            def wanted(iteration):
                if byte_dtype:
                    return reference(shape, dtype, rank, world, dim, iteration)
                return sum(value(r, iteration).float() for r in range(world)).chunk(world, dim)[rank].contiguous().to(dtype)
            inp = value(rank, 0).to(device)
            out = torch.empty_like(wanted(0), device=device)
            assert ca.should_custom_rs(inp, dim), (dtype, shape, dim)
            ca.reduce_scatter(inp, out, dim)
            exact(out, wanted(0))
            ca.reduce_scatter(inp, out, dim, registered=True)
            exact(out, wanted(0))
            ca.custom_reduce_scatter(inp, out, dim)
            exact(out, wanted(0))
            exact(public.reduce_scatter(inp, dim), wanted(0))
            with ca.capture():
                ca.custom_reduce_scatter(inp, out, dim)  # warmup must produce data
                exact(out, wanted(0))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    ca.custom_reduce_scatter(inp, out, dim)
                    inp.view(torch.uint8).fill_(0xA5)  # immediate producer reuse
            for iteration in range(args.repeats):
                inp.view(torch.uint8).copy_(value(rank, iteration).view(torch.uint8))
                if rank == 0 and iteration % 5 == 0:
                    time.sleep(.001)
                graph.replay()
                try:
                    exact(out, wanted(iteration))
                except AssertionError as exc:
                    raise AssertionError(f'dtype={dtype} shape={shape} dim={dim} rank={rank} replay={iteration}: {exc}') from exc
            del graph
            cases += 1
        if dtype == torch.int8 or dtype in FP8_DTYPES:
            # Exhaust every pair of 8-bit encodings, including special values,
            # cancellation, overflow, subnormals and halfway rounding cases.
            shape = (65536,)
            inp = pattern(shape, dtype, rank, pairs=True).to(device)
            want = reference(shape, dtype, rank, world, 0, pairs=True)
            out = torch.empty_like(want, device=device)
            ca.reduce_scatter(inp, out)
            exact(out, want)
            if dtype in (torch.int8, torch.float8_e4m3fn):
                ops.reduce_scatter(ca._ptr, inp, out, ca._pool['input'].data_ptr, ca._pool['input'].max_size)
                exact(out, want)
                ops.reduce_scatter_dim(ca._ptr, inp, out, 0, 0, inp.numel(), 0,
                                       ca._pool['input'].data_ptr, ca._pool['input'].max_size)
                exact(out, want)
            # Copy-in accepts unaligned input; output alignment is checked.
            backing = torch.empty(inp.numel() + 1, dtype=torch.uint8, device=device)
            offset = backing[1:].view(dtype)
            offset.view(torch.uint8).copy_(inp.view(torch.uint8))
            ca.reduce_scatter(offset, out)
            exact(out, want)
            bad = torch.empty(out.numel() + 1, dtype=torch.uint8, device=device)[1:].view(dtype)
            expect_error(lambda: ca.reduce_scatter(inp, bad))
            expect_error(lambda: ca.reduce_scatter(inp, inp[:out.numel()]))
            expect_error(lambda: ca.reduce_scatter(inp, out, dim=2))
            cases += 2
        if rank == 0:
            print('RS_DTYPE_PASS', str(dtype), flush=True)
    for dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        inp = torch.zeros(world * 16, dtype=dtype, device=device)
        out = torch.full((16,), 11, dtype=dtype, device=device)
        assert not ca.should_custom_rs(inp)
        assert ca.custom_reduce_scatter(inp, out) is None
        assert (out == 11).all().item()
        expect_error(lambda: ca.reduce_scatter(inp, out))
    # Reject dimensions beyond the shared Tensor ABI before native dispatch.
    for dtype in (torch.int8, *FP8_DTYPES, torch.float16, torch.bfloat16, torch.float32):
        shape = (world * 16,) + (1,) * 7 + (world,)
        inp = torch.zeros(shape, dtype=torch.uint8, device=device).to(dtype)
        for dim in (0, -1):
            output_shape = list(shape)
            output_shape[dim] //= world
            out = torch.empty(output_shape, dtype=dtype, device=device)
            out.view(torch.uint8).fill_(0xA5)
            assert not ca.should_custom_rs(inp, dim)
            assert ca.custom_reduce_scatter(inp, out, dim) is None
            expect_error(lambda: ca.reduce_scatter(inp, out, dim))
            assert (out.view(torch.uint8) == 0xA5).all().item()
            cases += 1
    inp = torch.zeros((16, world * 3), dtype=torch.int8, device=device)
    assert not ca.should_custom_rs(inp.t(), 0)
    assert not ca.should_custom_rs(inp.cpu(), 1)
    assert not ca.should_custom_rs(inp[:, :0], 1)
    assert not ca.should_custom_rs(torch.zeros(17, dtype=torch.int8, device=device), 0)
    old = ca.max_size
    ca.max_size = 16
    assert not ca.should_custom_rs(inp, 1)
    ca.max_size = old
    return cases + 4


def summarize(values):
    return dict(samples_us=values, median_us=statistics.median(values),
                p95_us=sorted(values)[math.ceil(len(values) * .95) - 1],
                cv_pct=100 * statistics.pstdev(values) / statistics.mean(values))


def benchmark(ca, args, rank, world, device):
    from aiter.dist.device_communicators.communicator_pynccl import PyNcclCommunicator
    rccl = PyNcclCommunicator(dist.group.WORLD, device)
    assert not rccl.disabled
    rows = []
    for dtype in (torch.int8, torch.float8_e4m3fn):
        for size in map(int, args.sizes.split(',')):
            assert size > 0 and size % 256 == 0
            shape = (16, size // 16)
            inp = ((torch.arange(size) % 7) - 3).reshape(shape).to(dtype).to(device)
            for dim in (0, 1):
                assert ca.should_custom_rs(inp, dim)
                output_shape = list(shape)
                output_shape[dim] //= world
                out = torch.empty(output_shape, dtype=dtype, device=device)
                rccl_shape = (shape[dim] // world, shape[1 - dim])
                ref_dtype = torch.int8 if dtype == torch.int8 else torch.float32
                reduced = torch.empty(rccl_shape, dtype=ref_dtype, device=device)
                converted = torch.empty(shape, dtype=ref_dtype, device=device)
                ordered = torch.empty((shape[dim], shape[1 - dim]), dtype=ref_dtype, device=device)
                baseline_out = torch.empty_like(out)
                def custom():
                    ca.reduce_scatter(inp, out, dim)
                def baseline():
                    # FP8 RCCL reference includes cast/layout/FP32 communication
                    # and the final cast. It is NOT a native FP8 RCCL collective.
                    if dtype == torch.int8 and dim == 0:
                        rccl.reduce_scatter(baseline_out, inp)
                        return
                    if dtype == torch.int8:
                        ordered.copy_(inp.movedim(dim, 0))
                    else:
                        converted.copy_(inp)
                        if dim != 0:
                            ordered.copy_(converted.movedim(dim, 0))
                    rccl.reduce_scatter(reduced, converted if dtype != torch.int8 and dim == 0 else ordered)
                    baseline_out.copy_(reduced.movedim(0, dim))
                want = (inp.cpu().float() * world).chunk(world, dim)[rank].contiguous().to(dtype)
                custom(); baseline()
                exact(out, want); exact(baseline_out, want)
                for mode in ('eager', 'graph'):
                    functions = dict(custom=custom, rccl=baseline)
                    graphs = {}
                    if mode == 'graph':
                        for key, fn in functions.items():
                            for _ in range(3): fn()
                            torch.cuda.synchronize(); dist.barrier()
                            graph = torch.cuda.CUDAGraph()
                            with ca.capture(), torch.cuda.graph(graph):
                                for _ in range(args.iters): fn()
                            graphs[key] = graph
                        functions = {key:g.replay for key, g in graphs.items()}
                    samples = {key:[] for key in functions}
                    for repeat in range(args.rounds):
                        order = list(functions)
                        if repeat % 2: order.reverse()
                        for key in order:
                            fn = functions[key]
                            for _ in range(3): fn()
                            torch.cuda.synchronize(); dist.barrier()
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(1 if mode == 'graph' else args.iters): fn()
                            end.record(); end.synchronize()
                            elapsed = torch.tensor(start.elapsed_time(end) * 1000 / args.iters)
                            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                            samples[key].append(elapsed.item())
                    row = dict(dtype=str(dtype), bytes=size, dim=dim, mode=mode,
                               rccl_version=rccl.nccl_version,
                               rccl_payload_bytes=size * (1 if dtype == torch.int8 else 4),
                               baseline='RCCL_INT8_SUM' if dtype == torch.int8 else 'cast_FP32_RCCL_SUM_cast_FP8',
                               **{key:summarize(v) for key,v in samples.items()})
                    rows.append(row)
                    if rank == 0: print('RS_BENCH', json.dumps(row), flush=True)
                    del graphs, functions
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--sizes', default='256,4096,8192,32768,262144,1048576,4194304',
                        help='comma-separated input bytes per rank (INT8/FP8 benchmarks)')
    args = parser.parse_args()
    global torch, dist, FP8_DTYPES
    import torch
    import torch.distributed as dist
    from aiter.utility.dtypes import FP8_DTYPES
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    ca = CustomAllreduce(dist.group.WORLD, local, max_size=64 << 20, enable_register_for_capturing=True)
    assert not ca.disabled and ca.transport == 'ipc'
    report = dict(rank=rank, world=world, torch=torch.__version__, hip=torch.version.hip,
                  arch=torch.cuda.get_device_properties(local).gcnArchName,
                  effective_registered=ca.enable_register_for_capturing, arguments=vars(args))
    report['cases'] = correctness(ca, args, rank, world, device)
    report['benchmarks'] = benchmark(ca, args, rank, world, device) if args.bench else []
    torch.cuda.synchronize()
    report['native'] = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = line.split()[-1]
        if 'module_custom_all_reduce' in path and path.endswith('.so') and path not in report['native']:
            report['native'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert report['native']
    report['status'] = 'passed'
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RS_PASS', json.dumps({k:v for k,v in report.items() if k != 'benchmarks'}), flush=True)
    dist.barrier()
    ca.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
