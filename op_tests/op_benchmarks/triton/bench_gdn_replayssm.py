# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Graph timing for compact GDN verify; optional SGLang 0.5.22 reference.

The checkpoint and history are read-only and speculative rows are overwritten
on each call. Timing therefore needs no full-state reset copy. This is verify
timing, excluding checkpoint commit, cursor updates, model work and networking.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

import aiter.ops.triton.fla.gdn_replayssm as gdn


class CaptureKernel:
    def __init__(self, kernel):
        self.kernel = kernel
        self.compiled = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.compiled = self.kernel[grid](*args, **kwargs)
            return self.compiled
        return launch


def graph(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        fn()
    for _ in range(3):
        capture.replay()
    torch.cuda.synchronize()
    return capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--history', type=int, default=8)
    parser.add_argument('--sglang-reference', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dump-ir', type=Path)
    args = parser.parse_args()
    if not 0 <= args.history <= 12:
        parser.error('history must leave space for the four draft tokens')
    torch.manual_seed(183)
    B, H, HV, D, L = args.batch, 16, 48, 128, 16
    slots = B + 1

    def random(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device='cuda', dtype=dtype) * .1

    q, k = random((B * 4, H, D)), random((B * 4, H, D))
    v = random((B * 4, HV, D))
    a, b = random((B * 4, HV)), random((B * 4, HV))
    al, bias = random((HV,), torch.float32), random((HV,), torch.float32)
    checkpoint = random((slots, HV, D, D), torch.float32)
    dc, kc = random((slots, HV, L, D)), random((slots, H, L, D))
    gc = -torch.rand((slots, HV, L), device='cuda') * .1
    rv, rk = torch.zeros_like(dc), torch.zeros_like(kc)
    ids = torch.arange(1, slots, device='cuda', dtype=torch.int32)
    cu = torch.arange(B + 1, device='cuda', dtype=torch.int32) * 4
    pos = torch.full((slots,), args.history, device='cuda', dtype=torch.int32)
    base = torch.arange(slots, device='cuda', dtype=torch.int32)
    flush = torch.zeros(slots, device='cuda', dtype=torch.int8)
    output = torch.empty_like(v)
    inputs = [q, k, v, a, b, al, bias, output, checkpoint, dc, kc, gc, rv, rk,
              None, cu, ids, ids, pos, base, flush, D ** -.5, L, 4, True]
    kernels = {}
    original = gdn._gdn_replayssm_recurrent_verify
    captured = CaptureKernel(original)
    try:
        gdn._gdn_replayssm_recurrent_verify = captured
        gdn.gdn_replayssm_verify(*inputs, False)
        kernels['aiter'] = captured.compiled
    finally:
        gdn._gdn_replayssm_recurrent_verify = original
    expected = output.clone()
    functions = {'aiter': lambda: gdn.gdn_replayssm_verify(*inputs, False)}
    if args.sglang_reference:
        import sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode as reference
        kernel = reference.gdn_replayssm_spec_circular_kernel
        captured = CaptureKernel(kernel)
        try:
            reference.gdn_replayssm_spec_circular_kernel = captured
            reference._launch_gdn_spec(*inputs, False, 64, 1, 2, 2, 4, 0, 'ieee')
            kernels['sglang'] = captured.compiled
        finally:
            reference.gdn_replayssm_spec_circular_kernel = kernel
        torch.testing.assert_close(output, expected, atol=2e-4, rtol=2e-2)
        functions['sglang'] = lambda: reference._launch_gdn_spec(
            *inputs, False, 64, 1, 2, 2, 4, 0, 'ieee'
        )
    graphs = {name: graph(fn) for name, fn in functions.items()}
    samples = {name: [] for name in graphs}
    for iteration in range(7):
        order = list(graphs) if iteration % 2 == 0 else list(reversed(graphs))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 100)
    result = {
        'batch': B, 'history': args.history, 'heads': H, 'value_heads': HV,
        'arch': torch.cuda.get_device_properties(q.device).gcnArchName,
        'torch': torch.__version__, 'scope': 'non-flush verify only; no reset copy',
        'timing': {name: {'median_us': statistics.median(times), 'samples_us': times}
                   for name, times in samples.items()},
        'resources': {},
    }
    for name, kernel in kernels.items():
        result['resources'][name] = {
            'registers': kernel.n_regs, 'spills': kernel.n_spills,
            'shared_bytes': kernel.metadata.shared,
        }
        if args.dump_ir:
            args.dump_ir.mkdir(parents=True, exist_ok=True)
            for kind in ['ttir', 'ttgir', 'amdgcn']:
                if kind in kernel.asm:
                    (args.dump_ir / (name + '.' + kind)).write_text(kernel.asm[kind])
    if 'sglang' in samples:
        result['speedup'] = statistics.median(
            samples['sglang']) / statistics.median(samples['aiter'])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
