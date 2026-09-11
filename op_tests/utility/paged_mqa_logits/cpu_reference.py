# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""PMD-friendly validation: host preparation/reference and only the Opus DUT on GPU."""

import importlib
import json
from pathlib import Path

import torch

from op_tests.test_paged_mqa_logits import make_case, reference, check_output


def run_cpu_reference(report_path, kernel_id=None):
    from aiter import paged_mqa_logits
    module = importlib.import_module("aiter.ops.opus.paged_mqa_logits")
    torch.set_num_threads(1)
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    assert arch in ("gfx938", "gfx946"), arch
    cases = [(f"r{r}_h{h}", (1, r, h, 67), None, False)
             for r in (1, 2, 4) for h in (32, 64)]
    cases += [
        ("min", (1, 1, 32, 1), [1], True),
        ("min_mtp", (1, 4, 64, 4), [4], True),
        ("n63", (1, 2, 32, 63), [63], False),
        ("n64", (1, 4, 64, 64), [64], False),
        ("n65", (2, 2, 32, 65), [63, 65], False),
        ("n128", (2, 4, 64, 128), [65, 128], False),
        ("n257", (2, 4, 64, 257), [129, 257], True),
        ("far_tail", (2, 2, 32, 1025), [2, 67], False),
        ("short_table", (1, 2, 64, 257), [73], False),
        ("scales", (2, 2, 32, 129), [67, 129], False),
        ("zero_weights", (1, 4, 64, 67), None, False),
        ("fp8_limits", (1, 2, 32, 65), None, False),
        ("repeat_pages", (2, 4, 64, 129), [73, 129], True),
        ("stream", (1, 2, 32, 65), None, True),
    ]
    records = []
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report_path.write_text(json.dumps({"arch": arch, "torch": torch.__version__,
            "kernel_id": kernel_id, "reference": "CPU FP32", "cases": records}, indent=2))

    for index, (name, shape, lengths, structured) in enumerate(cases):
        args = list(make_case(*shape, seed=946+index, lengths=lengths,
                              structured=structured, independent_cache=True, device="cpu"))
        q, cache, weights, context, table, n = args
        if name == "short_table":
            args[4] = table[:, :73].contiguous()
        if name == "scales":
            scales = 2.0 ** torch.linspace(-8, 8, cache.shape[0])
            cache.view(-1, 132)[:, 128:].copy_(scales[:, None].view(torch.uint8))
        if name == "zero_weights":
            weights.zero_()
        if name == "fp8_limits":
            args[0] = torch.full(shape[:3]+(128,), 448.0).to(torch.float8_e4m3fn)
            cache.view(-1, 132)[:, :128].copy_(torch.full((cache.shape[0], 128), 448.0)
                                             .to(torch.float8_e4m3fn).view(torch.uint8))
        if name == "repeat_pages":
            table[:, ::2] = table[:, :1]
        ref, valid = reference(args)
        dev = [t.to("cuda") for t in args[:5]] + [n]
        # Canary surrounds a contiguous output view; all writes are checked on CPU.
        backing_cpu = torch.full((shape[0]*shape[1]*n+32,), -1234567.0)
        backing = backing_cpu.to("cuda")
        out = backing[16:-16].view(shape[0]*shape[1], n)
        stream = torch.cuda.Stream() if name == "stream" else torch.cuda.current_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            returned = paged_mqa_logits(*dev, out=out, kernelId=kernel_id)
        stream.synchronize()
        assert returned is out
        host = out.cpu()
        error = check_output(host, ref, valid, structured)
        whole = backing.cpu()
        assert torch.equal(whole[:16], backing_cpu[:16])
        assert torch.equal(whole[-16:], backing_cpu[-16:])
        assert not (host == -1234567.0).any(), "output sentinel survived"
        for cpu, gpu in zip(args[:5], dev[:5]):
            assert torch.equal(cpu.view(torch.uint8), gpu.cpu().view(torch.uint8)), "input mutated"
        resolved = module._resolve_paged_mqa_kernel(kernel_id, rows=shape[0]*shape[1],
                                                   max_len=n, arch=arch)
        if arch == "gfx946":
            assert resolved == (4 if kernel_id is None else kernel_id)
        if name in ("min", "n65", "stream"):
            paged_mqa_logits(*dev, out=out, clean_logits=False, kernelId=kernel_id)
            torch.cuda.synchronize()
            assert torch.equal(out.cpu(), host), "repeat/clean=False changed the result"
        records.append({"name": name, "shape": list(shape), "lengths": context.tolist(),
                        "resolved_id": resolved, "structured": structured, "seed": 946+index,
                        "status": "PASS", **error})
        save()
        print(f"CPU_REF_PASS {name} id={resolved} max_abs={error['max_abs']}", flush=True)

    # Host metadata errors must be rejected before entering the JIT/DUT.
    checks = 0
    def rejects(call):
        nonlocal checks
        try:
            call()
        except (ValueError, NotImplementedError, RuntimeError):
            checks += 1
        else:
            raise AssertionError("invalid API call was accepted")
    rejects(lambda: paged_mqa_logits(*dev, kernelId=True))
    rejects(lambda: paged_mqa_logits(*dev, kernelId=99))
    rejects(lambda: paged_mqa_logits(*dev, kernelId=0 if arch == "gfx946" else 4))
    rejects(lambda: paged_mqa_logits(*dev, clean_logits=1))
    rejects(lambda: paged_mqa_logits(*dev[:-1], 0))
    rejects(lambda: paged_mqa_logits(dev[0].view(torch.uint8), *dev[1:]))
    rejects(lambda: paged_mqa_logits(dev[0], dev[1], dev[2][:, ::2], *dev[3:]))
    rejects(lambda: paged_mqa_logits(*dev, out=out[:, ::2]))
    rejects(lambda: paged_mqa_logits(args[0], *dev[1:]))
    # Direct pybind bypass must still enforce architecture/ID pairing.
    rejects(lambda: module._paged_mqa_logits_opus(*dev[:5], out, n,
                                                 0 if arch == "gfx946" else 4))
    records.append({"api_rejections": checks, "status": "PASS"})
    save()
    print(f"CPU_REFERENCE_PASS cases={len(cases)} rejections={checks}", flush=True)
