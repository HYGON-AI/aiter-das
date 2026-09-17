# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Paged MQA independent reference and reproducible comparison entry point.

整体用法：在仓库根目录运行 pytest，验证当前架构所有适用路径；仅验证 HCU
优化路径时加 -k TestPagedMQAHCU。该类同时检查默认分配和预分配 out、
S1/S64 页、负权重、非正规数、非对齐视图、随机共享页和 Graph 回放更新。
--cpu-reference 使用 CPU 构造输入和参考值，适合 gfx946 Perf Model；
--bench 是已有的预分配 Graph 基准，功能正确性应先通过再测性能。

From repository root:
  python -B op_tests/test_paged_mqa_logits.py
  python -B op_tests/test_paged_mqa_logits.py --backend baseline --json-out results.json
  python -B op_tests/test_paged_mqa_logits.py --backend opus --kernel-id 1 --json-out opus.json
  python -B op_tests/test_paged_mqa_logits.py --bench --kernel-id 1 --json-out bench.json
  python -B op_tests/test_paged_mqa_logits.py --bench --shape-set models
  python -B -m pytest -q op_tests/test_paged_mqa_logits.py
  python -B -m pytest -q op_tests/test_paged_mqa_logits.py -k TestPagedMQAHCU
  python -B op_tests/test_paged_mqa_logits.py --cpu-reference --kernel-id 4

With no arguments, compare Opus and both Triton baselines against the reference.
JSON defaults to hygon_tmp/paged_mqa_logits/<mode>_<timestamp>.json in this repo.
With --bench, print a paired comparison table after all rounds; JSON retains raw samples.

gfx938 IDs: 0/1 = initial full/wave reduction; 2 = DPP/K32; 3 = DPP/K64 prefetch.
gfx936/gfx938 IDs: 6 = independent requests; 7 = verified two-query K sharing.
Auto selects ID7 for H32/R1, B>=32, cache_tokens<=2*max_len, and
max_len>=4096 on gfx936 or >=16384 on gfx938;
otherwise ID6. IDs 0-3 remain explicit gfx938 comparisons.
gfx946 IDs: 4 = aligned LDS/K32 (Auto); 5 = aligned LDS/K64 prefetch.
Use --cpu-reference on gfx946; the Triton comparison mode requires gfx938.
Add --cpu-graph only when the target runtime supports HIP Graph replay.
Pytest runs all applicable cases from this file: gfx936/gfx938 use
TestPagedMQAHCU for ID6/7, S1/S64, sharing fallback, FP8 limits, offset
storage and graph replay. Public API checks also run on both architectures.
gfx946 runs the CPU-reference matrix for Auto/ID4/ID5; HCU-only cases skip.
All timing is a preallocated GPU Graph sequence including required cleanup.

Triton compatibility is owned by utility/paged_mqa_logits/paged_mqa_logits_triton_baseline.py.
"""

import argparse
import importlib
import hashlib
import json
from datetime import datetime
from pathlib import Path
import statistics
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from op_tests.utility.paged_mqa_logits.paged_mqa_logits_triton_baseline import prepare_baseline


PERF_CASES = [(1,1,32,256), (2,2,32,512), (2,4,32,1024), (1,4,64,256),
              (4,2,64,4096), (1,1,64,4096), (1,2,64,16384),
              (4,1,32,16384), (1,1,64,65536), (2,4,32,4096)]

# Rank-local indexer shapes: GLM-5.1 H32 and DeepSeek-V4-Flash H64, D128.
# Both indexers replicate heads across TP. V4 lengths below are C4 candidate
# counts (original context / 4), not raw context or top-k. These are synthetic
# decode workloads in this operator's S=1 FP8 contract, not serving traces.
MODEL_PERF_CASES = [
    (b,1,32,n) for b,n in [(1,4096),(16,4096),(64,4096),(1,32768),(4,32768),
                           (16,32768),(64,32768),(1,131072),(4,131072),(1,202752),(4,202752)]
] + [
    (b,1,64,n) for b,n in [(1,2048),(16,2048),(64,2048),(1,8192),(4,8192),
                           (4,32768),(16,32768),(64,32768),(1,131072),(4,131072),(1,262144)]
]


def make_case(b, r, h, n, seed=100, lengths=None, structured=False, independent_cache=False,
              device="cuda"):
    """Generate canonical byte cache without importing any Triton test helper."""
    torch.manual_seed(seed)
    pages = b*n if independent_cache else n
    qf = torch.randn(b, r, h, 128, device=device)*0.25
    kf = torch.randn(pages, 128, device=device)*0.25
    if structured:
        qf.zero_()
        for head in range(h):
            qf[:, :, head, head % 128] = 1 + (head % 3)
        kf = ((torch.arange(pages, device=device)[:, None]*3
               + torch.arange(128, device=device)[None, :]) % 9-4).float()
    q = qf.to(torch.float8_e4m3fn)
    k = kf.to(torch.float8_e4m3fn)
    scale = 2 ** (torch.rand(pages, device=device)*4-2)
    if structured:
        scale.fill_(0.5)
    cache = torch.empty((pages, 1, 1, 132), dtype=torch.uint8, device=device)
    packed = cache.view(pages, 132)
    packed[:, :128].copy_(k.view(torch.uint8))
    packed[:, 128:].copy_(scale[:, None].view(torch.uint8))
    weights = torch.randn(b*r, h, device=device)
    weights[:, ::7] = 0
    if structured:
        weights.copy_(((torch.arange(h, device=device) % 5)-2).float()[None, :])
    if lengths is None:
        lengths = [max(r, n-3-i*max(1, n//(b+2))) for i in range(b)]
    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    if independent_cache:
        table = torch.stack([torch.randperm(n, dtype=torch.int32, device=device)+i*n for i in range(b)])
    else:
        table = torch.randint(n, (b, n), dtype=torch.int32, device=device)
    return (q, cache, weights, context, table, n)


def reference(args):
    q, cache, weights, context, table, n = args
    b, r, h, d = q.shape
    packed = cache.view(-1, d+4)
    # Independent byte interpretation and page gather, FP32 arithmetic.
    key_values = packed[:, :d].view(torch.float8_e4m3fn).float()
    scales = packed[:, d:].contiguous().view(torch.float32).reshape(-1)
    out = torch.full((b*r, n), -float("inf"), device=q.device)
    valid = torch.zeros((b*r, n), dtype=torch.bool, device=q.device)
    for ib, length in enumerate(context.cpu().tolist()):
        pages = table[ib, :length].long()
        keys = key_values[pages] * scales[pages, None]
        scores = q[ib].float() @ keys.T
        scores = (scores.relu()*weights[ib*r:(ib+1)*r, :, None]).sum(dim=1)
        for ir in range(r):
            end = length-r+ir+1
            out[ib*r+ir, :end] = scores[ir, :end]
            valid[ib*r+ir, :end] = True
    return out, valid


def check_output(out, ref, valid, structured=False):
    assert torch.isfinite(out[valid]).all(), "nonfinite valid output"
    assert torch.isneginf(out[~valid]).all(), "invalid output must be -inf"
    tol = 0.0 if structured else 5e-2
    torch.testing.assert_close(out[valid], ref[valid], rtol=tol, atol=tol)
    error = out[valid]-ref[valid]
    return {"max_abs": error.abs().max().item(),
            "relative_l2": (error.norm()/ref[valid].norm().clamp_min(1e-10)).item()}


def correctness_cases():
    # Original eight parameter combinations plus long/MTP and boundary cases.
    cases = [(b,r,64,n) for b in (1,2) for r in (1,2) for n in (64,128)]
    cases += PERF_CASES
    cases += [(2,2,32,257), (2,4,64,513), (1,1,32,1024), (1,2,64,1024)]
    for i, shape in enumerate(cases):
        lengths = [255,257] if i == 18 else [256,257] if i == 19 else None
        if i == 20:
            lengths = [256]
        if i == 21:
            lengths = [73]
        yield f"case_{i}", make_case(*shape, seed=100+i, lengths=lengths), False
    yield "structured", make_case(2,4,64,257, structured=True), True


def guard_tests(args):
    checks = 0
    q, cache, w, context, table, n = args
    def rejects(values, **options):
        nonlocal checks
        try:
            prepare_baseline(*values, **options)
        except (ValueError, NotImplementedError):
            checks += 1
        else:
            raise AssertionError(f"invalid input accepted: {options}")
    rejects(args, varctx_schedule=torch.tensor([2], device=q.device))
    rejects(args, preshuffle=True)
    rejects((q.float(), cache, w, context, table, n))
    rejects((q, cache, w.double(), context, table, n))
    rejects((q, cache[:, :, :, :128], w, context, table, n))
    bad = context[:, None].expand(-1, q.shape[1]).contiguous()
    rejects((q, cache, w, bad, table, n))
    rejects((q, cache, w, torch.zeros_like(context), table, n))
    bad_table = table.clone()
    bad_table[0, 0] = cache.shape[0]
    rejects((q, cache, w, context, bad_table, n))
    rejects((q, cache, w, context, table[:, ::2], n))
    return checks


def run_correctness(backend, report_path, kernel_id=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    mod = importlib.import_module("aiter.ops.triton.attention.pa_mqa_logits")
    original_gluon = mod.enable_gluon_pa_mqa_logits
    records = []
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for name, args, structured in correctness_cases():
        q, cache, w, context, table, n = args
        ref, valid = reference(args)
        base = prepare_baseline(*args)
        size = ref.numel()
        guarded = torch.full((size+128,), 654321.0, device=q.device)
        out = guarded[64:64+size].view_as(ref)
        rec = {"name": name, "shape": list(q.shape), "max_len": n,
               "lengths": context.cpu().tolist(), "seed": 100 if structured else 100+len(records),
               "metadata": base.metadata, "results": {}}
        funcs = {"fused": base.run_fused, "stage1_sum": base.run_stage1_sum}
        if backend == "opus":
            from aiter import paged_mqa_logits
            funcs["opus"] = lambda out: paged_mqa_logits(*args, out=out, kernelId=kernel_id)
        for label, fn in funcs.items():
            out.fill_(float("nan"))
            fn(out)
            torch.cuda.synchronize()
            rec["results"][label] = check_output(out, ref, valid, structured)
            assert (guarded[:64] == 654321.0).all()
            assert (guarded[-64:] == 654321.0).all()
        assert base.check_workspace_canaries()
        assert mod.enable_gluon_pa_mqa_logits == original_gluon
        if q.shape[1] > 1:
            canonical = (context[:, None]-q.shape[1]
                         + torch.arange(1,q.shape[1]+1,device=q.device)).int().contiguous()
            base2 = prepare_baseline(q, cache, w, canonical, table, n)
            base2.run_fused(out)
            check_output(out, ref, valid, structured)
        rec["status"] = "PASS"
        records.append(rec)
        report_path.write_text(json.dumps(records, indent=2))
        print(json.dumps({k:v for k,v in rec.items() if k != "metadata"}), flush=True)
    rejected = guard_tests(make_case(2,2,32,513))
    records.append({"rejection_tests": rejected, "gluon_unchanged": True})
    report_path.write_text(json.dumps(records, indent=2))
    print(f"CORRECTNESS_PASS cases={len(records)-1} rejections={rejected}", flush=True)


def run_api_contracts(report_path, kernel_id=None):
    from aiter import paged_mqa_logits
    impl = importlib.import_module("aiter.ops.opus.paged_mqa_logits")
    assert impl._resolve_paged_mqa_kernel(None,rows=1,max_len=8192) == 6
    for arch in ("gfx936","gfx938"):
        options=dict(max_len=4096 if arch=="gfx936" else 16384,cache_tokens=4096,arch=arch)
        assert impl._resolve_paged_mqa_kernel(None,rows=31,**options) == 6
        assert impl._resolve_paged_mqa_kernel(None,rows=32,**options) == 7
        assert impl._resolve_paged_mqa_kernel(None,rows=32,heads=64,**options) == 6
        assert impl._resolve_paged_mqa_kernel(None,rows=32,next_n=2,**options) == 6
        assert impl._resolve_paged_mqa_kernel(None,rows=32,max_len=4095,cache_tokens=4095,arch=arch) == 6
        assert impl._resolve_paged_mqa_kernel(None,rows=32,max_len=4096,cache_tokens=8193,arch=arch) == 6
    assert impl._resolve_paged_mqa_kernel(None,rows=32,max_len=4096,arch="gfx946") == 4
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for label, shape, lengths in (
            ("minimal_mtp", (2,4,32,17), [4,4]),
            ("tile_tail", (2,2,64,67), [67,65]),
            ("zero_weights", (1,1,32,257), [257]),
            ("wide_scale", (1,2,64,513), [509]),
            ("finite_extremes", (1,1,32,129), [129])):
        args = make_case(*shape, seed=2026, lengths=lengths)
        q, cache, weights, context, table, n = args
        if label == "zero_weights":
            weights.zero_()
        elif label == "wide_scale":
            scale = torch.logspace(-8,8,cache.shape[0],base=2.0,device=q.device)
            cache.view(-1,132)[:,128:].copy_(scale[:,None].view(torch.uint8))
        elif label == "finite_extremes":
            q.view(torch.uint8).fill_(0x7e)  # E4M3FN +448
            cache.view(-1,132)[:,:128].fill_(0xfe)  # -448, exercises ReLU
            cache.view(-1,132)[::2,:128].fill_(0x7e)
        originals = [t.clone() for t in args[:-1]]
        ref, valid = reference(args)
        out = paged_mqa_logits(*args,kernelId=kernel_id)
        check_output(out, ref, valid)
        first = out.clone()
        returned = paged_mqa_logits(*args, out=out, kernelId=kernel_id)
        assert returned.data_ptr() == out.data_ptr() and torch.equal(first,out)
        out.fill_(-float("inf"))
        paged_mqa_logits(*args, out=out, clean_logits=False,kernelId=kernel_id)
        check_output(out,ref,valid)
        for before, after in zip(originals,args[:-1]):
            assert torch.equal(before.view(torch.uint8),after.view(torch.uint8)), "mutated input"
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out.fill_(float("nan"))
            paged_mqa_logits(*args, out=out,kernelId=kernel_id)
        stream.synchronize()
        check_output(out,ref,valid)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            paged_mqa_logits(*args, out=out,kernelId=kernel_id)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        error = check_output(out,ref,valid)
        records.append({"case":label,"status":"PASS","current_stream":True,
                        "graph":True,"inputs_unchanged":True,**error})
        print(json.dumps(records[-1]),flush=True)

    args = make_case(2,2,32,513)
    q, cache, w, context, table, n = args
    reject_count = 0
    def rejects(fn):
        nonlocal reject_count
        try:
            fn()
        except (ValueError, RuntimeError, NotImplementedError):
            reject_count += 1
        else:
            raise AssertionError("invalid public/private input accepted")
    rejects(lambda: paged_mqa_logits(*args,kernelId=99))
    rejects(lambda: paged_mqa_logits(*args,kernelId=True))
    rejects(lambda: paged_mqa_logits(*args,clean_logits=1))
    rejects(lambda: paged_mqa_logits(q.float(),cache,w,context,table,n))
    rejects(lambda: paged_mqa_logits(q,cache,w.double(),context,table,n))
    rejects(lambda: paged_mqa_logits(q,cache,w,context[:,None],table,n))
    rejects(lambda: paged_mqa_logits(q,cache[:,:,:,:128],w,context,table,n))
    rejects(lambda: paged_mqa_logits(q,cache,w,context,table[:,::2],n))
    rejects(lambda: paged_mqa_logits(*args,out=torch.empty((4,n),device=q.device,dtype=torch.float64)))
    rejects(lambda: paged_mqa_logits(q,cache,w,context,table,0))
    unaligned_storage = torch.empty(cache.numel()+1,dtype=torch.uint8,device=q.device)
    unaligned = unaligned_storage[1:].view_as(cache)
    rejects(lambda: paged_mqa_logits(q,unaligned,w,context,table,n))
    small = make_case(1,1,32,32)
    rejects(lambda: paged_mqa_logits(*small,out=small[2]))
    output = torch.empty((4,n),device=q.device)
    rejects(lambda: impl._paged_mqa_logits_opus(q,cache,w,context[:,None],table,output,n,0))
    rejects(lambda: impl._paged_mqa_logits_opus(q,cache,w.double(),context,table,output,n,0))
    rejects(lambda: impl._paged_mqa_logits_opus(q,cache,w,context,table,output,n,99))
    records.append({"rejection_tests":reject_count,"auto_selector_boundary_pass":True})
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(records,indent=2))
    print(f"API_PASS cases=5 rejections={reject_count}",flush=True)


def format_benchmark_table(records, max_cv_percent=5):
    """Summarize each ABBA pair without mixing its Opus timings with other pairs."""
    labels = {"fused": "Triton fused", "stage1_sum": "Triton stage1+sum",
              "opus_id0": "Opus ID0", "original": "Opus initial"}
    groups = {}
    for record in records:
        for baseline, pair in record["pairs"].items():
            key = (record["case_id"], tuple(record["shape"]),
                   record["resolved_kernel_id"], baseline)
            groups.setdefault(key, []).append(pair)
    if not groups:
        return "No benchmark measurements."
    headers = ["Case", "B/R/H/maxL", "Opus ID", "Rounds", "Baseline",
               "Base us", "Opus us", "Speedup", "Max CV", "Noise"]
    rows = []
    for (case_id, shape, kernel_id, baseline), pairs in groups.items():
        base_us = statistics.median(p["baseline"]["median_us"] for p in pairs)
        opus_us = statistics.median(p["opus"]["median_us"] for p in pairs)
        max_cv = max(p[side]["cv_percent"] for p in pairs for side in ("baseline", "opus"))
        rows.append([str(case_id), "/".join(map(str, shape)), str(kernel_id), str(len(pairs)),
                     labels.get(baseline, baseline), f"{base_us:.3f}", f"{opus_us:.3f}",
                     f"{base_us / opus_us:.3f}x", f"{max_cv:.2f}%",
                     "NOISY" if max_cv > max_cv_percent else "OK"])
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]

    def line(row):
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"

    lines = ["BENCHMARK_COMPARISON (D=128; complete device sequence)",
             "Times: median of per-round medians. Speedup: Base us / paired Opus us (>1 = Opus faster).",
             f"Max CV: maximum final CV across both sides and all rounds; NOISY if >{max_cv_percent:g}%.",
             line(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(line(row) for row in rows)
    return "\n".join(lines)


def run_benchmark(report_path, rounds=3, blocks=15, calls=20, kernel_id=None, shape_set="smoke"):
    """Frozen Graph device-sequence comparison; all paths include cleanup.

    Each baseline/Opus pair uses ABBA blocks. One extension (another `blocks`)
    retains original samples when CV exceeds 5%. No allocation/reference is
    captured. All raw timings and rejected/noisy measurements are preserved.
    """
    from aiter import paged_mqa_logits
    impl = importlib.import_module("aiter.ops.opus.paged_mqa_logits")
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {"method":{"rounds":rounds,"abba_blocks":blocks,"calls_per_graph":calls,
                        "eager_warmup":10,"graph_warmup":10,"includes_fill_sum_copy":True,
                        "inputs":"fixed warm-cache, seed=100+case_index", "max_cv_percent":5,
                        "kernel_id":kernel_id},
              "gpu":str(torch.cuda.get_device_properties(0)),
              "torch":torch.__version__, "records":[]}
    cases = MODEL_PERF_CASES if shape_set == "models" else PERF_CASES
    result["method"]["shape_set"] = shape_set
    if shape_set == "models":
        result["method"]["inputs"] = "disjoint B*N KV pages; random permutation per request; uniform lengths=N-3; seed=100+case_index"
        result["method"]["original_opus"] = "58dcd225 policy: ID0 for B*R*ceil(N/64)<=128, else ID1"
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/"op_tests/utility/paged_mqa_logits/paged_mqa_logits_triton_baseline.py",
               root/"csrc/opus_mqa_logits/include/gfx938/paged_mqa_logits.cuh",
               root/"aiter/ops/opus/paged_mqa_logits.py"]
    result["source_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    def stats(samples):
        return {"median_us":statistics.median(samples),
                "cv_percent":statistics.pstdev(samples)/statistics.mean(samples)*100,
                "samples_us":samples}
    for case_id, shape in enumerate(cases):
        options = {"independent_cache":True, "lengths":[shape[3]-3]*shape[0]} if shape_set == "models" else {}
        args = make_case(*shape,seed=100+case_id,**options)
        ref, valid = reference(args)
        base = prepare_baseline(*args)
        out = torch.empty_like(ref)
        funcs = {"fused":lambda:base.run_fused(out),
                 "stage1_sum":lambda:base.run_stage1_sum(out),
                 "opus":lambda:paged_mqa_logits(*args,out=out,kernelId=kernel_id)}
        if kernel_id == 1:
            funcs["opus_id0"] = lambda:paged_mqa_logits(*args,out=out,kernelId=0)
        if shape_set == "models":
            original_id = 0 if shape[0]*shape[1]*((shape[3]+63)//64)<=128 else 1
            funcs["original"] = lambda:paged_mqa_logits(*args,out=out,kernelId=original_id)
        graphs = {}
        for label,fn in funcs.items():
            for _ in range(10): fn()
            torch.cuda.synchronize()
            check_output(out,ref,valid)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(calls): fn()
            for _ in range(10): graph.replay()
            graphs[label] = graph
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        def sample(label):
            begin.record()
            graphs[label].replay()
            end.record()
            end.synchronize()
            return begin.elapsed_time(end)*1000/calls
        for round_id in range(rounds):
            rec = {"case_id":case_id,"shape":list(shape),"lengths":args[3].cpu().tolist(),
                   "resolved_kernel_id":impl._resolve_paged_mqa_kernel(kernel_id,rows=shape[0]*shape[1],max_len=shape[3]),
                   "seed":100+case_id,"round":round_id,"baseline_metadata":base.metadata,"pairs":{}}
            baselines = ("fused","stage1_sum","opus_id0") if kernel_id == 1 else ("fused","stage1_sum")
            if shape_set == "models":
                baselines += ("original",)
                rec.update(model="GLM-5.1" if shape[2]==32 else "DeepSeek-V4-Flash-C4",
                           original_kernel_id=original_id, cache_bytes=args[1].numel(),
                           source_context_len=shape[3] if shape[2]==32 else shape[3]*4)
            for baseline in baselines:
                data = {baseline:[],"opus":[]}
                for _ in range(blocks):
                    for label in (baseline,"opus","opus",baseline):
                        data[label].append(sample(label))
                primary = {label:stats(values.copy()) for label,values in data.items()}
                extended = max(v["cv_percent"] for v in primary.values())>5
                if extended:
                    for _ in range(blocks):
                        for label in (baseline,"opus","opus",baseline):
                            data[label].append(sample(label))
                rec["pairs"][baseline] = {
                    "baseline":stats(data[baseline]),"opus":stats(data["opus"]),
                    "extended":extended,"primary":primary,
                    "speedup":statistics.median(data[baseline])/statistics.median(data["opus"])}
            result["records"].append(rec)
            report_path.write_text(json.dumps(result,indent=2))
            print(f"BENCHMARK_PROGRESS case={case_id + 1}/{len(cases)} "
                  f"shape={shape} round={round_id + 1}/{rounds}", flush=True)
    print(format_benchmark_table(result["records"], result["method"]["max_cv_percent"]), flush=True)
    print("BENCHMARK_COMPLETE",flush=True)


def _skip_unless_hcu():
    if not torch.cuda.is_available():
        pytest.skip("requires gfx936/gfx938")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx936", "gfx938"):
        pytest.skip("ID6/7 and S64 regression requires gfx936/gfx938")


def _make_hcu_paged_case(b,r,h,n,s,*,shared=False,offset=False):
    capacity=(n+63)//64*64 if s==64 else n
    args=list(make_case(b,r,h,capacity,seed=20260914+n, lengths=[n]*b if shared else None))
    q,cache,w,ctx,tb,_=args
    if not shared:
        ctx=torch.tensor([max(r,n-i*7-3) for i in range(b)],device='cuda',dtype=torch.int32)
    if s==64:
        pages=capacity//64
        packed=torch.empty((pages,64*132),device='cuda',dtype=torch.uint8)
        old=cache.reshape(capacity,132)
        packed[:,:8192]=old[:,:128].reshape(pages,8192)
        packed[:,8192:]=old[:,128:].reshape(pages,256)
        cache=packed.reshape(pages,64,1,132)
        tb=torch.randint(pages,(b,(n+63)//64),device='cuda',dtype=torch.int32)
    if shared:
        tb=torch.arange((n+s-1)//s,device='cuda',dtype=torch.int32)[None].expand(b,-1).contiguous()
    if offset:
        qbase=torch.empty(q.numel()+4,device='cuda',dtype=torch.uint8)
        qview=qbase[4:].view(torch.float8_e4m3fn).reshape_as(q);qview.copy_(q);q=qview
        kbase=torch.empty(cache.numel()+4,device='cuda',dtype=torch.uint8)
        kview=kbase[4:].reshape_as(cache);kview.copy_(cache);cache=kview
        assert q.data_ptr()%16==4 and cache.data_ptr()%16==4
    return q,cache,w,ctx,tb,n


def _hcu_paged_reference(args):
    q,cache,w,ctx,tb,n=args
    s=cache.shape[1]
    if s==1:return reference(args)
    pages=cache.shape[0];raw=cache.reshape(pages,s*132)
    canonical=torch.empty((pages*s,1,1,132),device='cuda',dtype=torch.uint8)
    canonical.reshape(-1,132)[:,:128]=raw[:,:s*128].reshape(-1,128)
    canonical.reshape(-1,132)[:,128:]=raw[:,s*128:].reshape(-1,4)
    expanded=(tb[:,:,None]*s+torch.arange(s,device='cuda',dtype=torch.int32)).reshape(q.shape[0],-1)
    return reference((q,canonical,w,ctx,expanded,n))


def _validate_hcu_paged_case(args,kid=None,*,graph=False):
    from aiter import paged_mqa_logits
    ref,mask=_hcu_paged_reference(args)
    allocated=paged_mqa_logits(*args,kernelId=kid)
    check_output(allocated,ref,mask)
    guard=torch.full((ref.numel()+32,),654321.,device='cuda')
    out=guard[16:-16].reshape_as(ref)
    returned=paged_mqa_logits(*args,out=out,kernelId=kid)
    assert returned.data_ptr()==out.data_ptr()
    torch.cuda.synchronize();check_output(out,ref,mask)
    if graph:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):paged_mqa_logits(*args,out=out,kernelId=kid)
        stream.synchronize();g.replay();torch.cuda.synchronize();check_output(out,ref,mask)
    assert bool((guard[:16]==654321.).all()) and bool((guard[-16:]==654321.).all())
    return out


class TestPagedMQAHCU:
    """ID6/7, S1/S64 and graph regressions on gfx936/gfx938 only."""

    @pytest.fixture(autouse=True)
    def require_hcu(self):
        # Class scope keeps this guard from skipping gfx946 tests below.
        _skip_unless_hcu()
        torch.backends.cuda.matmul.allow_tf32 = False

    @pytest.mark.parametrize('s',[1,64])
    @pytest.mark.parametrize('r',[1,2,4])
    @pytest.mark.parametrize('h',[32,64])
    @pytest.mark.parametrize('n',[17,67,257])
    def test_independent_pages(self, s,r,h,n):
        _validate_hcu_paged_case(_make_hcu_paged_case(3,r,h,n,s),6)

    @pytest.mark.parametrize('s',[1,64])
    @pytest.mark.parametrize('n',[17,67,257,4097])
    @pytest.mark.parametrize('shared',[True,False])
    def test_grouped_and_fallback(self, s,n,shared):
        _validate_hcu_paged_case(_make_hcu_paged_case(9,1,32,n,s,shared=shared),7)

    @pytest.mark.parametrize('s',[1,64])
    @pytest.mark.parametrize('kid',[6,7])
    def test_offset_storage_and_graph(self, s,kid):
        _validate_hcu_paged_case(_make_hcu_paged_case(8,1,32,257,s,shared=True,offset=True),kid,graph=True)

    @pytest.mark.parametrize('s',[1,64])
    @pytest.mark.parametrize('b,n',[(2,257),(2,8193),(33,4097)])
    def test_graph_rechecks_mutated_metadata(self, s,b,n):
        from aiter import paged_mqa_logits
        args=_make_hcu_paged_case(b,1,32,n,s,shared=True)
        out=_validate_hcu_paged_case(args)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            paged_mqa_logits(*args,out=out)
            allocated=paged_mqa_logits(*args)
        q,k,w,ctx,tb,n=args
        # A middle page breaks sharing; a second pair has different lengths.
        middle=tb.shape[1]//2;tb[1,middle]=(tb[1,middle]+1)%k.shape[0]
        ctx[min(3,b-1)]-=9;w.mul_(-0.5)
        q.view(torch.uint8).bitwise_xor_(0x80)
        k.view(-1)[0]=0x01  # New subnormal value must be read on replay.
        g.replay();torch.cuda.synchronize()
        ref,mask=_hcu_paged_reference(args);check_output(out,ref,mask)
        check_output(allocated,ref,mask)

    @pytest.mark.parametrize('s',[1,64])
    def test_fp8_subnormals_and_finite_extremes(self, s):
        args=list(_make_hcu_paged_case(2,1,32,129,s,shared=True))
        q,k,w,ctx,tb,n=args
        pattern=torch.arange(256,device='cuda',dtype=torch.int32).to(torch.uint8)
        pattern[0x7f]=0;pattern[0xff]=0x80
        q.view(torch.uint8).reshape(-1).copy_(pattern.repeat((q.numel()+255)//256)[:q.numel()])
        raw=k.reshape(k.shape[0],-1)
        keys=pattern.repeat((k.shape[0]*s*128+255)//256)[:k.shape[0]*s*128].reshape(k.shape[0],s*128)
        raw[:,:s*128]=keys
        raw[:,s*128:]=torch.ones((k.shape[0],s),device='cuda').view(torch.uint8)
        for kid in (6,7):_validate_hcu_paged_case(tuple(args),kid)

    @pytest.mark.parametrize('n',[4097,65537])
    def test_shared_permuted_pages_and_replay(self,n):
        """S64 任意同序页表允许共享；跨检查块及 Graph 中页表变化必须重新验证。"""
        from aiter import paged_mqa_logits
        args=_make_hcu_paged_case(5,1,32,n,64,shared=True)
        q,k,w,ctx,tb,_=args
        tb.copy_(torch.randperm(tb.shape[1],device='cuda',dtype=torch.int32)[None].expand_as(tb))
        out=_validate_hcu_paged_case(args,7,graph=True)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):paged_mqa_logits(*args,out=out,kernelId=7)
        tb[1,-1]=(tb[1,-1]+1)%k.shape[0]
        ctx[3]-=17
        graph.replay();torch.cuda.synchronize()
        ref,mask=_hcu_paged_reference(args);check_output(out,ref,mask)

    @pytest.mark.parametrize('preallocated',[False,True])
    def test_inference_metadata_checks(self,preallocated):
        """原生分配入口与 out 入口均须拒绝梯度输入、非连续输入和错误类型。"""
        from aiter import paged_mqa_logits
        args=list(_make_hcu_paged_case(2,1,32,67,64))
        out=torch.empty((2,67),device='cuda') if preallocated else None
        for index,bad in (
                (2,args[2].clone().requires_grad_()),
                (2,args[2].t().contiguous().t()),
                (3,args[3].long()),
                (4,args[4].cpu())):
            invalid=args.copy();invalid[index]=bad
            with pytest.raises((ValueError,RuntimeError)):
                paged_mqa_logits(*invalid,out=out)
        with pytest.raises((ValueError,RuntimeError)):
            paged_mqa_logits(*args,out=torch.empty((2,67),device='cuda',requires_grad=True))

    def test_hcu_selector_rejects_invalid_modes(self):
        from aiter import paged_mqa_logits
        for args in (_make_hcu_paged_case(2,2,32,67,1),_make_hcu_paged_case(2,1,64,67,64)):
            with pytest.raises(ValueError,match='H=32.*R=1'):paged_mqa_logits(*args,kernelId=7)
        impl=importlib.import_module('aiter.ops.opus.paged_mqa_logits')
        with pytest.raises(ValueError):impl._resolve_paged_mqa_kernel(0,rows=1,max_len=64,arch='gfx936')
        with pytest.raises(ValueError):impl._resolve_paged_mqa_kernel(7,rows=32,max_len=4096,arch='gfx946')


def _skip_unless_gfx938():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("requires a gfx938 GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx938":
        pytest.skip("paged MQA FP8 runtime coverage is gfx938 only")


def test_paged_mqa_correctness(tmp_path):
    _skip_unless_gfx938()
    run_correctness("opus",tmp_path/"correctness.json")


def test_paged_mqa_public_contract(tmp_path):
    _skip_unless_hcu()
    run_api_contracts(tmp_path/"api.json")


@pytest.mark.parametrize('kid',[0,1,2,3])
@pytest.mark.parametrize('h',[32,64])
def test_paged_mqa_preserved_opus(kid,h):
    _skip_unless_gfx938()
    from aiter import paged_mqa_logits
    args=make_case(2,2,h,513,structured=True)
    ref,valid=reference(args)
    check_output(paged_mqa_logits(*args,kernelId=kid),ref,valid,structured=True)


def test_paged_mqa_gfx946_cpu_reference(tmp_path):
    """Current public API on gfx946, including automatic and both explicit IDs.

    CPU references avoid depending on unrelated model-side Torch/Triton kernels.
    ID6/7 and S64 regressions are grouped in TestPagedMQAHCU above.
    """
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("requires gfx946 (Perf Model or hardware)")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx946":
        pytest.skip("gfx946 public-API regression")
    from op_tests.utility.paged_mqa_logits.cpu_reference import run_cpu_reference
    for kernel_id in (None, 4, 5):
        label = "auto" if kernel_id is None else f"id{kernel_id}"
        run_cpu_reference(tmp_path/f"gfx946_{label}.json", kernel_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("baseline", "opus"), default="opus",
                        help="opus (default): compare Opus and both Triton baselines; baseline: Triton only")
    parser.add_argument("--json-out", type=Path,
                        help="JSON result path (default: repo/hygon_tmp/paged_mqa_logits/<mode>_<timestamp>.json)")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--cpu-reference", action="store_true",
                        help="CPU inputs/reference, Opus DUT only; suitable for gfx946 Perf Model")
    parser.add_argument("--cpu-graph", action="store_true",
                        help="also validate graph replay/metadata mutation with --cpu-reference; requires runtime graph support")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--shape-set", choices=("smoke", "models"), default="smoke",
                        help="benchmark matrix: smoke (default) or model indexer shapes with independent per-request KV pages")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=15)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--kernel-id", type=int, default=None)
    opts = parser.parse_args()
    if opts.cpu_graph and not opts.cpu_reference:
        parser.error("--cpu-graph requires --cpu-reference")
    if opts.json_out is None:
        mode = "benchmark" if opts.bench else "api" if opts.api_only else f"{opts.backend}_correctness"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        opts.json_out = Path(__file__).resolve().parents[1]/"hygon_tmp"/"paged_mqa_logits"/f"{mode}_{stamp}.json"
    print(f"JSON_OUT={opts.json_out.resolve()}", flush=True)
    if opts.cpu_reference:
        if opts.bench or opts.api_only or opts.backend != "opus":
            parser.error("--cpu-reference requires Opus correctness mode")
        from op_tests.utility.paged_mqa_logits.cpu_reference import run_cpu_reference
        run_cpu_reference(opts.json_out, opts.kernel_id, test_graph=opts.cpu_graph)
    elif opts.bench:
        run_benchmark(opts.json_out,opts.rounds,opts.blocks,opts.calls,opts.kernel_id,opts.shape_set)
    elif opts.api_only:
        run_api_contracts(opts.json_out,opts.kernel_id)
    else:
        run_correctness(opts.backend, opts.json_out,opts.kernel_id)
