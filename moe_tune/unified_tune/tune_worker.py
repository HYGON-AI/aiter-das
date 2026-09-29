# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""隔离进程内的环境探测、后端调优与公开 API 回读。

按 request/result JSON 协议执行任务，锁定系统 Triton/BoltOPs 来源；
记录运行环境和源码指纹，并验证配置命中及 eager/inplace/Graph 精度。"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_tune.tune_spec import TuneSpec, QUANTS, log_shape_progress
from unified_tune.tune_dependencies import pin_installed_packages, imported_package_files

_DEPENDENCIES = None


def sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def runtime_info(device):
    global _DEPENDENCIES
    if _DEPENDENCIES is None:
        _DEPENDENCIES = pin_installed_packages()
        sys.path.insert(0, str(ROOT))  # Only AITER uses this task's source checkout.
    import torch
    import triton
    import aiter
    from aiter.jit.core import AITER_ROOT_DIR
    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise RuntimeError(f"logical device {device} is unavailable")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    aiter_root = Path(AITER_ROOT_DIR)
    bolt_root = None
    bolt_info = dict(boltops=None, capability=None, compiler_options=[], backend=None,
                     triton_config_dir=None)
    try:
        import boltops
        from boltops.utility.triton_capability import get_triton_capabilities, get_triton_config_dir
        import boltops.fused_moe.triton.moe_config_utils as configs
        cap = get_triton_capabilities()
        bolt_root = Path(boltops.__file__).resolve().parent
        bolt_info.update(boltops=str(Path(boltops.__file__).resolve()), capability=cap.version,
            compiler_options=sorted(cap.compiler_options), backend=cap.backend,
            triton_config_dir=str(get_triton_config_dir(configs._TRITON_CONFIGS_DIR, cap)))
    except Exception as exc:
        # Keep the environment report available even when a dependency is missing.
        bolt_info['boltops_error'] = f'{type(exc).__name__}: {exc}'
    paths = list((ROOT / "moe_tune").rglob("*.py"))
    if bolt_root is not None:
        paths += list((bolt_root / "tools/fused_moe_triton_tune").glob("*.py"))
        paths += list((bolt_root / "fused_moe/triton").glob("*.py"))
        paths += list((bolt_root / "utility").glob("*capability*.py"))
    paths += [aiter_root / "aiter/moe.py", aiter_root / "aiter/fused_moe_asm_wna16.py"]
    paths += list((aiter_root / "aiter/jit").glob("*.so"))
    sources = {str(p.resolve()): sha256(p) for p in sorted(set(paths)) if p.is_file()}
    return dict(arch=properties.gcnArchName.split(":")[0], device=device, device_name=properties.name,
        total_memory=properties.total_memory, python=sys.version, torch=torch.__version__,
        hip=torch.version.hip, triton=triton.__version__,
        visible_devices={key: os.environ.get(key) for key in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')},
        aiter=str(Path(aiter.__file__).resolve()), dependencies=_DEPENDENCIES, **bolt_info,
        asm_config_dir=str(aiter_root / "aiter/configs"),
        sources=sources)


def replay(spec, backend, workdir, options, runtime):
    import torch
    from unified_tune.backends.common import make_data, reference, check, get_config, public_call, benchmark
    from contextlib import ExitStack
    results, hits = [], []
    expected_pair = []
    with ExitStack() as stack:
        if backend == "asm":
            import aiter.fused_moe_asm_wna16 as asm
            path = workdir / spec.asm_filename()
            asm._cached_data_by_quant.clear(); asm.get_moe_asm_solution.cache_clear()
            stack.enter_context(patch.object(asm, "get_csv_path", return_value=str(path)))
            hits.append(dict(path=str(path), sha256=sha256(path)))
        else:
            import boltops.fused_moe.triton.moe_config_utils as cu
            from boltops.utility.triton_capability import get_triton_capabilities, get_triton_config_dir
            stack.enter_context(patch.object(cu, "_TRITON_CONFIGS_DIR", workdir))
            cu.get_moe_configs.cache_clear()
            directory = get_triton_config_dir(workdir, get_triton_capabilities())
            for bottom, filename in enumerate(spec.triton_filenames(runtime["arch"])):
                path = directory / filename
                expected = {int(k): v for k, v in json.loads(path.read_text()).items()}
                expected_pair.append(expected)
                actual = cu.get_moe_configs(spec.experts, spec.inter_dim, QUANTS[spec.quant_type][1], spec.q_size_n, spec.q_size_k, bool(bottom))
                if actual != expected or set(actual) != set(spec.tokens):
                    raise RuntimeError(f"config lookup does not match generated artifact: {path}")
                hits.append(dict(path=str(path), sha256=sha256(path)))
        for m in spec.tokens:
            log_shape_progress(spec, m, backend, options.get("progress"), stage="VALIDATE")
            data = make_data(spec, m); ref = reference(spec, data)
            cfg = get_config(spec, m, backend)
            if backend == "asm":
                import csv
                rows = list(csv.DictReader(path.open()))
                row = next(r for r in rows if int(r["token"]) == m)
                if f"{cfg.config['SOL_ID1']}+{cfg.config['SOL_ID2']}" != row["sol_id"]:
                    raise RuntimeError("ASM selected solution differs from generated CSV")
            call = public_call(spec, data, cfg)
            actual_lookups = []
            if backend == "triton":
                lookup = cu.try_get_optimal_moe_config

                def trace_lookup(*args, **kwargs):
                    result = lookup(*args, **kwargs)
                    bottom = bool(kwargs.get("is_bottom", False))
                    selected = result[0] if bottom else result
                    if selected != expected_pair[int(bottom)][m]:
                        raise RuntimeError("executed Triton lookup differs from requested token artifact")
                    actual_lookups.append(dict(is_bottom=bottom, token=m, config=selected.copy()))
                    return result

                with patch.object(cu, "try_get_optimal_moe_config", trace_lookup):
                    accuracy = check(call(), ref)
                if {item["is_bottom"] for item in actual_lookups} != {False, True}:
                    raise RuntimeError("public Triton execution did not read both stage configs")
            else:
                accuracy = check(call(), ref)
            inplace_accuracy = check(public_call(spec, data, cfg, inplace=True)(), ref)
            timing = benchmark(call, options["warmup"], options["iterations"])
            # Graph replay checks the normal inference path, independently of candidate timing.
            for _ in range(3): call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = call()
            for _ in range(3): graph.replay()
            torch.cuda.synchronize()
            graph_accuracy = check(graph_output, ref)
            graph_timing = benchmark(call, options['warmup'], options['iterations'], use_graph=True)
            results.append(dict(token=m, backend=cfg.solution_type, config=cfg.config, accuracy=accuracy,
                inplace_accuracy=inplace_accuracy, graph_accuracy=graph_accuracy, end_to_end=timing,
                graph_end_to_end=graph_timing, executed_lookups=actual_lookups))
            print(f"REPLAY PASS backend={backend} M={m} error_ratio={accuracy['error_ratio']}", flush=True)
            del data, ref, graph, graph_output, call
            torch.cuda.empty_cache()
    return dict(status="passed", hits=hits, cases=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    started = time.monotonic()
    result = {}
    try:
        action = request["action"]
        if action == "tune" and request["backend"] == "triton":
            spec = TuneSpec(**request["spec"])
            os.environ.update(TRAINING_MODE="1", TRITON_COMPILE_ONLY="0", TRITON_PRINT_AUTOTUNING="1",
                              AUTOTUNE_TEST_TYPE=QUANTS[spec.quant_type][0], EP_SIZE="1", CUR_GROUP="0")
        info = runtime_info(request["options"]["device"])
        if action == "probe":
            result = dict(status="passed", runtime=info)
        else:
            spec = TuneSpec(**request["spec"])
            workdir = Path(request["workdir"]).resolve()
            if spec.backend_error(request['backend']):
                raise RuntimeError(spec.backend_error(request['backend']))
            if info.get('boltops_error'):
                raise RuntimeError("The current AITER MoE runtime requires BoltOPs, including ASM's imported helpers: "
                                   + info['boltops_error'])
            if action == "tune":
                module = importlib.import_module(f"unified_tune.backends.{request['backend']}_backend")
                result = module.tune(spec, workdir, dict(request["options"], progress=request.get("progress")), info)
                result["status"] = "tuned"
            elif action == "validate":
                result = replay(spec, request["backend"], workdir,
                                dict(request["options"], progress=request.get("progress")), info)
            else:
                raise ValueError(f"unknown action {action}")
            result["runtime"] = info
        result['dependency_modules'] = imported_package_files(_DEPENDENCIES)
        result["wall_seconds"] = time.monotonic() - started
        returncode = 0
    except Exception as exc:
        traceback.print_exc()
        result = dict(status="failed", error=str(exc), wall_seconds=time.monotonic() - started)
        returncode = 1
    args.result.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
