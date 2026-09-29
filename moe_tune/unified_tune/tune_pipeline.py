# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""编排调优、独立验证与可选配置安装。

启动隔离 worker，控制超时/中断、日志、状态与断点续跑；
双后端分别记录结果，并对成功产物执行校验和安装。"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .tune_artifacts import exclusive_lock, install, resume_valid, validate_files, write_json
from .tune_spec import digest, log_shape_progress

WORKER = Path(__file__).with_name("tune_worker.py")
ROOT = WORKER.parents[2]


def _forward_shape_progress(stream):
    """跟读完整进度行；原生调优诊断保留在日志文件，不刷满控制台。"""
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line:
            return
        if not line.endswith("\n"):
            stream.seek(position)
            return
        if line.startswith(("[shape ", "M=")):
            print(line, end="", flush=True)


def execute_worker(request, request_path, result_path, log_path, timeout):
    write_json(request_path, request)
    result_path.unlink(missing_ok=True)
    command = [sys.executable, "-I", "-B", str(WORKER), "--request", str(request_path), "--result", str(result_path)]
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    # Inherited training flags must not leak into probe/validation processes.
    for key in ("TRAINING_MODE", "TRITON_COMPILE_ONLY", "AUTOTUNE_TEST_TYPE", "CUR_GROUP", "EP_SIZE",
                "NUM_GROUPS_COMPILE", "SPLITK_SIZE"):
        environment.pop(key, None)
    # The input contract is one native inference chunk, independent of caller overrides.
    environment["TRITON_FUSED_MOE_CHUNK_SIZE"] = "32768"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(dict(command=command, request=request), sort_keys=True) + "\n"); log.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=(os.name != "nt"))
        with log_path.open("r", encoding="utf-8", errors="replace") as progress_log:
            deadline = time.monotonic() + timeout
            try:
                while True:
                    _forward_shape_progress(progress_log)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        child.wait(timeout=min(0.2, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            except BaseException as exc:
                if child.poll() is None:
                    if os.name == "nt": child.terminate()
                    else: os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if os.name == "nt": child.kill()
                        else: os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise RuntimeError(f"{request['action']} timed out after {timeout}s; log: {log_path}") from exc
                raise
            finally:
                _forward_shape_progress(progress_log)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    expected_status = {"probe": "passed", "tune": "tuned", "validate": "passed"}[request['action']]
    if child.returncode or result.get("status") != expected_status:
        raise RuntimeError(f"{request['action']} failed ({child.returncode}): {result.get('error', 'no result')}; log: {log_path}")
    return result


def run(args, specs, backends):
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    options = {k: getattr(args, k) for k in ("device", "search", "warmup", "iterations")}
    probe_dir = root / ".runs" / uuid.uuid4().hex; probe_dir.mkdir(parents=True)
    print(f"Checking runtime on logical GPU {args.device} ...", flush=True)
    probe = execute_worker(dict(action="probe", options=options), probe_dir / "probe_request.json",
                           probe_dir / "probe.json", probe_dir / "probe.log", args.timeout)
    runtime = probe["runtime"]
    failures = []
    shape_total = sum(len(spec.tokens) for spec in specs)
    shape_offset = 0
    for spec in specs:
        case_dir = root / runtime["arch"] / spec.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(case_dir / ".lock"):
            for backend_index, backend in enumerate(backends, start=1):
                progress = dict(shape_offset=shape_offset, shape_total=shape_total,
                                backend_index=backend_index, backend_total=len(backends))
                directory = case_dir / backend
                directory.mkdir(parents=True, exist_ok=True)
                manifest_path = directory / "manifest.json"
                # Device load/free memory and timings do not participate in the identity.
                fingerprint = digest(dict(spec=spec.to_dict(), backend=backend, options=options, runtime=runtime))
                previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                manifest = dict(fingerprint=fingerprint, spec=spec.to_dict(), backend=backend,
                                options=options, runtime=runtime, status="pending")
                try:
                    if args.resume and resume_valid(previous, fingerprint):
                        manifest = previous
                        files = list(manifest["artifact_hashes"])
                        for m in spec.tokens:
                            log_shape_progress(spec, m, backend, progress, stage="RESUME")
                        print(f"RESUME {backend} {spec.case_id}: verified artifacts", flush=True)
                    else:
                        if previous and not args.resume:
                            raise ValueError(f"{manifest_path} already exists; use --resume or a new --output-dir")
                        write_json(manifest_path, manifest)
                        request = dict(spec=spec.to_dict(), backend=backend, options=options,
                                       workdir=str(directory), progress=progress)
                        started = time.monotonic()
                        print(f"TUNE {backend} {spec.case_id}, tokens={spec.tokens}, search={args.search}", flush=True)
                        tuned = execute_worker(dict(request, action="tune"), directory / "tune_request.json",
                            directory / "tune_result.json", directory / "tune.log", args.timeout)
                        files = tuned["files"]
                        hashes = validate_files(spec, backend, files, runtime["arch"])
                        validation = execute_worker(dict(request, action="validate"), directory / "validate_request.json",
                            directory / "validation.json", directory / "validation.log", args.timeout)
                        manifest.update(status="validated", artifact_hashes=hashes, tuning=tuned,
                                        validation=validation, wall_seconds=time.monotonic() - started)
                        write_json(manifest_path, manifest)
                    if args.install_configs:
                        validate_files(spec, backend, files, runtime["arch"])
                        manifest["installation"] = install(spec, backend, files, runtime,
                            replace_existing=args.replace_existing_configs)
                        manifest["status"] = "installed"
                        write_json(manifest_path, manifest)
                    print(f"PASS {backend}: {manifest_path}", flush=True)
                except (Exception, KeyboardInterrupt) as exc:
                    if manifest.get("status") in ("validated", "installed"):
                        manifest["installation_error"] = str(exc)
                    else:
                        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
                    # Do not replace a prior successful manifest when the user omitted --resume.
                    if not previous or args.resume:
                        write_json(manifest_path, manifest)
                    failures.append(dict(case_id=spec.case_id, backend=backend, error=str(exc)))
                    print(f"FAIL {backend}: {exc}", file=sys.stderr, flush=True)
                    if isinstance(exc, KeyboardInterrupt): raise
        shape_offset += len(spec.tokens)
    write_json(probe_dir / "run_result.json", dict(status="failed" if failures else "passed", failures=failures,
                                                  output_dir=str(root), backends=backends))
    return 1 if failures else 0
