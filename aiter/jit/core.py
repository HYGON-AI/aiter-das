# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 Hygon Info Technologies Ltd.
 
import functools
import importlib
import json
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import types
import typing
import copy
from typing import Any, Callable, List, Optional

from packaging.version import Version, parse

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{this_dir}/utils/")
from chip_info import get_configured_gfx, get_gfx
from cpp_extension import (
    _jit_compile,
    executable_path,
    get_hip_version,
    hcu_sanitize_display,
)
from file_baton import FileBaton
from torch_guard import torch_compile_guard  # noqa: E402

AITER_REBUILD = int(os.environ.get("AITER_REBUILD", "0"))

aiter_lib = None


def mp_lock(
    lockPath: str,
    MainFunc: Callable,
    FinalFunc: Optional[Callable] = None,
    WaitFunc: Optional[Callable] = None,
):
    """
    Using FileBaton for multiprocessing.
    """
    baton = FileBaton(lockPath)
    if baton.try_acquire():
        try:
            ret = MainFunc()
        finally:
            if FinalFunc is not None:
                FinalFunc()
            baton.release()
    else:
        baton.wait()
        if WaitFunc is not None:
            ret = WaitFunc()
        ret = None
    return ret


logger = logging.getLogger("aiter")

PY = sys.executable
this_dir = os.path.dirname(os.path.abspath(__file__))

AITER_ROOT_DIR = os.path.abspath(f"{this_dir}/../../")
AITER_LOG_MORE = int(os.getenv("AITER_LOG_MORE", 0))
AITER_LOG_TUNED_CONFIG = int(os.getenv("AITER_LOG_TUNED_CONFIG", 0))

# config_env start here
def update_config_files(file_path: str, merge_name: str):
    path_list = file_path.split(os.pathsep) if file_path else []
    if len(path_list) <= 1:
        return file_path
    df_list = []
    ## merge config files
    ##example: AITER_CONFIG_GEMM_A4W4="/path1:/path2"
    import pandas as pd

    df_list.append(pd.read_csv(path_list[0]))
    for i, path in enumerate(path_list[1:]):
        if os.path.exists(path):
            df = pd.read_csv(path)
            ## check columns
            assert (
                df.columns.tolist() == df_list[0].columns.tolist()
            ), f"Column mismatch between {path_list[0]} and {path}, {df_list[0].columns.tolist()}, {df.columns.tolist()}"

            df_list.append(df)
        else:
            logger.info(f"path {i+1}: {path} (not exist)")
    merge_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    ## get keys from untuned file to drop_duplicates
    untuned_name = (
        re.sub(r"(?:_)?tuned$", r"\1untuned", merge_name)
        if re.search(r"(?:_)?tuned$", merge_name)
        else merge_name.replace("tuned", "untuned")
    )
    untuned_path = f"{AITER_ROOT_DIR}/aiter/configs/{untuned_name}.csv"
    if os.path.exists(untuned_path):
        untunedf = pd.read_csv(untuned_path)
        keys = untunedf.columns
        merge_df = (
            merge_df.sort_values("us")
            .drop_duplicates(subset=keys, keep="first")
            .reset_index(drop=True)
        )
    else:
        logger.warning(
            f"Untuned config file not found: {untuned_path}. Using all columns for deduplication."
        )
    new_file_path = f"/tmp/{merge_name}.csv"
    merge_df.to_csv(new_file_path, index=False)
    return new_file_path


def get_config_file(env_name, default_file, tuned_file_name):
    config_env_file = os.getenv(env_name)
    # default_file = f"{AITER_ROOT_DIR}/aiter/configs/{tuned_file_name}.csv"
    from pathlib import Path

    if not config_env_file:
        model_config_dir = Path(f"{AITER_ROOT_DIR}/aiter/configs/model_configs/")
        op_tuned_file_list = [
            p
            for p in model_config_dir.glob(f"*{tuned_file_name}*")
            if (p.is_file() and "untuned" not in str(p))
        ]

        if not op_tuned_file_list:
            config_file = default_file
        else:
            tuned_files = ":".join(str(p) for p in op_tuned_file_list)
            tuned_files = default_file + ":" + tuned_files
            logger.info(
                f"merge tuned file under model_configs/ and configs/ {tuned_files}"
            )
            config_file = update_config_files(tuned_files, tuned_file_name)
    else:
        config_file = update_config_files(config_env_file, tuned_file_name)
        # print(f"get config file from environment ", config_file)
    return config_file


AITER_CONFIG_GEMM_A4W4 = os.getenv(
    "AITER_CONFIG_GEMM_A4W4",
    f"{AITER_ROOT_DIR}/aiter/configs/a4w4_blockscale_tuned_gemm.csv",
)
AITER_CONFIG_GEMM_A8W8 = os.getenv(
    "AITER_CONFIG_GEMM_A8W8",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_tuned_gemm.csv",
)
AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE = os.getenv(
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv",
)
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE = os.getenv(
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_blockscale_tuned_gemm.csv",
)
AITER_CONFIG_FMOE = os.getenv(
    "AITER_CONFIG_FMOE",
    f"{AITER_ROOT_DIR}/aiter/configs/tuned_fmoe.csv",
)

AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE = os.getenv(
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
)

AITER_CONFIG_A8W8_BATCHED_GEMM = os.getenv(
    "AITER_CONFIG_A8W8_BATCHED_GEMM",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_tuned_batched_gemm.csv",
)

AITER_CONFIG_BF16_BATCHED_GEMM = os.getenv(
    "AITER_CONFIG_BF16_BATCHED_GEMM",
    f"{AITER_ROOT_DIR}/aiter/configs/bf16_tuned_batched_gemm.csv",
)

AITER_CONFIG_GEMM_BF16 = os.getenv(
    "AITER_CONFIG_GEMM_BF16",
    f"{AITER_ROOT_DIR}/aiter/configs/tuned_gemm.csv",
)
AITER_CONFIG_GEMM_A4W4_FILE = get_config_file(
    "AITER_CONFIG_GEMM_A4W4", AITER_CONFIG_GEMM_A4W4, "a4w4_blockscale_tuned_gemm"
)

AITER_CONFIG_GEMM_A8W8_FILE = get_config_file(
    "AITER_CONFIG_GEMM_A8W8", AITER_CONFIG_GEMM_A8W8, "a8w8_tuned_gemm"
)
AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE = get_config_file(
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE,
    "a8w8_bpreshuffle_tuned_gemm",
)
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE = get_config_file(
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE,
    "a8w8_blockscale_tuned_gemm",
)
AITER_CONFIG_FMOE_FILE = get_config_file(
    "AITER_CONFIG_FMOE", AITER_CONFIG_FMOE, "tuned_fmoe"
)

AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE = get_config_file(
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE,
    "a8w8_blockscale_bpreshuffle_tuned_gemm",
)

AITER_CONFIG_A8W8_BATCHED_GEMM_FILE = get_config_file(
    "AITER_CONFIG_A8W8_BATCHED_GEMM",
    AITER_CONFIG_A8W8_BATCHED_GEMM,
    "a8w8_tuned_batched_gemm",
)

AITER_CONFIG_BF16_BATCHED_GEMM_FILE = get_config_file(
    "AITER_CONFIG_BF16_BATCHED_GEMM",
    AITER_CONFIG_BF16_BATCHED_GEMM,
    "bf16_tuned_batched_gemm",
)

AITER_CONFIG_GEMM_BF16_FILE = get_config_file(
    "AITER_CONFIG_GEMM_BF16", AITER_CONFIG_GEMM_BF16, "bf16_tuned_gemm"
)

# config_env end here

find_aiter = importlib.util.find_spec("aiter")
if find_aiter is not None:
    if find_aiter.submodule_search_locations:
        package_path = find_aiter.submodule_search_locations[0]
    elif find_aiter.origin:
        package_path = find_aiter.origin
    package_path = os.path.dirname(package_path)
    package_parent_path = os.path.dirname(package_path)

    try:
        with open(f"{this_dir}/../install_mode", "r") as f:
            # develop mode
            isDevelopMode = f.read().strip() == "develop"
    except FileNotFoundError:
        # pip install -e
        isDevelopMode = True

    if isDevelopMode:
        AITER_META_DIR = AITER_ROOT_DIR
    # install mode
    else:
        AITER_META_DIR = os.path.abspath(f"{AITER_ROOT_DIR}/aiter_meta/")
else:
    AITER_META_DIR = AITER_ROOT_DIR
    logger.warning("aiter is not installed.")
sys.path.insert(0, AITER_META_DIR)
AITER_CSRC_DIR = f"{AITER_META_DIR}/csrc"
AITER_GRADLIB_DIR = f"{AITER_META_DIR}/gradlib"
gfx = get_gfx()
AITER_ASM_DIR = f"{AITER_META_DIR}/hsa/{gfx}/"
os.environ["AITER_ASM_DIR"] = AITER_ASM_DIR
CK_3RDPARTY_DIR = os.environ.get(
    "CK_DIR", f"{AITER_META_DIR}/3rdparty/composable_kernel"
)
CK_DIR = CK_3RDPARTY_DIR

MOE_C_3RDPARTY_DIR = os.environ.get(
    "MOE_C_DIR", f"{AITER_META_DIR}/3rdparty/moe_c"
)

MOE_C_DIR = MOE_C_3RDPARTY_DIR

os.environ["AITER_META_DIR"] = AITER_META_DIR


@functools.lru_cache(maxsize=1)
def get_asm_dir():
    return AITER_ASM_DIR


@functools.lru_cache(maxsize=1)
def get_user_jit_dir() -> str:
    if "AITER_JIT_DIR" in os.environ:
        path = os.getenv("AITER_JIT_DIR", "")
        os.makedirs(path, exist_ok=True)
        sys.path.insert(0, path)
        return path
    else:
        if os.access(this_dir, os.W_OK):
            return this_dir
    home_jit_dir = f"{os.path.expanduser('~')}/.aiter/{os.path.basename(this_dir)}"
    if not os.path.exists(home_jit_dir):
        shutil.copytree(this_dir, home_jit_dir)
    return home_jit_dir


bd_dir = f"{get_user_jit_dir()}/build"
# copy ck to build, thus hippify under bd_dir
if multiprocessing.current_process().name == "MainProcess":
    os.makedirs(bd_dir, exist_ok=True)
    # if os.path.exists(f"{bd_dir}/ck/library"):
    #     shutil.rmtree(f"{bd_dir}/ck/library")
# CK_DIR = f"{bd_dir}/ck"


def validate_and_update_archs():
    configured_gfx = get_configured_gfx()
    if not os.getenv("GPU_ARCHS", "").strip() and configured_gfx != "native":
        logger.info(
            "GPU_ARCHS is unset; map Perf Model GPU_CHIP=%s to %s",
            os.getenv("GPU_CHIP", ""),
            configured_gfx,
        )
    archs = configured_gfx.split(";")
    archs = [arch.strip() for arch in archs if arch.strip()]
    # List of allowed architectures
    allowed_archs = [
        "native",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx1100",
        "gfx950",
        "gfx928",
        "gfx92a",
        "gfx936",
        "gfx938",
        "gfx946",
    ]

    # hipcc --offload-arch=native shells out to amdgpu-arch, which fails when
    # HSA cannot enumerate GPUs (KFD process limit). Resolve to a concrete gfx.
    resolved = []
    for arch in archs:
        if arch == "native":
            gfx = get_gfx()
            if not gfx or gfx == "native":
                raise RuntimeError(
                    "cannot determine GPU architecture; set GPU_ARCHS "
                    "(e.g. GPU_ARCHS=gfx938)"
                )
            arch = gfx.split(";")[0].strip()
        resolved.append(arch)

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in resolved
    ), f"One of GPU archs of {resolved} is invalid or not supported"
    return resolved


def get_code_object_version() -> Optional[str]:
    """Return the explicitly requested HIP code-object version, if any.

    AITER_CODE_OBJECT_VERSION is the public AITER entry point.  The CK variable
    is accepted as a compatibility alias so AITER JIT and CK subprocesses can
    share one value.  No default is imposed: older ROCm environments keep their
    existing compiler/runtime behavior unless the user opts in.
    """
    aiter_cov = os.getenv("AITER_CODE_OBJECT_VERSION", "").strip()
    ck_cov = os.getenv("CK_CODE_OBJECT_VERSION", "").strip()
    if aiter_cov and ck_cov and aiter_cov != ck_cov:
        raise ValueError(
            "AITER_CODE_OBJECT_VERSION and CK_CODE_OBJECT_VERSION disagree: "
            f"{aiter_cov!r} != {ck_cov!r}"
        )

    cov = aiter_cov or ck_cov
    if not cov:
        return None
    if cov not in {"4", "5", "6"}:
        raise ValueError(
            "HIP code-object version must be one of 4, 5, or 6; "
            f"got {cov!r}"
        )

    # Blob generators and third-party build subprocesses inherit the same
    # canonical value.  The resulting compiler flag is also part of the JIT
    # build arguments, so changing COV invalidates the extension cache key.
    os.environ["AITER_CODE_OBJECT_VERSION"] = cov
    os.environ["CK_CODE_OBJECT_VERSION"] = cov
    return cov


def module_vgpr_compiler_flag(module_name: str):
    archs = get_gfx().split(";")
    archs = [arch.strip() for arch in archs if arch.strip()]
    archs_768 = {"gfx936", "gfx938"}
    use_768 = [arch for arch in archs if arch in archs_768]
    use_512 = [arch for arch in archs if arch not in archs_768]
    if use_768 and use_512:
        logger.info(
            f"enable per-arch VGPR compiler flags for [{module_name}]: {archs}"
        )
        return " ".join(
            f"-Xarch_{arch} -mllvm=-support-"
            f"{768 if arch in archs_768 else 512}-vgprs=true"
            for arch in archs
        )
    vgpr_flag = "-support-768-vgprs=true" if use_768 else "-support-512-vgprs=true"
    return f" -mllvm {vgpr_flag} "


def module_jit_debug_flags(module_name: str):
    """Return opt-in HIP debug flags for a selected JIT module."""
    enabled = os.getenv("AITER_JIT_DEBUG", "0").strip().lower()
    if enabled not in ("1", "true", "yes", "on"):
        return []

    module_filter = os.getenv("AITER_JIT_DEBUG_MODULES", "").strip()
    if module_filter:
        selected_modules = {
            name for name in re.split(r"[,;:\s]+", module_filter) if name
        }
        if "*" not in selected_modules and module_name not in selected_modules:
            return []

    flags = shlex.split(os.getenv("AITER_JIT_DEBUG_FLAGS", "-g"))
    if flags:
        logger.info("enable JIT debug flags %s for [%s]", flags, module_name)
    return flags


def module_gfx936_grouped_gemm_selected_enabled() -> bool:
    """Mirror CK's gfx936 selected-instance build boundary."""
    archs = {arch.strip() for arch in get_gfx().split(";") if arch.strip()}
    return "gfx936" in archs and archs.isdisjoint({"gfx938", "gfx946"})


@functools.lru_cache()
def hip_flag_checker(flag_hip: str, archs=(), hipcc_path: Optional[str] = None) -> bool:
    hipcc = hipcc_path or executable_path("hipcc")
    concrete = [a for a in archs if a and a != "native"]
    cmd = [hipcc]
    cmd += [f"--offload-arch={arch}" for arch in concrete]
    cmd += shlex.split(str(flag_hip))
    cmd += ["-x", "hip", "-E", "-P", "/dev/null", "-o", "/dev/null"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return True

    logger.warning(
        "HIP flag probe failed: rc=%s command=%s\nstdout:\n%s\nstderr:\n%s",
        result.returncode,
        shlex.join(cmd),
        result.stdout,
        result.stderr,
    )
    return False

def _path_under_prefix(path: str, prefix: str) -> bool:
    if not path:
        return False
    rp = os.path.realpath(path)
    rprefix = os.path.realpath(prefix)
    try:
        common = os.path.commonpath([rp, rprefix])
    except ValueError:
        return False
    return common == rprefix


_DTK_VERSION_PATTERN = re.compile(r"(?:DTK[-_ ]*)?(\d{2})[._](\d{2})", re.IGNORECASE)


def _parse_dtk_version(value: str) -> Optional[tuple[int, int]]:
    match = _DTK_VERSION_PATTERN.search(value or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


@functools.lru_cache(maxsize=1)
def get_dtk_version() -> Optional[tuple[int, int]]:
    """Return the installed DTK major/minor without inferring it from HIP.

    DTK installations expose ``/opt/dtk/.dtk_version``. Versioned DTK paths
    are retained as a fallback, and ``AITER_DTK_VERSION`` is an explicit build
    override for packaging environments that relocate the toolkit.
    """
    candidates = [os.getenv("AITER_DTK_VERSION", "")]
    version_files = ["/opt/dtk/.dtk_version"]
    rocm_path = os.getenv("ROCM_PATH", "")
    if rocm_path:
        version_files.append(os.path.join(rocm_path, ".dtk_version"))

    for version_file in version_files:
        try:
            with open(version_file, encoding="utf-8") as stream:
                candidates.append(stream.read(256))
        except OSError:
            pass

    for executable in (shutil.which("aicc"), shutil.which("hipcc"), rocm_path):
        if executable:
            candidates.append(os.path.realpath(executable))

    for candidate in candidates:
        version = _parse_dtk_version(candidate)
        if version is not None:
            return version
    return None


@functools.lru_cache(maxsize=1)
def detect_dtk_env() -> bool:
    # Simplified detection logic:
    # 1) If 'aicc' is present (in PATH or at /opt/dtk/bin/aicc), treat it as hipcc alias and use it for compilation.
    # 2) Otherwise fall back to the normal hipcc resolution (executable_path("hipcc")).
    # DTK environment is determined when the selected hipcc (or ROCM_PATH) is under /opt/dtk.

    # Try to locate 'aicc' first (DTK's renamed hipcc)
    aicc_path = shutil.which("aicc")
    if not aicc_path:
        candidate = "/opt/dtk/bin/aicc"
        if os.path.exists(candidate):
            aicc_path = os.path.realpath(candidate)

    hipcc = ""
    hipcc_in_dtk = False

    if aicc_path:
        # Use aicc as the hipcc implementation by exporting HIPCC so other code that calls executable_path("hipcc")
        # will pick up the aicc binary.
        hipcc = os.path.realpath(aicc_path)
        os.environ["HIPCC"] = hipcc
        hipcc_in_dtk = _path_under_prefix(hipcc, "/opt/dtk")
        logger.info(f"Found 'aicc' and using it as hipcc: {hipcc}")
    else:
        # Fallback to normal hipcc resolution (may raise/abort in executable_path)
        try:
            hipcc = executable_path("hipcc")
        except Exception:
            # If executable_path fails, try a best-effort lookup via shutil.which
            hipcc = shutil.which("hipcc") or ""
            if hipcc:
                hipcc = os.path.realpath(hipcc)
        hipcc_in_dtk = _path_under_prefix(hipcc, "/opt/dtk") if hipcc else False

    # Also consider ROCM_PATH pointing under /opt/dtk
    rocm_path = os.getenv("ROCM_PATH", "")
    rocm_in_dtk = _path_under_prefix(rocm_path, "/opt/dtk")

    enabled = hipcc_in_dtk or rocm_in_dtk
    if enabled:
        logger.info(
            f"DTK environment detected (hipcc={hipcc}, ROCM_PATH={rocm_path}), enabling -DDTK_ENV"
        )
    else:
        logger.info(
            f"Non-DTK environment (hipcc={hipcc}, ROCM_PATH={rocm_path}), DTK_ENV disabled"
        )
    return enabled


def find_optional_hipcc() -> Optional[str]:
    """Return a real hipcc driver path without consulting the HIPCC override.

    DTK's hipcc is a driver wrapper whose argv[0] selects HIP mode, so keep the
    wrapper path rather than resolving it to the underlying clang executable.
    This helper is only used by modules that explicitly opt in.
    """
    candidates = ["/opt/dtk/bin/hipcc", shutil.which("hipcc")]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)
    return None

def check_and_set_ninja_worker():
    max_num_jobs_cores = max(1, os.cpu_count() * 0.8)
    import psutil

    # calculate the maximum allowed NUM_JOBS based on free memory
    free_memory_gb = psutil.virtual_memory().available / (1024**3)  # free memory in GB
    max_num_jobs_memory = int(free_memory_gb / 0.5)  # assuming 0.5 GB per job

    # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
    max_jobs = int(max(1, min(max_num_jobs_cores, max_num_jobs_memory)))
    max_jobs_env = os.environ.get("MAX_JOBS")
    if max_jobs_env is not None:
        try:
            max_processes = int(max_jobs_env)
            # too large value
            if max_processes > max_jobs:
                os.environ["MAX_JOBS"] = str(max_jobs)
        # error value
        except ValueError:
            os.environ["MAX_JOBS"] = str(max_jobs)
    # none value
    else:
        os.environ["MAX_JOBS"] = str(max_jobs)


def rename_cpp_to_cu(els, dst, hipify, recursive=False):
    def do_rename_and_mv(name, src, dst, ret):
        newName = name
        if hipify:
            if name.endswith(".cpp") or name.endswith(".cu"):
                newName = name.replace(".cpp", ".cu")
                ret.append(f"{dst}/{newName}")
            shutil.copy(f"{src}/{name}", f"{dst}/{newName}")
        else:
            if name.endswith(".cpp") or name.endswith(".cu"):
                ret.append(f"{src}/{newName}")

    ret = []
    for el in els:
        if not os.path.exists(el):
            logger.warning(f"---> {el} not exists!!!!!!")
            continue
        if os.path.isdir(el):
            for entry in os.listdir(el):
                if os.path.isdir(f"{el}/{entry}"):
                    if recursive:
                        ret += rename_cpp_to_cu(
                            [f"{el}/{entry}"], dst, hipify, recursive
                        )
                    continue
                do_rename_and_mv(entry, el, dst, ret)
        else:
            do_rename_and_mv(os.path.basename(el), os.path.dirname(el), dst, ret)
    return ret


@torch_compile_guard()
def check_numa_custom_op() -> None:
    numa_balance_set = os.popen("cat /proc/sys/kernel/numa_balancing").read().strip()
    if numa_balance_set == "1":
        logger.debug(
            "WARNING: NUMA balancing is enabled, which may cause errors. "
            "It is recommended to disable NUMA balancing by running \"sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'\" "
        )


@functools.lru_cache()
def check_numa():
    check_numa_custom_op()


__mds = {}


@torch_compile_guard()
def get_module_custom_op(md_name: str) -> None:
    global __mds
    if md_name not in __mds:
        if "AITER_JIT_DIR" in os.environ:
            __mds[md_name] = importlib.import_module(md_name)
        else:
            __mds[md_name] = importlib.import_module(f"{__package__}.{md_name}")

        if AITER_LOG_MORE:
            logger.info(f"import [{md_name}] under {__mds[md_name].__file__}")
    return


@functools.lru_cache(maxsize=1024)
def get_module(md_name):
    check_numa()
    get_module_custom_op(md_name)
    return __mds[md_name]


rebuilded_list = ["module_aiter_enum"]


def rm_module(md_name):
    os.system(f"rm -rf {get_user_jit_dir()}/{md_name}.so")


def clear_build(md_name):
    os.system(f"rm -rf {bd_dir}/{md_name}")


def build_module(
    md_name,
    srcs,
    flags_extra_cc,
    flags_extra_hip,
    blob_gen_cmd,
    extra_include,
    extra_ldflags,
    verbose,
    is_python_module,
    is_standalone,
    torch_exclude,
    hipify=False,
    prefer_hipcc=False,
):
    lock_path = f"{bd_dir}/lock_{md_name}"
    startTS = time.perf_counter()
    target_name = f"{md_name}.so" if not is_standalone else md_name

    def MainFunc():
        if AITER_REBUILD == 1:
            rm_module(md_name)
            clear_build(md_name)
        elif AITER_REBUILD >= 2:
            rm_module(md_name)
        op_dir = f"{bd_dir}/{md_name}"
        logger.info(f"start build [{md_name}] under {op_dir}")

        opbd_dir = f"{op_dir}/build"
        src_dir = f"{op_dir}/build/srcs"
        os.makedirs(src_dir, exist_ok=True)

        if os.path.exists(f"{get_user_jit_dir()}/{target_name}"):
            os.remove(f"{get_user_jit_dir()}/{target_name}")
            
        sources = rename_cpp_to_cu(srcs, src_dir, hipify)

        flags_cc = ["-O3", "-std=c++20"]
        flags_hip = [
            # "-DLEGACY_HIPBLAS_DIRECT",
            "-DUSE_PROF_API=1",
            "-D__HIP_PLATFORM_HCC__=1",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-mllvm --amdgpu-kernarg-preload-count=16",
            # "-v --save-temps",
            "-Wno-unused-result",
            "-Wno-switch-bool",
            "-Wno-vla-cxx-extension",
            "-Wno-undefined-func-template",
            "-Wno-macro-redefined",
            # "-Wno-missing-template-arg-list-after-template-kw",
            "-fgpu-flush-denormals-to-zero",
        ]

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        hip_version = parse(get_hip_version().split()[-1].rstrip("-").replace("-", "+"))
        if hip_version > Version("5.5.00000"):
            flags_hip += ["-mllvm --lsr-drop-solution=1"]
        if hip_version > Version("5.7.23302"):
            flags_hip += ["-fno-offload-uniform-block"]
        if hip_version > Version("6.1.40090"):
            flags_hip += ["-mllvm -enable-post-misched=0"]
        if hip_version > Version("6.2.41132"):
            flags_hip += [
                "-mllvm -amdgpu-early-inline-all=true",
                "-mllvm -amdgpu-function-calls=false",
            ]
        if hip_version > Version("6.2.41133"):
            flags_hip += ["-mllvm -amdgpu-coerce-illegal-types=1"]
        if get_gfx() == "gfx946" and int(os.getenv("AITER_FP4x2", "1")) > 0:
            flags_hip += ["-D__Float4_e2m1fn_x2"]

        if not torch_exclude:
            import torch

            if hasattr(torch, "float4_e2m1fn_x2"):
                flags_hip += ["-DTORCH_Float4_e2m1fn_x2"]

        # Preserve the existing aicc-first DTK detection for every module.
        if detect_dtk_env():
            flags_cc.append("-DDTK_ENV")
            flags_hip.append("-DDTK_ENV")
            dtk_version = get_dtk_version()
            if dtk_version is not None:
                dtk_major, dtk_minor = dtk_version
                dtk_version_flags = [
                    f"-DAITER_DTK_VERSION_MAJOR={dtk_major}",
                    f"-DAITER_DTK_VERSION_MINOR={dtk_minor}",
                ]
                flags_cc.extend(dtk_version_flags)
                flags_hip.extend(dtk_version_flags)
                logger.info("Detected DTK version %d.%02d", dtk_major, dtk_minor)
            else:
                logger.info(
                    "DTK version is unknown; version-gated capabilities remain disabled"
                )

        module_hipcc = find_optional_hipcc() if prefer_hipcc else None
        if prefer_hipcc:
            if module_hipcc:
                logger.info(
                    "module [%s] selected optional hipcc compiler: %s",
                    md_name,
                    module_hipcc,
                )
            else:
                logger.info(
                    "module [%s] did not find executable hipcc; falling back to aicc",
                    md_name,
                )

        flags_cc += flags_extra_cc
        flags_hip += flags_extra_hip
        flags_hip += module_jit_debug_flags(md_name)
        # HIP>6.1 defaults to -enable-post-misched=0. A module can opt back
        # in with -enable-post-misched=1; drop the =0 copy so uniquing cannot
        # keep both (LLVM would then follow whichever flag comes last).
        if any("enable-post-misched=1" in str(f) for f in flags_extra_hip):
            flags_hip = [
                f for f in flags_hip if "enable-post-misched=0" not in str(f)
            ]
        archs = validate_and_update_archs()
        flags_hip += [f"--offload-arch={arch}" for arch in archs]
        parallel_jobs = os.getenv("AITER_HIP_PARALLEL_JOBS")
        if parallel_jobs is not None:
            if not parallel_jobs.isdigit() or int(parallel_jobs) < 1:
                raise ValueError("AITER_HIP_PARALLEL_JOBS must be a positive integer")
            flags_hip.append(f"-parallel-jobs={int(parallel_jobs)}")
        if "gfx92a" in archs:
            # Hygon gfx92a exposes torch.float8_e4m3fn (OCP FP8). Keep CK
            # Tile's device-side numeric limits and software conversion in
            # the same format as the public Torch dtype.
            flags_hip.append("-DCK_TILE_USE_OCP_FP8=1")
        code_object_version = get_code_object_version()
        if code_object_version is not None:
            flags_hip.append(f"-mcode-object-version={code_object_version}")
            logger.info(
                "enable HIP code-object version %s for [%s]",
                code_object_version,
                md_name,
            )
        fp8_archs = {"gfx938", "gfx92a"}
        enable_fp8 = (
            int(os.getenv("AITER_ENABLE_FP8", "0")) > 0
            or any(arch in fp8_archs for arch in archs)
            or get_gfx() in fp8_archs
        )
        if enable_fp8:
            flags_hip.append("-DGPU_ENABLE_FP8")   # device
            flags_cc.append("-DGPU_ENABLE_FP8")    # host

        flags_hip = sorted(set(flags_hip))  # remove same flags
        checked_flags_hip = []
        for flag_hip in flags_hip:
            if hip_flag_checker(flag_hip, tuple(archs), module_hipcc):
                checked_flags_hip.append(flag_hip)
            elif flag_hip.startswith("--offload-arch=") or "-Xarch_" in flag_hip:
                raise RuntimeError(
                    "required GPU target/compiler flag is not supported: "
                    f"{hcu_sanitize_display(flag_hip)}"
                )
        flags_hip = checked_flags_hip
        check_and_set_ninja_worker()

        def exec_blob(blob_gen_cmd, op_dir, src_dir, sources):
            if blob_gen_cmd:
                blob_dir = f"{op_dir}/blob"
                os.makedirs(blob_dir, exist_ok=True)
                if AITER_LOG_MORE:
                    logger.info(f"exec_blob ---> {PY} {blob_gen_cmd.format(blob_dir)}")
                os.system(f"{PY} {blob_gen_cmd.format(blob_dir)}")
                sources += rename_cpp_to_cu([blob_dir], src_dir, hipify, recursive=True)
            return sources

        if isinstance(blob_gen_cmd, list):
            for s_blob_gen_cmd in blob_gen_cmd:
                sources = exec_blob(s_blob_gen_cmd, op_dir, src_dir, sources)
        else:
            sources = exec_blob(blob_gen_cmd, op_dir, src_dir, sources)

        extra_include_paths = [
            f"{CK_DIR}/include",
            f"{CK_DIR}/library/include",
        ]
        if not hipify:
            extra_include_paths += [
                f"{AITER_CSRC_DIR}/include",
                f"{op_dir}/blob",
            ] + extra_include
            if not is_standalone:
                extra_include_paths += [f"{AITER_CSRC_DIR}/include/torch"]
        else:
            old_bd_include_dir = f"{op_dir}/build/include"
            extra_include_paths.append(old_bd_include_dir)
            os.makedirs(old_bd_include_dir, exist_ok=True)
            rename_cpp_to_cu(
                [f"{AITER_CSRC_DIR}/include"] + extra_include,
                old_bd_include_dir,
                hipify,
            )

            if not is_standalone:
                bd_include_dir = f"{op_dir}/build/include/torch"
                os.makedirs(bd_include_dir, exist_ok=True)
                rename_cpp_to_cu(
                    [f"{AITER_CSRC_DIR}/include/torch"],
                    bd_include_dir,
                    hipify,
                )

        try:
            _jit_compile(
                md_name,
                sorted(set(sources)),
                extra_cflags=flags_cc,
                extra_cuda_cflags=flags_hip,
                extra_ldflags=extra_ldflags,
                extra_include_paths=extra_include_paths,
                build_directory=opbd_dir,
                verbose=verbose or AITER_LOG_MORE > 1,
                with_cuda=True,
                is_python_module=is_python_module,
                is_standalone=is_standalone,
                torch_exclude=torch_exclude,
                hipify=hipify,
                hipcc_path=module_hipcc,
            )
            if is_python_module and not is_standalone:
                shutil.copy(f"{opbd_dir}/{target_name}", f"{get_user_jit_dir()}")
            else:
                shutil.copy(
                    f"{opbd_dir}/{target_name}", f"{AITER_ROOT_DIR}/op_tests/cpp/mha"
                )
        except Exception as e:
            tag = f"\033[31mfailed jit build [{md_name}]\033[0m"
            history = re.sub(
                "error:",
                "\033[31merror:\033[0m",
                "-->".join(traceback.format_exception(*sys.exc_info())),
                flags=re.I,
            )
            history = hcu_sanitize_display(history)
            logger.error(
                f"{tag}\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\n-->[History]: {{}}{tag}\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191".format(
                    history,
                )
            )
            raise SystemExit(
                f"[aiter] build [{md_name}] under {opbd_dir} failed !!!!!!"
            ) from e

    def FinalFunc():
        logger.info(
            f"\033[32mfinish build [{md_name}], cost {time.perf_counter()-startTS:.1f}s \033[0m"
        )

    mp_lock(lockPath=lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)


def get_args_of_build(
    ops_name: str, exclude=[], build_module_name: Optional[str] = None
):
    d_opt_build_args = {
        "srcs": [],
        "md_name": "",
        "flags_extra_cc": [],
        "flags_extra_hip": [],
        "extra_ldflags": None,
        "extra_include": [],
        "verbose": False,
        "hipify": False,
        "is_python_module": True,
        "is_standalone": False,
        "torch_exclude": False,
        "hip_clang_path": None,
        "prefer_hipcc": False,
        "blob_gen_cmd": "",
        "skip_if": False,
    }

    def convert(d_ops: dict, module_name: str):
        converted_ops = {}
        for k, val in d_ops.items():
            if isinstance(val, list):
                converted_list = []
                for el in val:
                    if isinstance(el, str):
                        if "torch" in el:
                            import torch as torch
                        converted_el = eval(el)
                    else:
                        converted_el = el
                    # Build configs may gate a source or flag by returning
                    # None. Do not pass an empty placeholder to the compiler.
                    if converted_el is not None:
                        converted_list.append(converted_el)
                converted_ops[k] = converted_list
            elif isinstance(val, str):
                converted_ops[k] = eval(val)
            else:
                converted_ops[k] = val

        # undefined compile features will be replaced with default value
        resolved_build_args = copy.deepcopy(d_opt_build_args)
        resolved_build_args.update(converted_ops)
        return resolved_build_args

    with open(this_dir + "/optCompilerConfig.json", "r") as file:
        data = json.load(file)
        if isinstance(data, dict):
            # parse all ops, return list
            if ops_name == "all":
                all_ops_list = []
                d_all_ops = {
                    "flags_extra_cc": [],
                    "flags_extra_hip": [],
                    "extra_include": [],
                    "extra_ldflags": [],
                    "blob_gen_cmd": [],
                }
                # traverse opts
                for ops_name, d_ops in data.items():
                    # Cannot contain tune ops
                    if ops_name.endswith("tune"):
                        continue
                    # exclude
                    if ops_name in exclude:
                        continue
                    single_ops = convert(d_ops, ops_name)
                    d_single_ops = {
                        "md_name": ops_name,
                        "srcs": single_ops["srcs"],
                        "flags_extra_cc": single_ops["flags_extra_cc"],
                        "flags_extra_hip": single_ops["flags_extra_hip"],
                        "extra_include": single_ops["extra_include"],
                        "extra_ldflags": single_ops["extra_ldflags"],
                        "blob_gen_cmd": single_ops["blob_gen_cmd"],
                        "verbose": single_ops["verbose"],
                        "hipify": single_ops["hipify"],
                        "skip_if": single_ops.get("skip_if", False),
                    }
                    for k in d_all_ops.keys():
                        if isinstance(single_ops[k], list):
                            d_all_ops[k] += single_ops[k]
                        elif isinstance(single_ops[k], str) and single_ops[k] != "":
                            d_all_ops[k].append(single_ops[k])
                    all_ops_list.append(d_single_ops)

                return all_ops_list, d_all_ops
            # no find opt_name in json.
            elif data.get(ops_name) is None:
                logger.warning(
                    "Not found this operator ("
                    + ops_name
                    + ") in 'optCompilerConfig.json'. "
                )
                return d_opt_build_args
            # parser single opt
            else:
                compile_ops_ = data.get(ops_name)
                return convert(compile_ops_, build_module_name or ops_name)
        else:
            logger.warning(
                "ERROR: pls use dict_format to write 'optCompilerConfig.json'! "
            )

def compile_ops(
    _md_name: str,
    fc_name: Optional[str] = None,
    gen_func: Optional[Callable[..., dict[str, Any]]] = None,
    gen_fake: Optional[Callable[..., Any]] = None,
    develop: bool = False,
):
    def decorator(func):
        func.arg_checked = False

        @functools.wraps(func)
        def wrapper(*args, custom_build_args={}, **kwargs):
            loadName = fc_name
            md_name = _md_name
            if fc_name is None:
                loadName = func.__name__
            try:
                module = None
                if gen_func is not None:
                    custom_build_args.update(gen_func(*args, **kwargs))
                elif AITER_REBUILD and md_name not in rebuilded_list:
                    rebuilded_list.append(md_name)
                    raise ModuleNotFoundError("start rebuild")
                if module is None:
                    try:
                        module = get_module(md_name)
                    except Exception as e:
                        md = custom_build_args.get("md_name", md_name)
                        module = get_module(md)
            except ModuleNotFoundError:
                build_module_name = custom_build_args.get("md_name", md_name)
                d_args = get_args_of_build(
                    md_name, build_module_name=build_module_name
                )
                d_args.update(custom_build_args)

                if d_args.get("skip_if", False):
                    logger.info(f"skip build [{md_name}] due to skip_if condition")
                    return None

                # update module if we have coustom build
                md_name = build_module_name

                srcs = d_args["srcs"]
                flags_extra_cc = d_args["flags_extra_cc"]
                flags_extra_hip = d_args["flags_extra_hip"]
                blob_gen_cmd = d_args["blob_gen_cmd"]
                extra_include = d_args["extra_include"]
                extra_ldflags = d_args["extra_ldflags"]
                verbose = d_args["verbose"]
                is_python_module = d_args["is_python_module"]
                is_standalone = d_args["is_standalone"]
                torch_exclude = d_args["torch_exclude"]
                hipify = d_args.get("hipify", False)
                prefer_hipcc = d_args.get("prefer_hipcc", False)
                hip_clang_path = d_args.get("hip_clang_path", None)
                prev_hip_clang_path = None
                if hip_clang_path is not None and os.path.exists(hip_clang_path):
                    prev_hip_clang_path = os.environ.get("HIP_CLANG_PATH", None)
                    os.environ["HIP_CLANG_PATH"] = hip_clang_path
                build_module(
                    md_name,
                    srcs,
                    flags_extra_cc,
                    flags_extra_hip,
                    blob_gen_cmd,
                    extra_include,
                    extra_ldflags,
                    verbose,
                    is_python_module,
                    is_standalone,
                    torch_exclude,
                    hipify,
                    prefer_hipcc,
                )

                if hip_clang_path is not None:
                    if prev_hip_clang_path is not None:
                        os.environ["HIP_CLANG_PATH"] = prev_hip_clang_path
                    else:
                        os.environ.pop("HIP_CLANG_PATH", None)

                if is_python_module:
                    module = get_module(md_name)
                if md_name not in __mds:
                    __mds[md_name] = module

            if isinstance(module, types.ModuleType):
                op = getattr(module, loadName)
            else:
                return None

            def check_args():
                get_asm_dir()
                import inspect
                import re

                import torch

                enum_types = ["ActivationType", "QuantType"]

                if not op.__doc__.startswith("Members:"):
                    doc_str = op.__doc__.split("\n")[0]
                    doc_str = re.sub(r"<(.*?)\:.*?>", r"\g<1>", doc_str)
                    doc_str = doc_str.replace("list[", "List[")
                    doc_str = doc_str.replace("tuple[", "Tuple[")
                    doc_str = doc_str.replace("collections.abc.Sequence[", "List[")
                    doc_str = doc_str.replace("typing.SupportsInt", "int")
                    doc_str = doc_str.replace("typing.SupportsFloat", "float")
                    # A|None  -->  Optional[A]
                    pattern = r"([\w\.]+(?:\[[^\]]+\])?)\s*\|\s*None"
                    doc_str = re.sub(pattern, r"Optional[\1]", doc_str)
                    for el in enum_types:
                        doc_str = re.sub(f" aiter.*{el} ", f" {el} ", doc_str)
                    try:
                        from ..utility.aiter_types import aiter_tensor_t as _aiter_tensor_t
                    except ImportError:
                        _aiter_tensor_t = None
                    namespace = {
                        "List": List,
                        "Optional": Optional,
                        "torch": torch,
                        "typing": typing,
                    }
                    if _aiter_tensor_t is not None:
                        namespace["aiter_tensor_t"] = _aiter_tensor_t

                    exec(
                        f"from aiter import*\ndef {doc_str}: pass",
                        namespace,
                    )
                    foo = namespace[doc_str.split("(")[0]]
                    sig = inspect.signature(foo)
                    func.__signature__ = sig
                    ann = {k: v.annotation for k, v in sig.parameters.items()}
                    ann["return"] = sig.return_annotation
                    callargs = inspect.getcallargs(func, *args, **kwargs)
                    for el, arg in callargs.items():
                        expected_type = ann[el]
                        got_type = type(arg)
                        origin = typing.get_origin(expected_type)
                        sub_t = typing.get_args(expected_type)

                        if origin is None:
                            if not isinstance(arg, expected_type) and not (
                                # aiter_enum can be int
                                any(el in str(expected_type) for el in enum_types)
                                and isinstance(arg, int)
                            ):
                                raise TypeError(
                                    f"{loadName}: {el} needs to be {expected_type} but got {got_type}"
                                )
                        elif origin is list:
                            if (
                                not isinstance(arg, list)
                                # or not all(isinstance(i, sub_t) for i in arg)
                            ):
                                raise TypeError(
                                    f"{loadName}: {el} needs to be List[{sub_t}] but got {arg}"
                                )
                        elif origin is typing.Union or origin is types.UnionType:
                            if arg is not None and not isinstance(arg, sub_t):
                                raise TypeError(
                                    f"{loadName}: {el} needs to be Optional[{sub_t}] but got {arg}"
                                )
                        else:
                            raise TypeError(f"Unsupported type: {expected_type}")

                    func_hints = typing.get_type_hints(func)
                    if ann["return"] is None:
                        func_hints["return"] = None
                    # if ann != func_hints:
                    #     logger.warning(
                    #         f"type hints mismatch, override to --> {doc_str}"
                    #     )
                return True

            if not func.arg_checked:
                if develop:
                    func.arg_checked = True  # skip type-check when develop=True; tensors are converted below
                else:
                    func.arg_checked = check_args()

            if AITER_LOG_MORE == 2:
                from ..test_common import log_args

                log_args(func, *args, **kwargs)

            # develop=True: convert torch.Tensor → pybind aiter_tensor_t and inject HIP stream.
            # develop=False (default): all existing ops pass through unchanged.
            if develop:
                import torch
                from ..utility.dtypes import torch_to_aiter_pybind

                args = tuple(
                    torch_to_aiter_pybind(a) if isinstance(a, torch.Tensor) else a
                    for a in args
                )
                kwargs = {
                    k: (torch_to_aiter_pybind(v) if isinstance(v, torch.Tensor) else v)
                    for k, v in kwargs.items()
                }
                module._set_current_hip_stream(
                    torch.cuda.current_stream().cuda_stream
                )

            return op(*args, **kwargs)

        @torch_compile_guard(device="cuda", gen_fake=gen_fake, calling_func_=func)
        def custom_wrapper(*args, **kwargs):
            return wrapper(*args, **kwargs)

        return custom_wrapper

    return decorator
