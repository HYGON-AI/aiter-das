#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
用法：bash rebuild_aiter.sh [wheel|develop]
  wheel    清理历史编译产物，在当前工程快速全量构建 wheel，输出到 dist/（默认）。
  develop  保留历史编译产物，卸载已安装的 aiter，然后可编辑安装当前工程。
           安装时不全量预编译，已有库继续复用，缺失算子按需 JIT 编译。
默认保留 gfx936 和 gfx938；可用 GPU_ARCHS 覆盖目标架构。
EOF
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi
mode=${1:-wheel}
case "$mode" in
    wheel|develop) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "$source_root"
test -f setup.py
test -f aiter/jit/optCompilerConfig.json

# 两种模式共用快速构建参数；不复制整份源码，也不依赖个人工作目录。
# 默认并发面向大型编译服务器，可按可用 CPU 和内存调整。
# 三项参数分别控制模块并发、每模块 Ninja 作业数和 HIP 编译器内部并行度。
export GPU_ARCHS="${GPU_ARCHS:-gfx936;gfx938}"
export AITER_PREBUILD_LOG_PROGRESS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset AITER_JIT_DIR AITER_REBUILD PREBUILD_THREAD_NUM MAX_JOBS
export AITER_PREBUILD_THREADS=${AITER_PREBUILD_THREADS:-16}
# 保留原先每模块的 CPU 预算：176 个 CPU 对应 28，256 个 CPU 对应 40。
export AITER_NINJA_JOBS=${AITER_NINJA_JOBS:-$(python -c 'import os; print(max(1, int(os.cpu_count() * 0.8) // 5))')}
export AITER_HIP_PARALLEL_JOBS=${AITER_HIP_PARALLEL_JOBS:-2}
if [[ -f /opt/dtk/include/hipdnn/sdk/nlohmann/json.hpp ]]; then
    export CPLUS_INCLUDE_PATH="/opt/dtk/include/hipdnn/sdk${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi
# pip 的元数据阶段未必带 editable 参数，显式传递模式供 setup.py 识别。
export AITER_BUILD_MODE="$mode"
started=$SECONDS

if [[ $mode == wheel ]]; then
    echo "### [1/3] 清理历史编译产物"
    # 与原全量构建保持相同清理范围；保留源码和 dist/ 下已有的 wheel。
    rm -rf -- aiter/jit/aiter_.so aiter/jit/build aiter/jit/*.so \
        aiter/jit/jit/build build aiter.egg-info aiter_meta
    export PREBUILD_KERNELS=1
else
    echo "### [1/3] 保留历史编译产物，准备开发安装"
    export PREBUILD_KERNELS=0
fi

# wheel 清理完成后再创建日志目录，避免将本轮日志一起删除。
mkdir -p build
run_dir=$(mktemp -d "$source_root/build/rebuild.$mode.XXXXXX")
build_log="$run_dir/build.log"
export TMPDIR="$run_dir/tmp"
mkdir -p "$TMPDIR"
echo "Source: $source_root"
echo "Mode: $mode"
echo "GPU_ARCHS=$GPU_ARCHS modules=$AITER_PREBUILD_THREADS ninja=$AITER_NINJA_JOBS hip=$AITER_HIP_PARALLEL_JOBS"
echo "Build log: $build_log"

run_logged() {
    # 完整输出同步写入日志，终端显示模块进度、打包/安装阶段及错误。
    # 设置 AITER_PREBUILD_VERBOSE=1 可显示全部输出；pipefail 保证失败状态不会被管道吞掉。
    if ! "$@" 2>&1 | tee -a "$build_log" | python -u -c '
import os, re, sys
total = completed = 0
verbose = os.getenv("AITER_PREBUILD_VERBOSE", "0") != "0"
for line in sys.stdin:
    message = line.lstrip()
    if message.startswith("[aiter-prebuild]"):
        match = re.search(r"total_modules=(\d+)", message)
        if match:
            total = int(match.group(1))
        if re.match(r"\[aiter-prebuild\] (DONE|SKIP) ", message) and total:
            completed += 1
            line = line.rstrip() + f" [{completed}/{total}]\n"
        print(line, end="", flush=True)
    elif verbose or re.match(r"running |creating .*\.whl|Obtaining |Preparing |Building |Created |Installing |Successfully |Found existing |Uninstalling |Skipping |AITER source:", message) or re.search(r"Traceback|Error:|Exception:|ERROR|FAILED:|error:|Warning:|WARNING|warning:", line):
        print(line, end="", flush=True)
    '; then
        echo "构建或安装失败，完整日志：$build_log" >&2
        tail -n 80 "$build_log" >&2
        return 1
    fi
}

if [[ $mode == wheel ]]; then
    echo "### [2/3] 快速全量编译并打包 wheel"
    run_logged python -u setup.py bdist_wheel
    echo "### [3/3] wheel 已生成到工程 dist/"
    sha256sum "$source_root"/dist/*.whl
else
    echo "### [2/3] 卸载已安装的 aiter，并可编辑安装当前工程"
    # 离开源码目录并排除 PYTHONPATH 干扰，避免本地 egg-info 掩盖已安装的包。
    # 使用同一 Python 的 pip，卸载失败会立即退出，不继续执行安装。
    (cd "$run_dir"; run_logged env -u PYTHONPATH python -m pip uninstall aiter -y)
    run_logged python -m pip install -v -e . --no-build-isolation --no-deps
    echo "### [3/3] 核对可编辑安装指向当前源码"
    # 离开源码根目录再检查，避免当前目录优先级掩盖错误的安装路径。
    cd "$run_dir"
    run_logged python -c '
import importlib.util, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
spec = importlib.util.find_spec("aiter")
expected = root / "aiter" / "__init__.py"
if spec is None or spec.origin is None or pathlib.Path(spec.origin).resolve() != expected:
    raise RuntimeError(f"aiter import does not resolve to {expected}: {spec}")
if (root / "aiter" / "install_mode").read_text().strip() != "develop":
    raise RuntimeError("aiter install_mode is not develop")
print(f"AITER source: {spec.origin}")
' "$source_root"
fi
echo "Elapsed: $((SECONDS-started)) seconds"
echo "Build log: $build_log"
