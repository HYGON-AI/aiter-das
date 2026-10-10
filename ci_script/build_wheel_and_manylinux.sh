#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Building AITER wheel in ${repo_root}"
cd "${repo_root}"

echo "Initializing AITER submodules from the current commit..."
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive

# das-build.sh is AITER's canonical build entry point.  It initializes the
# direct submodules and owns the GPU_ARCHS/PREBUILD_KERNELS build choices.
export PYTHONUNBUFFERED=1
export CPLUS_INCLUDE_PATH="/opt/dtk/include/hipdnn/sdk:${CPLUS_INCLUDE_PATH:-}"
bash rebuild_aiter.sh

wheel_count="$(find dist -maxdepth 1 -type f -name '*.whl' | wc -l)"
if [[ "${wheel_count}" -ne 1 ]]; then
  echo "Expected exactly one wheel in dist/ after rebuild_aiter.sh, found ${wheel_count}" >&2
  exit 1
fi

shopt -s nullglob
wheels=(dist/aiter*.whl)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one AITER wheel in dist/, found ${#wheels[@]}" >&2
  exit 1
fi

torch_path="$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")"
if [[ -z "${torch_path}" || ! -d "${torch_path}/lib" ]]; then
  echo "Unable to locate the PyTorch library directory" >&2
  exit 1
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${torch_path}/lib:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${torch_path}/lib"
fi

platform="manylinux_$(ldd --version | awk 'NR==1 {print $NF}' | tr '.' '_')_x86_64"
echo "Repairing ${wheels[0]} for ${platform}"
auditwheel show "${wheels[0]}"

excludes=(
  --exclude "libhsa-runtime64.so.1"
  --exclude "libgalaxyhip.so*"
  --exclude "libMIOpen.so*"
  --exclude "librccl.so*"
  --exclude "libhipblas.so*"
  --exclude "libhipfft.so*"
  --exclude "libhiprand.so*"
  --exclude "libhipsolver.so*"
  --exclude "libhipsparse.so*"
  --exclude "libhipnn.so*"
  --exclude "librocblas.so*"
  --exclude "librocsolver.so*"
  --exclude "librocfft.so*"
  --exclude "librocrand.so*"
  --exclude "librocsparse.so*"
  --exclude "librocm_smi64.so*"
  --exclude "librocfft-device-*.so*"
  --exclude "libc10.so*"
  --exclude "libc10_hip.so*"
  --exclude "libtorch_cpu.so*"
  --exclude "libtorch_hip.so*"
  --exclude "libtorch_python.so*"
  --exclude "libtorch.so*"
  --exclude "libtorchaudio*"
  --exclude "_torchaudio*"
  --exclude "libavcodec*"
  --exclude "libavdevice*"
  --exclude "libavfilter*"
  --exclude "libavformat*"
  --exclude "libavutil*"
  --exclude "libsox*"
  --exclude "libhipblaslt.so*"
  --exclude "libnuma.so*"
)

auditwheel repair --plat "${platform}" --strip "${excludes[@]}" -w dist "${wheels[0]}"

repaired=(dist/*manylinux*.whl)
if [[ "${#repaired[@]}" -ne 1 ]]; then
  echo "Expected exactly one repaired manylinux AITER wheel in dist/, found ${#repaired[@]}" >&2
  exit 1
fi
