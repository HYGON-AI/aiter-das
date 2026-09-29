#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

: "${TORCH_VERSION:?TORCH_VERSION must be set}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${DTK_PKG:-}" ]]; then
  cp -f "${DTK_PKG}" /opt/
  cd /opt || exit 1
  tar -xzf "$(basename "${DTK_PKG}")"
  rm -rf dtk
  mv dtk-* dtk
elif [[ ! -f /opt/dtk/env.sh ]]; then
  echo "DTK_PKG is unset and no DTK installation exists at /opt/dtk." >&2
  exit 1
fi
source /opt/dtk/env.sh

# GitHub Actions starts a fresh shell for each `run` step.  Preserve the DTK
# environment for the build and test steps when this script is sourced there.
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for env_name in PATH LD_LIBRARY_PATH LIBRARY_PATH CMAKE_PREFIX_PATH CPATH \
                  PKG_CONFIG_PATH HIP_PATH ROCM_PATH CPLUS_INCLUDE_PATH; do
    if [[ -v "${env_name}" ]]; then
      printf '%s=%s\n' "${env_name}" "${!env_name}" >> "${GITHUB_ENV}"
    fi
  done
fi

cd "${repo_root}" || exit 1

# Install the newest AICC package available on the Runner's shared storage.
# Its absence is non-fatal because some Runners include AICC already.
#
# AICC filename format:
#   dtk_llvm_<commit>_<YYYYMMDD[HHMM]>.run
#
# Set AICC_NIGHTLY_DIR in the self-hosted Runner configuration when a newer
# AICC package should be installed.  Leaving it unset uses the AICC already
# installed on the Runner.
aicc_dir="${AICC_NIGHTLY_DIR:-}"
latest_aicc_run=""
latest_aicc_timestamp=""

shopt -s nullglob
for aicc_candidate in ${aicc_dir:+"${aicc_dir}"/dtk_llvm_*.run}; do
  aicc_candidate_name="${aicc_candidate##*/}"
  aicc_timestamp="${aicc_candidate_name%.run}"
  aicc_timestamp="${aicc_timestamp##*_}"

  if [[ ! "${aicc_timestamp}" =~ ^[0-9]{8}([0-9]{4})?$ ]]; then
    echo "Warning: skipping AICC installer with an invalid timestamp: ${aicc_candidate_name}" >&2
    continue
  fi

  if [[ -z "${latest_aicc_timestamp}" ]] ||
     (( 10#${aicc_timestamp} > 10#${latest_aicc_timestamp} )); then
    latest_aicc_timestamp="${aicc_timestamp}"
    latest_aicc_run="${aicc_candidate}"
  fi
done
shopt -u nullglob

if [[ -n "${latest_aicc_run}" ]]; then
  aicc_installer="${latest_aicc_run##*/}"

  echo "Selected AICC installer: ${latest_aicc_run}"
  echo "Selected AICC timestamp: ${latest_aicc_timestamp}"

  cp "${latest_aicc_run}" "/opt/${aicc_installer}"
  cd /opt || exit 1
  chmod +x "${aicc_installer}"

  # GitLab Runner enables pipefail. When the installer exits, `yes` may get
  # SIGPIPE and make the whole pipeline look failed even though installation
  # succeeded. Temporarily disable pipefail and keep the installer's status.
  set +o pipefail
  yes | bash "${aicc_installer}"
  aicc_status="${PIPESTATUS[1]}"
  set -o pipefail

  if [[ "${aicc_status}" -ne 0 ]]; then
    echo "AICC installation failed: ${aicc_installer}" >&2
    exit "${aicc_status}"
  fi

  echo "AICC installation completed: ${aicc_installer}"
else
  echo "No AICC installer found; continuing with the existing environment."
fi

cd "${repo_root}" || exit 1

actual_torch_version="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
if [[ "${actual_torch_version}" != "${TORCH_VERSION}" ]]; then
  echo "Expected torch==${TORCH_VERSION} in the CI image, found ${actual_torch_version}." >&2
  exit 1
fi

# The SGLang CI image already contains Torch, Triton, BoltOps and every
# requirements.txt dependency.  Do not replace its compatible runtime.
python -c 'import boltops, einops, ninja, numpy, packaging, pandas, psutil, pybind11, pytest, tabulate, torch, triton, yaml, zmq'

# Match the build-tool version required by the source release.
python -m pip install --index-url https://pypi.org/simple setuptools==79.0.1

# These build tools are absent from the SGLang image but are published on
# public PyPI. Do not reinstall them when a future CI image already provides
# them.
if ! command -v auditwheel >/dev/null || ! command -v patchelf >/dev/null; then
  python -m pip install --index-url https://pypi.org/simple auditwheel patchelf
fi

optest_pip_args=()
if [[ -n "${OPTEST_PIP_INDEX_URL:-}" ]]; then
  optest_pip_args+=(--index-url "${OPTEST_PIP_INDEX_URL}")
fi
if [[ -n "${OPTEST_PIP_TRUSTED_HOST:-}" ]]; then
  optest_pip_args+=(--trusted-host "${OPTEST_PIP_TRUSTED_HOST}")
fi
# optest is not currently published on public PyPI.  A future SGLang CI image
# should preinstall it; until then, configure its organisation package source.
if ! command -v optest >/dev/null; then
  if [[ "${#optest_pip_args[@]}" -eq 0 ]]; then
    echo "optest is absent from the CI image. Preinstall optest in the image or set OPTEST_PIP_INDEX_URL." >&2
    exit 1
  fi
  python -m pip install "${optest_pip_args[@]}" optest
fi

hipcc --version
python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip)"
