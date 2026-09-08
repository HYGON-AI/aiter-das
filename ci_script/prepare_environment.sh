#!/usr/bin/env bash

: "${DTK_PKG:?DTK_PKG must point to the DTK archive mounted on the Runner}"
: "${TORCH_VERSION:?TORCH_VERSION must be set}"

cp -f "${DTK_PKG}" /opt/
cd /opt
tar -xzf "$(basename "${DTK_PKG}")"
rm -rf dtk
mv dtk-* dtk
source /opt/dtk/env.sh

cd "${CI_PROJECT_DIR:?CI_PROJECT_DIR must be set by GitLab}"

# Install the newest AICC package available on the Runner's shared storage.
# Its absence is non-fatal because some Runners include AICC already.
#
# AICC filename format:
#   dtk_llvm_<commit>_<YYYYMMDD[HHMM]>.run
#
# Compare the timestamp extracted from the filename instead of sorting the full
# path, because the parent directory "/ArchivedFile/ai_cc" contains an
# underscore that would shift the sort field.
aicc_dir="/ArchivedFile/ai_cc/nightly"
latest_aicc_run=""
latest_aicc_timestamp=""

shopt -s nullglob
for aicc_candidate in "${aicc_dir}"/dtk_llvm_*.run; do
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
  cd /opt
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

cd "${CI_PROJECT_DIR:?CI_PROJECT_DIR must be set by GitLab}"

python -m pip install packaging -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install zmq -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install tabulate -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install wheel -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install -r requirements.txt --trusted-host 10.68.20.101
python -m pip install  ciupload auditwheel patchelf
python -m pip install torch=="${TORCH_VERSION}" triton 
python -m pip install torch=="${TORCH_VERSION}" boltops
PYTHON_VERSION="$(python --version 2>&1 | awk -F '[ .]' '{print $2 "." $3}')"
export PYTHON_VERSION

declare -A NUMPY_VERSIONS=(
  ["3.8"]="1.21.6"
  ["3.9"]="1.22.4"
  ["3.10"]="1.24.3"
  ["3.11"]="1.26.2"
  ["3.12"]="1.26.2"
  ["3.13"]="2.1.2"
  ["3.14"]="2.3.4"
)

NUMPY_VERSION="${NUMPY_VERSIONS[$PYTHON_VERSION]}"
if [[ -z "${NUMPY_VERSION}" ]]; then
  echo "Warning: no numpy version configured for Python ${PYTHON_VERSION}; using numpy==1.24.3"
  NUMPY_VERSION="1.24.3"
fi

echo "Installing numpy==${NUMPY_VERSION} ..."
python -m pip uninstall -y numpy 2>/dev/null || true
python -m pip install numpy=="${NUMPY_VERSION}" \
  --force-reinstall \
  --no-deps \
  --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/

python -c "import numpy; print(f'numpy {numpy.__version__} installed successfully')"

python -m pip install optest -i http://10.16.1.201:9929/nightly/dtk2604/+simple/ --trusted-host 10.16.1.201 --force-reinstall

hipcc --version
python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip)"
