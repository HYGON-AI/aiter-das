#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Start an interactive BW1100-capable SGLang CI container for reproducing the
# GitHub Actions job locally.  The source tree is mounted at /workspace.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${AITER_CI_IMAGE:-10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.18-latest}"
container_name="${AITER_CI_CONTAINER_NAME:-aiter-sglang-ci}"

exec docker run -dit \
  --shm-size 100g \
  --network host \
  --name "${container_name}" \
  --privileged \
  --device /dev/kfd \
  --device /dev/mkfd \
  --device /dev/dri \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -u root \
  -v "${repo_root}":/workspace \
  -v /opt/hyhal:/opt/hyhal:ro \
  "${image}" \
  /bin/bash
