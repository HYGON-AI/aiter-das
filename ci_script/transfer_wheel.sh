#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

# Share the target commit before environment setup, then hand off the NMZ wheel.
package_url="${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/generic/aiter-ci-dtk${DTK_VERSION//./}-py${PY_VERSION}-torch${TORCH_VERSION}/${CI_PIPELINE_ID}/wheel.tar.gz"
resource_name="wheel"
if [[ "$1" == *-target ]]; then
  package_url="${package_url%/*}/ci-target-sha"
  resource_name="target commit"
fi
archive="$(mktemp)"
trap 'rm -f "${archive}"' EXIT

case "$1" in
  upload|upload-target)
    if [[ "$1" == "upload-target" ]]; then
      cp dist/ci-target-sha "${archive}"
    else
      tar -czf "${archive}" dist/*manylinux*.whl dist/ci-target-sha
    fi
    curl -fsSL --connect-timeout 30 --max-time 300 --retry 3 --retry-delay 5 --retry-connrefused \
      --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --upload-file "${archive}" "${package_url}"
    echo "AITER ${resource_name} published for pipeline ${CI_PIPELINE_ID}, torch ${TORCH_VERSION}"
    ;;
  download|download-target)
    if [[ "$1" == "download" ]]; then
      expected_target="$(cat dist/ci-target-sha)"
    fi
    # Match NMZ's 3-hour build/test limit; BW has another 3 hours for its own tests.
    wait_started=$SECONDS
    waiting_logged=0
    deadline=$((wait_started + 10800))
    while (( SECONDS < deadline )); do
      status="$(curl -sL --connect-timeout 30 --max-time 300 \
        --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --output "${archive}" \
        --write-out '%{http_code}' "${package_url}")" || status=000
      case "${status}" in
        200)
          elapsed=$((SECONDS - wait_started))
          if [[ "$1" == "download-target" ]]; then
            test -s "${archive}"
            mkdir -p dist
            cp "${archive}" dist/ci-target-sha
          else
            if [[ "$(tar -xOzf "${archive}" dist/ci-target-sha)" != "${expected_target}" ]]; then
              echo "NMZ wheel target commit differs from the prepared environment; rerun the pipeline" >&2
              exit 1
            fi
            tar -xzf "${archive}"
            test -f dist/*manylinux*.whl
            test -s dist/ci-target-sha
          fi
          echo "Received NMZ ${resource_name} for pipeline ${CI_PIPELINE_ID}, torch ${TORCH_VERSION} (waited $((elapsed / 60))m $((elapsed % 60))s)"
          exit 0
          ;;
        000|404|408|429|5??)
          if (( waiting_logged == 0 )); then
            echo "Waiting for NMZ ${resource_name} (torch ${TORCH_VERSION}, HTTP ${status})"
            waiting_logged=1
          fi
          sleep 10
          ;;
        *)
          echo "Cannot download NMZ ${resource_name}: HTTP ${status}; check Package Registry access" >&2
          exit 1
          ;;
      esac
    done
    echo "Timed out waiting for NMZ ${resource_name} after 3 hours (torch ${TORCH_VERSION})" >&2
    exit 1
    ;;
  *)
    echo "Usage: $0 upload|download|upload-target|download-target" >&2
    exit 2
    ;;
esac
