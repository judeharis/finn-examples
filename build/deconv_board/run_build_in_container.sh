#!/usr/bin/env bash
# Jude: Created
# Runs inside the FINN container. Invoked as the SOLE argument to run-docker.sh --
# see build/espcn/run_build_in_container.sh for why it cannot be `bash -c '...'`.
# DECONV_PE / DECONV_SIMD / DECONV_CLK_NS arrive as `docker -e`, set by build_on_host.sh.
set -euo pipefail

# the directory this script lives in (build_on_host.sh mounts it at the same path)
DECONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== in-container build starting $(date -Is) ==="
echo "FINN_ROOT=${FINN_ROOT:-unset}  FINN_BUILD_DIR=${FINN_BUILD_DIR:-unset}  HOME=${HOME:-unset}"
echo "PE=${DECONV_PE:-unset}  SIMD=${DECONV_SIMD:-unset}  CLK_NS=${DECONV_CLK_NS:-unset}"

# Guards the silent failure where HOME is unwritable and finn never gets installed.
python -c 'import finn, qonnx, brevitas; print("imports OK:", finn.__file__)'

cd "$DECONV_DIR"
python build.py
echo "=== in-container build finished $(date -Is) ==="
