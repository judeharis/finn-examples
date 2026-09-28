#!/usr/bin/env bash
# Jude: Created
# Host-side launcher for the standalone Deconvolution_hls board build. Runs unattended.
#
#   ./build_on_host.sh [PE] [SIMD] [CLK_NS]      defaults: 1 1 5.0
#
# Writes output_deconv_pe<PE>_simd<SIMD>_<CLK>ns/ and logs to build_on_host_<same>.log.
# Same two workarounds as build/espcn/build_on_host.sh: this dir is outside
# $FINN_ROOT so it needs its own bind mount, and the container payload must be a
# single token because run-docker.sh word-splits $DOCKER_CMD. The config therefore
# travels as `docker -e` in FINN_DOCKER_EXTRA rather than as arguments.
set -uo pipefail

PE="${1:-1}"
SIMD="${2:-1}"
CLK_NS="${3:-5.0}"

# The FINN fork is a peer of finn-examples (not upstream's build/finn clone).
FINN_DIR="${FINN_DIR:-/mnt/Crucial/WorkspaceB/AMD/finn}"
DECONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${DECONV_DIR}/build_on_host_pe${PE}_simd${SIMD}_${CLK_NS}ns.log"

# These normally come from ~/.bashrc, which a non-interactive shell (cron, ssh,
# an agent) never sources. Without them run-docker.sh refuses to start, and
# env.sh trips `set -u` on FINN_DOCKER_EXTRA before anything reaches the log.
: "${FINN_XILINX_PATH:=/mnt/Crucial/Xilinx2024/}"
: "${FINN_XILINX_VERSION:=2024.1}"
: "${NUM_DEFAULT_WORKERS:=16}"
: "${FINN_DOCKER_EXTRA:=}"
export FINN_XILINX_PATH FINN_XILINX_VERSION NUM_DEFAULT_WORKERS FINN_DOCKER_EXTRA

cd "$FINN_DIR" || exit 1
# shellcheck disable=SC1091
source ./env.sh
export FINN_DOCKER_EXTRA="${FINN_DOCKER_EXTRA} -v ${DECONV_DIR}:${DECONV_DIR} \
-e DECONV_PE=${PE} -e DECONV_SIMD=${SIMD} -e DECONV_CLK_NS=${CLK_NS} "

{
  echo "=== launching $(date -Is)  PE=${PE} SIMD=${SIMD} CLK_NS=${CLK_NS} ==="
  script -qec "./run-docker.sh ${DECONV_DIR}/run_build_in_container.sh" /dev/null
  echo "=== launcher exit $? at $(date -Is) ==="
} 2>&1 | tee "$LOG"
