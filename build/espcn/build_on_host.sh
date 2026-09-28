#!/usr/bin/env bash
# Jude: Created
# Host-side launcher for the ESPCN FINN build. Runs unattended.
#
#   ./build_on_host.sh            # logs to build_on_host.log
#
# Two non-obvious requirements, both handled here:
#  1. build/espcn lives OUTSIDE $FINN_ROOT (the FINN fork at AMD/finn), so
#     run-docker.sh does not mount it -- added via FINN_DOCKER_EXTRA.
#  2. run-docker.sh:294 interpolates $DOCKER_CMD *unquoted*, so a multi-word command
#     is word-split: `bash -c 'cd X && python build.py'` reaches docker as
#     `bash -c cd X && python build.py` -- bash gets only `cd`, the build silently
#     no-ops and the container exits 0 looking like success. Hence the payload is a
#     single token: run_build_in_container.sh.
# `script -qec` supplies the pty that run-docker.sh's hardcoded -t needs.
set -uo pipefail

# The FINN fork is a peer of finn-examples, not upstream's build/finn clone
# (get-finn.sh) -- do not run get-finn.sh. Override FINN_DIR to use another checkout.
FINN_DIR="${FINN_DIR:-/mnt/Crucial/WorkspaceB/AMD/finn}"
ESPCN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${ESPCN_DIR}/build_on_host.log"

cd "$FINN_DIR" || exit 1
# shellcheck disable=SC1091
source ./env.sh
export FINN_DOCKER_EXTRA="${FINN_DOCKER_EXTRA} -v ${ESPCN_DIR}:${ESPCN_DIR} "

{
  echo "=== launching $(date -Is) ==="
  script -qec "./run-docker.sh ${ESPCN_DIR}/run_build_in_container.sh" /dev/null
  echo "=== launcher exit $? at $(date -Is) ==="
} 2>&1 | tee "$LOG"
