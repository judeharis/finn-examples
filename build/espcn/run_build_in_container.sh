#!/usr/bin/env bash
# Jude: Created
# Runs inside the FINN container. Invoked as the SOLE argument to run-docker.sh:
#
#     ./run-docker.sh /mnt/.../build/espcn/run_build_in_container.sh
#
# It has to be a script rather than `run-docker.sh bash -c '...'` because
# run-docker.sh:294 builds CMD_TO_RUN by interpolating $DOCKER_CMD *unquoted*, so
# any multi-word command is word-split: `bash -c 'cd X && python build.py'` reaches
# docker as `bash -c cd X && python build.py`, i.e. bash gets only `cd`, the build
# never runs, and the container exits 0 looking like a success.
set -euo pipefail

ESPCN_DIR=/mnt/Crucial/WorkspaceB/AMD/finn-examples/build/espcn

echo "=== in-container build starting $(date -Is) ==="
echo "FINN_ROOT=${FINN_ROOT:-unset}  FINN_BUILD_DIR=${FINN_BUILD_DIR:-unset}  HOME=${HOME:-unset}"

# The failure mode this guards against is silent: if HOME is not writable the
# entrypoint's `pip install --user -e` calls all fail and finn is simply absent.
python -c 'import finn, qonnx, brevitas; print("imports OK:", finn.__file__)'

cd "$ESPCN_DIR"
python build.py
echo "=== in-container build finished $(date -Is) ==="
