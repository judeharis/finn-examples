#!/usr/bin/env bash
# Jude: Created
# Ship a standalone Deconvolution_hls build to the KV260 and run board/run_deconv.py.
#
#   ./deploy_to_board.sh <output_dir> [ssh-host] [frames]    defaults: kriaB, 4
#   FCLK=100 ./deploy_to_board.sh ...                         run below the synth clock
#
# e.g. ./deploy_to_board.sh output_deconv_pe1_simd1_5ns
#
# Lands in ~/deconv-finn-<tag> on the board and touches nothing else there; loading
# the bitstream does replace whatever overlay (e.g. ESPCN) is currently in the PL.
set -euo pipefail

OUT_ARG="${1:?usage: $0 <output_dir> [ssh-host] [frames]}"
HOST="${2:-kriaB}"
FRAMES="${3:-4}"
DECONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$DECONV_DIR" && cd "$OUT_ARG" && pwd)"
TAG="$(basename "$OUT")"; TAG="${TAG#output_deconv_}"
DEPLOY="$OUT/deploy"
REMOTE="deconv-finn-$TAG"
STAGE="$OUT/board_stage"

[ -f "$DEPLOY/bitfile/finn-accel.bit" ] || { echo "no bitfile in $DEPLOY -- build not finished?"; exit 1; }
ls "$OUT"/verification_output/verify_folded_hls_cppsim_0_SUCCESS.npy >/dev/null 2>&1 \
  || echo "WARNING: no cppsim SUCCESS in $OUT/verification_output -- board result has no cppsim backing"

# --- test frames ----------------------------------------------------------------
# Frame 0 is the build's own input/golden (ONNX ConvTranspose, already checked
# against PyTorch in build.py). Frames 1..N-1 are fresh seeded inputs. Every frame's
# golden must agree across PyTorch nn.ConvTranspose2d and a hand-written numpy
# ConvTranspose (reference.py), and frame 0 must also equal the ONNX golden.
mkdir -p "$STAGE"
python3 - "$OUT" "$DECONV_DIR" "$FRAMES" "$STAGE" <<'PY'
import json, sys
import numpy as np
out, here, n, stage = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, here)
from reference import numpy_convtranspose, torch_convtranspose
cfg = json.load(open(f"{out}/deconv_config.json"))
S, P = cfg["S"], cfg["P"]
W = np.load(f"{here}/data/espcn_convtranspose_W.npy")     # [CI, CO, K, K]

x0 = np.load(f"{out}/input.npy")                          # (1, H, W, CI)
rng = np.random.default_rng(1)
xs = np.concatenate([x0] + [rng.integers(0, 16, size=x0.shape) for _ in range(n - 1)])  # UINT4
g_torch = torch_convtranspose(xs, W, S, P)
g_numpy = numpy_convtranspose(xs, W, S, P)
for i in range(n):
    assert np.array_equal(g_torch[i], g_numpy[i]), f"frame {i}: PyTorch and numpy goldens disagree"
assert np.array_equal(g_torch[0], np.load(f"{out}/expected_output.npy")[0]), \
    "frame 0: PyTorch golden disagrees with the build's ONNX golden"
np.save(f"{stage}/frames_in.npy", xs.astype(np.float32))
np.save(f"{stage}/frames_gold.npy", g_torch.astype(np.float32))
print(f"prepared {n} frames: PyTorch == numpy on all, == ONNX golden on frame 0")
PY
cp "$OUT/deconv_config.json" "$DECONV_DIR/board/run_deconv.py" "$DECONV_DIR/board/fast_packing.py" "$STAGE/"

# --- copy -----------------------------------------------------------------------
echo "shipping to $HOST:~/$REMOTE ..."
ssh "$HOST" "rm -rf ~/$REMOTE && mkdir -p ~/$REMOTE"
scp -q -r "$DEPLOY"/* "$HOST:~/$REMOTE/"
scp -q "$STAGE"/* "$HOST:~/$REMOTE/driver/"
# Same guarded-import fix as the ESPCN deploy: without it the generated driver dies
# on `import qonnx.core.modelwrapper`, which the shipped qonnx subset lacks.
FINN_DIR="${FINN_DIR:-/mnt/Crucial/WorkspaceB/AMD/finn}"
scp -q "$FINN_DIR/src/finn/util/data_packing.py" \
       "$HOST:~/$REMOTE/driver/finn/util/data_packing.py"

# Newer qonnx (the merged FINN image) has a module-level `from onnx import GraphProto,
# ModelProto` in qonnx/util/basic.py, used only as annotations on qonnx_make_model.
# The board's PYNQ venv has no onnx, so the driver dies on import. Guard it in the
# board-side copy rather than installing onnx into the shared venv.
ssh "$HOST" "python3 - ~/$REMOTE/driver/qonnx/util/basic.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "from onnx import GraphProto, ModelProto\n"
new = ("try:\n    from onnx import GraphProto, ModelProto\n"
       "except ImportError:  # board venv has no onnx; annotations only\n"
       "    GraphProto = ModelProto = object\n")
if old in s:
    open(p, "w").write(s.replace(old, new, 1))
    print("guarded onnx import in driver/qonnx/util/basic.py")
PY

# --- run ------------------------------------------------------------------------
# `source activate` (not the venv python alone) puts the working xclbinutil first on
# PATH; XILINX_XRT/BOARD are normally set by a profile script ssh never sources.
FCLK_ARG=""; [ -n "${FCLK:-}" ] && FCLK_ARG="--fclk $FCLK"
PYNQ_ENV="source /usr/local/share/pynq-venv/bin/activate; export XILINX_XRT=/usr BOARD=KV260"

# Put the board back on its baseline bitstream after every run, pass or fail --
# requested by the board's owner. The trap covers Ctrl-C and ssh failures too.
RESET_BIT=/home/ubuntu/bitstreams/CPU_1_0.bit
reset_board() {
  echo "resetting $HOST PL to $RESET_BIT ..."
  ssh "$HOST" "$PYNQ_ENV; python -u -c 'from pynq import Overlay; \
      o = Overlay(\"$RESET_BIT\"); print(\"reset ok:\", o.bitfile_name, o.is_loaded())'" \
    || echo "WARNING: board reset FAILED -- load $RESET_BIT by hand"
}
trap reset_board EXIT

echo "running on $HOST ..."
set +e
# -u: over a non-tty ssh, python block-buffers stdout and nothing shows until exit
ssh "$HOST" "$PYNQ_ENV; cd ~/$REMOTE/driver && python -u run_deconv.py $FCLK_ARG"
RC=$?
set -e
scp -q "$HOST:~/$REMOTE/driver/board_results.json" "$OUT/board_results.json" 2>/dev/null \
  && echo "results -> $OUT/board_results.json"
exit $RC
