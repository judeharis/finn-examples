#!/usr/bin/env bash
# Jude: Created
# Ship the built ESPCN accelerator (bitstream + FINN PYNQ driver) to the KV260 and
# run it. Assumes build.py has already produced output_.../deploy/.
#
#   ./deploy_to_board.sh [ssh-host]        default host: kriaB
#
# What goes over: deploy/bitfile/{finn-accel.bit,.hwh} and the FINN-generated PYNQ
# driver (driver.py, driver_base.py, finn/, qonnx/, runtime_weights/), plus our
# board app run_espcn.py and the input/golden vectors.
set -euo pipefail

HOST="${1:-kriaB}"
ESPCN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default: the fused-deconv build. ESPCN_FUSED_DECONV=0 deploys the pixel-padding build
# instead (build_on_host.sh, same flag).
if [ "${ESPCN_FUSED_DECONV:-1}" = 1 ]; then
  OUT="$ESPCN_DIR/output_espcn-bsd300_kriasom"
else
  OUT="$ESPCN_DIR/output_espcn-bsd300_kriasom_pixelpad"
fi
DEPLOY="$OUT/deploy"
REMOTE=espcn-finn

[ -d "$DEPLOY" ] || { echo "no $DEPLOY -- run build.py first (needs DEPLOYMENT_PACKAGE in generate_outputs)"; exit 1; }
[ -f "$DEPLOY/bitfile/finn-accel.bit" ] || { echo "no bitfile in $DEPLOY"; exit 1; }

# --- prepare the I/O vectors in the layout the accelerator wants ---------------
# The accelerator takes UINT8 NHWC (1,128,128,3); input.npy is NCHW float32 already
# scaled to 0..255. The golden is the build's own folded-HLS-cppsim output, NOT the
# Brevitas quant_espcn_x2_w4a4_base/output.npy -- see the notes in run_espcn.py.
GOLD=$(ls -t "$OUT"/verification_output/verify_folded_hls_cppsim_0_*.npy 2>/dev/null | head -1)
[ -n "$GOLD" ] || { echo "no cppsim golden in $OUT/verification_output"; exit 1; }

python3 - "$ESPCN_DIR" "$GOLD" <<'PY'
import sys, numpy as np, os
espcn, gold = sys.argv[1], sys.argv[2]
os.makedirs(f"{espcn}/board", exist_ok=True)
x = np.load(f"{espcn}/quant_espcn_x2_w4a4_base/input.npy")          # (1,3,128,128) f32 0..255
np.save(f"{espcn}/board/input_nhwc_uint8.npy",
        np.round(x).astype(np.uint8).transpose(0, 2, 3, 1))
g = np.load(gold)                                                    # (1,3,256,256) f32 in [0,1]
s = g * 255.0
assert np.abs(s - np.round(s)).max() < 1e-3, "golden is not uint8/255 as expected"
np.save(f"{espcn}/board/golden_nhwc_uint8.npy",
        np.round(s).astype(np.uint8).transpose(0, 2, 3, 1))
print("prepared board/input_nhwc_uint8.npy and board/golden_nhwc_uint8.npy")
PY

# --- copy ---------------------------------------------------------------------
echo "shipping to $HOST:~/$REMOTE ..."
ssh "$HOST" "rm -rf ~/$REMOTE && mkdir -p ~/$REMOTE"
scp -q -r "$DEPLOY"/* "$HOST:~/$REMOTE/"
scp -q "$ESPCN_DIR/board/run_espcn.py" \
       "$ESPCN_DIR/board/input_nhwc_uint8.npy" \
       "$ESPCN_DIR/board/golden_nhwc_uint8.npy" "$HOST:~/$REMOTE/driver/"

# FINN dev's MakePYNQDriver ships a data_packing.py matching its own driver_base and
# importing nothing the board lacks, so the feature/deconv-era copy of FINN's file is
# no longer needed here.

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

# --- run ----------------------------------------------------------------------
# `source activate` matters beyond picking the interpreter: it puts
# /usr/local/share/pynq-venv/bin first on PATH, which is where the *working*
# xclbinutil lives. The system /usr/bin/unwrapped/xclbinutil segfaults and its
# wrapper exits 0 regardless, so PYNQ fails later with a confusing
# FileNotFoundError on t.xclbin.
# XILINX_XRT/BOARD come from /etc/profile.d/pynq_venv.sh, which a non-interactive
# ssh never sources.
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
# -u: over a non-tty ssh, python block-buffers stdout and nothing shows until exit
ssh "$HOST" "$PYNQ_ENV; cd ~/$REMOTE/driver && python -u run_espcn.py --runs 5"
