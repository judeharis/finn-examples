# Copyright (C) 2023, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from custom_steps import *

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import *

model_name = "espcn-bsd300"
# Jude: Edited
# FUSED_DECONV comes from custom_steps (default on; env ESPCN_FUSED_DECONV=0 for pixel
# padding, which builds into output_..._pixelpad). folding_config_chrc_cap.json names nodes of an older FINN
# (MatrixVectorActivation_0, ...) that no longer exist, so it is a no-op and SetFolding
# does all folding. The fused build pins every non-deconv layer (Thresholding, FMPadding, SWG,
# MVAU) to the pixel-padding build's folding, so the deconv is the only difference; without
# the pins SetFolding's two-pass relaxation re-folds them to the deconv's pace. The deconv
# itself is left to SetFolding (PE3/SIMD4 for ESPCN).
variant = "" if FUSED_DECONV else "_pixelpad"
folding_config = (
    "folding_config_fused_deconv.json" if FUSED_DECONV else "folding_config_chrc_cap.json"
)
# Jude: Done


espcn_build_steps = [
    custom_step_qonnx_tidy_up,
    custom_step_add_pre_proc,
    "step_qonnx_to_finn",
    "step_tidy_up",
    custom_step_streamline,
    custom_step_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    # Jude: Edited, Removed
    # Keep commented: step_transpose_decomposition does not exist in this FINN
    # (v0.10.1-742-gee7d7115). String steps are resolved by a bare dict lookup,
    # build_dataflow_step_lookup[name] in build_dataflow.py:73, so enabling it
    # raises KeyError before the build does any work. It appears to come from a
    # newer FINN branch. Not needed here: it would decompose the Transpose nodes
    # InferPixelPaddingDeconv leaves behind, and custom_step_streamline already
    # absorbs those.
    # "step_transpose_decomposition",
    # Jude: Done
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    # Jude: Edited, Removed
    # step_measure_rtlsim_performance dropped: critical_path_cycles is 104,263,595 and
    # rtlsim_batch_size is 100, i.e. ~1e10 cycles of XSI simulation. This is what the
    # 2026-03-12 run stalled inside (log ends mid-step, no traceback).
    # step_out_of_context_synthesis dropped: gated on DataflowOutputType.OOC_SYNTH, which
    # is not in generate_outputs, so it was a no-op anyway.
    # Jude: Done
    "step_synthesize_bitfile",
    "step_make_driver",
    "step_deployment_package",
]

model_file = "quant_espcn_x2_w4a4_base/qonnx_model.onnx"

cfg = build_cfg.DataflowBuildConfig(
    steps=espcn_build_steps,
    # Jude: Edited
    output_dir="output_%s_kriasom%s" % (model_name, variant),
    # Jude: Done
    synth_clk_period_ns=5.0,
    target_fps=10000,
    fpga_part="xck26-sfvc784-2LV-c",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    board="KV260_SOM",
    enable_build_pdb_debug=False,
    verbose=False,
    split_large_fifos=True,
    # Jude: Edited
    folding_config_file=folding_config,
    # Jude: Done
    auto_fifo_depths=False,
    rtlsim_batch_size=100,
    verify_input_npy="quant_espcn_x2_w4a4_base/input.npy",
    # Jude: Edited
    # qonnx execution of the exported model, not the Brevitas output.npy: the two differ by
    # 1-3 levels on ~135 values at quantization boundaries, and verify_step's hard-coded
    # atol=1e-3 is below one level (1/255), so output.npy fails every step of a correct build.
    # Written by models/get_model.py.
    verify_expected_output_npy="quant_espcn_x2_w4a4_base/output_qonnx.npy",
    # Jude: Done
    verify_steps=[
        build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
        build_cfg.VerificationStepType.TIDY_UP_PYTHON,
        build_cfg.VerificationStepType.STREAMLINED_PYTHON,
        build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
    ],
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        # Jude: Edited, Removed
        build_cfg.DataflowOutputType.BITFILE,
        # Without these two, step_make_driver and step_deployment_package are both
        # wholly wrapped in an `if <type> in cfg.generate_outputs` and silently emit
        # nothing -- no driver/ and no deploy/, so nothing to put on the board.
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        # Jude: Done
    ],
)

build.build_dataflow_cfg(model_file, cfg)
