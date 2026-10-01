# Jude: Created
"""Standalone board build of the fused Deconvolution_hls layer, at the ESPCN geometry.

Tests the fused deconv kernel on the KV260 without touching the ESPCN build
(build/espcn/build.py, custom_steps.py) or any FINN transformation. There is no
ConvTranspose -> Deconvolution_hls transform in FINN, so instead of converting a
network we hand-build a graph that already holds the HW node -- exactly as
tests/fpgadataflow/test_fpgadataflow_deconv.py:create_deconv_node does -- and
enter the stock builder at step_create_dataflow_partition. Every step from there
on is FINN's own, unmodified; FINN's test_fpgadataflow_ipstitch_zynqbuild_end2end
does the same thing for a hand-built MVAU graph.

The layer is ESPCN's ConvTranspose_0 (K=6, S=2, P=2, 128x128x32 -> 256x256x3)
with its trained INT8 weights, taken from the streamlined ESPCN model (see
data/espcn_convtranspose_W.npy). Those are exactly the weights MVAU_hls_3
multiplies in the deployed design, spatially flipped -- checked on 2026-09-11.

Output is the raw INT32 accumulator, not ESPCN's thresholded UINT8: the fused
kernel does not fuse an activation, and checking the full accumulator is the
stronger bit-exactness test.

The golden (expected_output.npy) is an ONNX ConvTranspose run by onnxruntime, and
must equal PyTorch nn.ConvTranspose2d (reference.py) before anything is built.
Neither comes from FINN, so a board match means the hardware is right, not merely
self-consistent.

Configure through the environment (build_on_host.sh forwards these):
    DECONV_PE      output-channel parallelism, must divide 3   (default 1)
    DECONV_SIMD    input-channel parallelism, must divide 32   (default 1)
    DECONV_CLK_NS  synthesis clock period in ns                (default 5.0)
"""
import json
import os

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe
from reference import torch_convtranspose

HERE = os.path.dirname(os.path.abspath(__file__))

PE = int(os.environ.get("DECONV_PE", 1))
SIMD = int(os.environ.get("DECONV_SIMD", 1))
CLK_NS = float(os.environ.get("DECONV_CLK_NS", 5.0))

# ESPCN ConvTranspose_0, as it appears in custom_step_streamline.onnx
K, S, P = 6, 2, 2
H = W = 128
CI, CO = 32, 3
IDT = DataType["UINT4"]  # MultiThreshold_3 output feeding the deconv
WDT = DataType["INT8"]
ODT = DataType["INT32"]  # raw accumulator; worst case |acc| < 2^19 for these weights
HO = (H - 1) * S - 2 * P + K
WO = (W - 1) * S - 2 * P + K

SEED = 0


def fits(dt, a):
    """DataType.allowed() is scalar-only (it calls float(value)), so check arrays here."""
    return bool(np.all(a == np.round(a)) and a.min() >= dt.min() and a.max() <= dt.max())


def make_reference_model(w_deconv):
    """Plain ONNX ConvTranspose in NCHW -- the golden, as in the FINN deconv test."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, CI, H, W])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, CO, HO, WO])
    node = helper.make_node(
        "ConvTranspose",
        ["inp", "W"],
        ["outp"],
        dilations=(1, 1),
        group=1,
        kernel_shape=(K, K),
        pads=(P, P, P, P),
        strides=(S, S),
    )
    graph = helper.make_graph([node], "ref_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="deconv-ref"))
    model.set_initializer("W", w_deconv)
    return model.transform(InferShapes())


def make_hw_model(w_deconv):
    """One Deconvolution_hls node, NHWC, weights as [CO, K, K, CI]."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, H, W, CI])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, HO, WO, CO])
    node = helper.make_node(
        "Deconvolution_hls",
        ["inp", "W"],
        ["outp"],
        # PrepareIP names the HLS top function and its project dirs after the node
        name="Deconvolution_hls_0",
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        KernelDim=[K, K],
        IFMChannels=CI,
        OFMChannels=CO,
        IFMDim=[H, W],
        Stride=[S, S],
        Padding=[P, P],
        PE=PE,
        SIMD=SIMD,
        inputDataType=IDT.name,
        weightDataType=WDT.name,
        outputDataType=ODT.name,
        cpp_interface="hls_vector",
        hls_style="freerunning",
    )
    graph = helper.make_graph([node], "deconv_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="deconv-board"))
    model.set_tensor_datatype("inp", IDT)
    model.set_tensor_datatype("outp", ODT)
    model.set_tensor_datatype("W", WDT)
    model.set_initializer("W", w_deconv.transpose(1, 2, 3, 0))
    return model.transform(InferShapes())


def main():
    assert CO % PE == 0, f"PE={PE} must divide CO={CO}"
    assert CI % SIMD == 0, f"SIMD={SIMD} must divide CI={CI}"
    tag = "pe%d_simd%d_%gns" % (PE, SIMD, CLK_NS)
    out_dir = os.path.join(HERE, "output_deconv_" + tag)
    os.makedirs(out_dir, exist_ok=True)

    w_deconv = np.load(os.path.join(HERE, "data/espcn_convtranspose_W.npy"))
    assert w_deconv.shape == (CI, CO, K, K), w_deconv.shape
    assert fits(WDT, w_deconv), "weights do not fit %s" % WDT.name

    # Seeded full-range UINT4 input: exercises every activation value, which a
    # real ESPCN feature map would not.
    np.random.seed(SEED)
    x_nchw = gen_finn_dt_tensor(IDT, [1, CI, H, W])
    y_nchw = oxe.execute_onnx(make_reference_model(w_deconv), {"inp": x_nchw})["outp"]
    x_nhwc = np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1))
    y_nhwc = np.ascontiguousarray(y_nchw.transpose(0, 2, 3, 1))
    assert fits(ODT, y_nhwc), "reference output does not fit %s" % ODT.name
    # Second, independent golden: PyTorch nn.ConvTranspose2d, as DeConv_hls_benchmark uses.
    # Stop here rather than build hardware against a reference two frameworks disagree on.
    y_torch = torch_convtranspose(x_nhwc, w_deconv, S, P)
    assert np.array_equal(y_torch, y_nhwc), "PyTorch and ONNX goldens disagree"
    print("golden: ONNX ConvTranspose == PyTorch nn.ConvTranspose2d (%d values)" % y_nhwc.size)
    in_npy = os.path.join(out_dir, "input.npy")
    gold_npy = os.path.join(out_dir, "expected_output.npy")
    np.save(in_npy, x_nhwc.astype(np.float32))
    np.save(gold_npy, y_nhwc.astype(np.float32))

    model = make_hw_model(w_deconv)
    model_file = os.path.join(out_dir, "deconv_hw.onnx")
    model.save(model_file)

    exp_cycles = getCustomOp(model.graph.node[0]).get_exp_cycles()
    with open(os.path.join(out_dir, "deconv_config.json"), "w") as f:
        json.dump(
            dict(
                K=K, S=S, P=P, H=H, W=W, CI=CI, CO=CO, PE=PE, SIMD=SIMD,
                idt=IDT.name, wdt=WDT.name, odt=ODT.name,
                synth_clk_period_ns=CLK_NS, seed=SEED,
                get_exp_cycles=int(exp_cycles),
                reference_output_range=[float(y_nhwc.min()), float(y_nhwc.max())],
            ),
            f,
            indent=2,
        )
    print("get_exp_cycles() = %d  (%.1f ms at %g ns)" % (exp_cycles, exp_cycles * CLK_NS * 1e-6, CLK_NS))

    # The graph is already all-HW and already specialized, so everything before
    # step_create_dataflow_partition (tidy-up, streamlining, convert_to_hw,
    # specialize_layers) has nothing to do and is left out. PE/SIMD are already set on the
    # node. On FINN dev the FOLDED_HLS_CPPSIM verification runs in step_minimize_bit_width
    # (it used to run in step_apply_folding_config), so that step is in the list: for a lone
    # Deconvolution it changes nothing else (no minimize_* methods, no thresholds).
    # FINN dev refuses a build with neither target_fps nor folding_config_file
    # (config check "folding_missing"), so the same PE/SIMD also go into a config file.
    fold_json = os.path.join(out_dir, "folding_config.json")
    with open(fold_json, "w") as f:
        json.dump({"Defaults": {}, "Deconvolution_hls_0": {"PE": PE, "SIMD": SIMD}}, f, indent=2)
    steps = [
        "step_create_dataflow_partition",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_synthesize_bitfile",
        "step_make_driver",
        "step_deployment_package",
    ]

    cfg = build_cfg.DataflowBuildConfig(
        steps=steps,
        output_dir=out_dir,
        synth_clk_period_ns=CLK_NS,
        fpga_part="xck26-sfvc784-2LV-c",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        board="KV260_SOM",
        # Default is True with the largefifo_rtlsim strategy, which would simulate
        # the whole ~5e7-cycle layer in XSI. A single node between two IODMAs needs
        # no FIFO sizing: the chain is linear, so FIFO depth cannot deadlock it.
        auto_fifo_depths=False,
        folding_config_file=fold_json,
        enable_build_pdb_debug=False,
        verbose=False,
        verify_input_npy=in_npy,
        verify_expected_output_npy=gold_npy,
        verify_steps=[build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM],
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.BITFILE,
            # step_make_driver and step_deployment_package are no-ops without these
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    build.build_dataflow_cfg(model_file, cfg)


if __name__ == "__main__":
    main()
