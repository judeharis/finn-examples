import brevitas_examples.super_resolution.models as models
import brevitas_examples.super_resolution.utils as utils
import os
import torch
import numpy as np
from brevitas.export import export_qonnx

# Jude: Edited
# Run inside the FINN container (torch 2.7.0, brevitas 0.11.0 -- what exported the current
# qonnx_model.onnx). On the host (torch 2.10, brevitas 0.12.1) the dataloader yields a
# slightly different image (22,932 of 49,152 input values differ) and export_qonnx fails.
#     ./run-docker.sh python /mnt/Crucial/WorkspaceB/AMD/finn-examples/build/espcn/models/get_model.py
# (with build/espcn bind-mounted, as build_on_host.sh does). The paths below are relative to
# this file's directory, so it runs from any working directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# get_model_by_name returns the model in TRAIN mode, where every forward updates the BatchNorm
# running mean/var. Before 2026-09-11 this script ran a train-mode forward on 0..255 data and
# then exported, so the deployed qonnx_model.onnx carried bn1-bn3 statistics dragged by that
# pass (weights and quantizer scales were unaffected) and lost 5.4 dB PSNR (models/eval_psnr.py).
# eval() first keeps the pretrained model exactly as trained.
model = models.get_model_by_name('quant_espcn_x2_w4a4_base', True).eval()
# Jude: Done
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
_, testloader = utils.get_bsd300_dataloaders(
        "../data",
        num_workers=0,
        batch_size=1,
        upscale_factor=model.upscale_factor,
        crop_size=256,
        download=True)

os.makedirs("../quant_espcn_x2_w4a4_base", exist_ok=True)

# Jude: Edited
x = testloader.dataset[0][0].unsqueeze(0).to(device)  # NCHW, 0..1 -- what the model expects
# input.npy is 0..255: the FINN graph prepends ToTensor() (/255) in custom_step_add_pre_proc.
inp = torch.round(x * 255)
model = model.to(device)
with open(f"../quant_espcn_x2_w4a4_base/input.npy", "wb") as f:
        np.save(f, inp.cpu().numpy())
# The golden must be model(inp / 255), not model(inp): feeding 0..255 to a model trained
# on 0..1 saturates its input quantizer and gives an output uncorrelated with the right
# one (pearson r = 0.15), which made every FINN verify_step FAIL on a correct design.
# inp / 255 rather than x, so the model sees exactly the rounded values FINN sees.
with torch.no_grad():
        golden = model(inp / 255)
with open(f"../quant_espcn_x2_w4a4_base/output.npy", "wb") as f:
        np.save(f, golden.cpu().numpy())
# Jude: Done
print(f"Saved I/O to ../quant_espcn_x2_w4a4_base as numpy arrays")

export_qonnx(
    model.cpu(),
    input_t=inp.cpu(),
    export_path=f"../quant_espcn_x2_w4a4_base/qonnx_model.onnx",
    opset_version=13)
print(f"Saved QONNX model to ../quant_espcn_x2_w4a4_base/qonnx_model.onnx")

# Jude: Edited
# FINN's verify_steps compare at a hard-coded atol=1e-3, below one output level (1/255), and
# Brevitas (float) and FINN (integer) disagree by 1-3 levels on ~135 of 196,608 values at
# quantization boundaries -- so against output.npy every step reports FAIL on a correct build.
# output_qonnx.npy is the reference build.py verifies against instead: qonnx's own executor
# run on the exported model, independent of the FINN build. The input is divided by 255 in
# float32 exactly as the ToTensor Div node custom_step_add_pre_proc prepends.
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes

qmodel = ModelWrapper("../quant_espcn_x2_w4a4_base/qonnx_model.onnx").transform(InferShapes())
qin = inp.cpu().numpy().astype(np.float32) / np.float32(255)
qout = execute_onnx(qmodel, {qmodel.graph.input[0].name: qin})[qmodel.graph.output[0].name]
with open("../quant_espcn_x2_w4a4_base/output_qonnx.npy", "wb") as f:
        np.save(f, qout)
lv = np.round(np.abs(qout - golden.cpu().numpy()) * 255)
print(f"Saved ../quant_espcn_x2_w4a4_base/output_qonnx.npy; vs Brevitas output.npy: "
      f"{int((lv == 0).sum())} equal, {int((lv > 0).sum())} differ, max {int(lv.max())} levels")
# Jude: Done
