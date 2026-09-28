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

# get_model_by_name returns the model in TRAIN mode, where Brevitas recalibrates activation
# quantizer scales from whatever data passes through. Before 2026-09-11 this script ran a
# train-mode forward on 0..255 data and then exported, so the exported model carries scales
# shifted by that pass. eval() first keeps the pretrained model exactly as trained.
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
