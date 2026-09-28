# Jude: Created
# BSD300 test-set PSNR of the clean pretrained quant_espcn_x2_w4a4_base and of the model
# actually deployed on the KV260. Run inside the FINN container (see get_model.py):
#
#     cd finn && source env.sh
#     ./run-docker.sh python /mnt/Crucial/WorkspaceB/AMD/finn-examples/build/espcn/models/eval_psnr.py
#
# The deployed qonnx_model.onnx was exported by the pre-2026-09-11 get_model.py, which left the
# model in TRAIN mode, ran model(round(x*255)) on test image 0 and then exported -- so the
# BatchNorm running mean/var were updated from that 0..255 input. This script rebuilds that
# model by replaying the same steps, and proves the replay by exporting it and comparing every
# initializer with the deployed ONNX, and its output with output.npy.
import os
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from brevitas.export import export_qonnx
import brevitas_examples.super_resolution.models as models
import brevitas_examples.super_resolution.utils as utils

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DEPLOYED = os.path.join(HERE, "..", "quant_espcn_x2_w4a4_base")
NAME = "quant_espcn_x2_w4a4_base"

torch.manual_seed(123456)


def loader(crop_size):
    _, testloader = utils.get_bsd300_dataloaders(
        DATA, num_workers=0, batch_size=1, upscale_factor=2, crop_size=crop_size, download=True)
    return testloader


def clean_model():
    return models.get_model_by_name(NAME, True).eval()


def deployed_model(export_path):
    # Replays the old get_model.py exactly, including exporting while still in train mode.
    model = models.get_model_by_name(NAME, True)
    x = loader(256).dataset[0][0].unsqueeze(0)
    inp = torch.round(x * 255)
    model(inp)
    export_qonnx(model, input_t=inp, export_path=export_path, opset_version=13)
    return model.eval(), inp


def initializers(path):
    return {i.name: numpy_helper.to_array(i) for i in onnx.load(path).graph.initializer}


def check_replay(model, inp, export_path):
    ref = initializers(os.path.join(DEPLOYED, "qonnx_model.onnx"))
    got = initializers(export_path)
    assert ref.keys() == got.keys(), f"initializer names differ: {ref.keys() ^ got.keys()}"
    bad = [k for k in ref if not np.array_equal(ref[k], got[k])]
    assert not bad, f"{len(bad)} of {len(ref)} initializers differ, e.g. {bad[:5]}"
    print(f"replay check: all {len(ref)} initializers equal the deployed qonnx_model.onnx")
    with torch.no_grad():
        out = model(inp / 255).numpy()
    golden = np.load(os.path.join(DEPLOYED, "output.npy"))
    assert np.array_equal(out, golden), f"max |diff| vs output.npy {np.abs(out - golden).max()}"
    print("replay check: output on image 0 equals output.npy")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        export_path = os.path.join(tmp, "replay.onnx")
        deployed, inp = deployed_model(export_path)
        check_replay(deployed, inp, export_path)
    clean = clean_model()
    print(f"\n{'crop':>5}  {'clean':>8}  {'deployed':>8}  {'delta':>7}   (PSNR dB, BSD300 test, "
          f"{len(loader(256).dataset)} images)")
    with torch.no_grad():
        for crop in (256, 512):
            testloader = loader(crop)
            c = utils.evaluate_avg_psnr(testloader, clean)
            d = utils.evaluate_avg_psnr(testloader, deployed)
            print(f"{crop:>5}  {c:8.2f}  {d:8.2f}  {d - c:+7.2f}")


if __name__ == "__main__":
    main()
