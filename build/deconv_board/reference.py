# Jude: Created
"""Independent ConvTranspose references for the standalone deconv goldens.

Two implementations that share no code with each other or with FINN:

  torch_convtranspose  -- nn.ConvTranspose2d, set up the way
                          DeConv_hls_benchmark/scripts/deconv_benchmark.py:init_layer does
  numpy_convtranspose  -- the scatter definition, written out by hand

Both take NHWC input (N, H, W, CI) and ConvTranspose weights [CI, CO, K, K] (the PyTorch and
ONNX layout) and return NHWC int64. Computed in float64: every partial sum here is an integer
far below 2^53, so the result is exact, not approximately equal.
"""
import numpy as np


def torch_convtranspose(x_nhwc, w, stride, pad):
    import torch
    import torch.nn as nn

    ci, co, k, _ = w.shape
    layer = nn.ConvTranspose2d(ci, co, kernel_size=k, stride=stride, padding=pad,
                               output_padding=0, dilation=1, groups=1, bias=False).double()
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(np.asarray(w, dtype=np.float64)))
        x = torch.from_numpy(np.ascontiguousarray(np.asarray(x_nhwc, dtype=np.float64).transpose(0, 3, 1, 2)))
        y = layer(x).numpy().transpose(0, 2, 3, 1)
    yr = np.round(y)
    assert np.array_equal(y, yr), "torch output is not integer-valued"
    return yr.astype(np.int64)


def numpy_convtranspose(x_nhwc, w, stride, pad):
    k = w.shape[2]
    w = np.asarray(w).astype(np.int64)
    out = []
    for x in np.asarray(x_nhwc).astype(np.int64):
        x = x.transpose(2, 0, 1)
        _, h, wd = x.shape
        full = np.zeros((w.shape[1], (h - 1) * stride + k, (wd - 1) * stride + k), dtype=np.int64)
        for ky in range(k):
            for kx in range(k):
                full[:, ky:ky + (h - 1) * stride + 1:stride, kx:kx + (wd - 1) * stride + 1:stride] += \
                    np.einsum("ihw,io->ohw", x, w[:, :, ky, kx])
        ho, wo = (h - 1) * stride - 2 * pad + k, (wd - 1) * stride - 2 * pad + k
        out.append(full[:, pad:pad + ho, pad:pad + wo].transpose(1, 2, 0))
    return np.stack(out)
