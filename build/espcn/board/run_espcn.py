#!/usr/bin/env python3
# Jude: Created
"""ESPCN super-resolution runner for the FINN accelerator on a KV260.

Copy alongside the generated driver and run with the Kria-PYNQ venv interpreter,
as root (PYNQ needs /dev/mem):

    sudo /usr/local/share/pynq-venv/bin/python run_espcn.py \
        --bitfile ../bitfile/finn-accel.bit \
        --input input.npy \
        --golden verify_folded_hls_cppsim_0.npy

Why not the shipped validate.py: it hardcodes `--dataset mnist|cifar10` and a
top-1 accuracy metric, which means nothing for super-resolution.

On the golden: use the build's own verify_folded_hls_cppsim output -- the post-HLS
behaviour of the exact design synthesised. It equals quant_espcn_x2_w4a4_base/
output_qonnx.npy bit for bit. The Brevitas output.npy is NOT a hardware golden: it
differs from the integer datapath by 1-3 levels on ~135 of 196,608 values.
(Before 2026-09-11 output.npy was also computed on 0..255 input, pearson r = 0.15.)
"""
import argparse
import time

import numpy as np
from driver import io_shape_dict
from driver_base import FINNExampleOverlay


def to_accel_layout(x, expected):
    """input.npy is NCHW float32 already scaled to 0..255; the accelerator wants
    UINT8 NHWC. Transpose only if that is actually what closes the gap."""
    if x.ndim == 4 and x.shape[1] == 3 and tuple(x.shape) != tuple(expected):
        x = x.transpose(0, 2, 3, 1)
    if x.min() < 0 or x.max() > 255:
        raise ValueError(f"input out of uint8 range: [{x.min()}, {x.max()}]")
    return np.ascontiguousarray(x.astype(np.uint8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bitfile", default="../bitfile/finn-accel.bit")
    p.add_argument("--input", default="input_nhwc_uint8.npy")
    p.add_argument("--golden", default="golden_nhwc_uint8.npy", help="reference .npy to compare against")
    p.add_argument("--out", default="board_output.npy")
    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--runs", type=int, default=5, help="timed repetitions")
    p.add_argument("--platform", default="zynq-iodma")
    p.add_argument("--weight-dir", default="weights/")
    # driver_base defaults fclk_mhz to 100, but the build targeted
    # synth_clk_period_ns=5.0 (200 MHz) and post-route timing closed with
    # WNS +0.150 ns / 0 failing endpoints. Leaving it at 100 halves throughput
    # for nothing: measured 344 ms @100 MHz vs 172 ms @200 MHz, bit-exact at both.
    p.add_argument("--fclk", type=float, default=200.0, help="accelerator clock MHz")
    args = p.parse_args()

    accel = FINNExampleOverlay(
        bitfile_name=args.bitfile,
        platform=args.platform,
        io_shape_dict=io_shape_dict,
        batch_size=args.batchsize,
        fclk_mhz=args.fclk,
        weight_dir=args.weight_dir,
    )
    print(f"accelerator clock: {args.fclk} MHz")
    ishape, oshape = accel.ishape_normal(), accel.oshape_normal()
    print(f"accelerator expects input  {ishape} ({accel.idt()})")
    print(f"accelerator expects output {oshape} ({accel.odt()})")

    x = to_accel_layout(np.load(args.input), ishape)
    if tuple(x.shape) != tuple(ishape):
        raise SystemExit(f"input shape {x.shape} != accelerator {tuple(ishape)}")
    print(f"input ready: {x.shape} {x.dtype} range [{x.min()}, {x.max()}]")

    y = accel.execute(x)
    print(f"output: {y.shape} {y.dtype} range [{y.min()}, {y.max()}]")
    np.save(args.out, y)
    print(f"saved -> {args.out}")

    if args.golden:
        g = np.load(args.golden)
        if g.ndim == 4 and g.shape[1] == 3 and tuple(g.shape) != tuple(y.shape):
            g = g.transpose(0, 2, 3, 1)          # NCHW golden -> NHWC
        if g.shape != y.shape:
            print(f"WARNING: golden {g.shape} vs output {y.shape}; skipping comparison")
        else:
            diff = np.abs(y.astype(np.float64) - g.astype(np.float64))
            exact = bool((y == g).all())
            print(f"\ncomparison vs {args.golden}")
            print(f"  bit-exact      : {exact}")
            print(f"  max |diff|     : {diff.max():.6g}")
            print(f"  mean |diff|    : {diff.mean():.6g}")
            print(f"  mismatched     : {int((y != g).sum())} / {y.size}")
            print(f"  pearson r      : {np.corrcoef(y.ravel(), g.ravel())[0, 1]:.6f}")

    # Timed repetitions. execute() includes fold/pack/copy, so this is end-to-end
    # per-inference latency as an application would see it, not just accelerator time.
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        accel.execute(x)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    print(f"\nend-to-end over {args.runs} runs:")
    print(f"  mean {times.mean()*1e3:.2f} ms   min {times.min()*1e3:.2f} ms"
          f"   -> {1.0/times.mean():.2f} fps")
    print(f"  (build estimate was 10.6 fps / 521 ms latency)")


if __name__ == "__main__":
    main()
