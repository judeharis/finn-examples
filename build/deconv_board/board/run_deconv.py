#!/usr/bin/env python3
# Jude: Created
"""Board runner for the standalone fused Deconvolution_hls accelerator (KV260).

Copy alongside the FINN-generated driver and run inside the activated Kria-PYNQ venv
as root -- see deploy_to_board.sh for the environment it needs.

Checks three things, because each can fail independently:

  1. single frame   -- frame 0 via execute(), against the build's ONNX golden
  2. frame sequence -- frames 1..N-1, one execute() each. The kernel is
     free-running (ap_ctrl_none) and keeps its loop counters and buffer pointers
     in `static` variables, so the first frame after a bitstream load is the only
     one that starts from reset. FINN's rtlsim only ever ran one frame, so this is
     the first test of the end-of-frame wrap-around.
  3. batched        -- all N frames back-to-back in one DMA transfer
Then times the accelerator alone (throughput_test: DMA start to done) and turns
that into cycles per frame, to compare against the cycle model.
"""
import argparse
import json

import numpy as np
from driver import io_shape_dict
from driver_base import FINNExampleOverlay

import fast_packing


def compare(tag, y, g):
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if y.shape != g.shape:
        print(f"  {tag:28s} SHAPE MISMATCH {y.shape} vs {g.shape}")
        return False
    bad = int((y != g).sum())
    ok = bad == 0
    msg = f"  {tag:28s} {'bit-exact' if ok else 'MISMATCH'}   mismatched {bad} / {y.size}"
    if not ok:
        idx = np.argwhere(y != g)[0]
        msg += f"   first at {tuple(idx)}: got {y[tuple(idx)]:.0f} want {g[tuple(idx)]:.0f}"
    print(msg)
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bitfile", default="../bitfile/finn-accel.bit")
    p.add_argument("--frames-in", default="frames_in.npy", help="(N,128,128,32) NHWC")
    p.add_argument("--frames-gold", default="frames_gold.npy", help="(N,256,256,3) NHWC")
    p.add_argument("--config", default="deconv_config.json")
    p.add_argument("--fclk", type=float, default=None,
                   help="accelerator clock MHz; default is the synthesis clock. driver_base's "
                        "own default is 100, which silently halves throughput at 5 ns.")
    p.add_argument("--runs", type=int, default=3, help="throughput_test repetitions")
    p.add_argument("--platform", default="zynq-iodma")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    fclk = args.fclk if args.fclk else 1000.0 / cfg["synth_clk_period_ns"]
    xs = np.load(args.frames_in)
    gs = np.load(args.frames_gold)
    n = xs.shape[0]
    assert gs.shape[0] == n, "frames_in and frames_gold disagree on N"

    accel = FINNExampleOverlay(
        bitfile_name=args.bitfile,
        platform=args.platform,
        io_shape_dict=io_shape_dict,
        batch_size=1,
        fclk_mhz=fclk,
        runtime_weight_dir="runtime_weights/",
    )
    # The stock pack/unpack take minutes per frame for UINT4/INT32 on the A53s.
    fast_packing.install(accel)
    print(f"config: K={cfg['K']} S={cfg['S']} P={cfg['P']} {cfg['H']}x{cfg['W']}x{cfg['CI']} -> "
          f"CO={cfg['CO']}  PE={cfg['PE']} SIMD={cfg['SIMD']}  synth {cfg['synth_clk_period_ns']} ns")
    print(f"accelerator clock : {fclk:g} MHz")
    print(f"input  {accel.ishape_normal()} {accel.idt()}   output {accel.oshape_normal()} {accel.odt()}")
    print(f"frames: {n}\n")

    results = []
    print("correctness")
    y0 = accel.execute(xs[0:1])
    np.save("board_output_frame0.npy", y0)
    results.append(compare("single frame 0", y0, gs[0:1]))
    for i in range(1, n):
        results.append(compare(f"sequential frame {i}", accel.execute(xs[i:i + 1]), gs[i:i + 1]))
    # rerun frame 0 last: catches state left over from the frames before it
    results.append(compare("frame 0 again", accel.execute(xs[0:1]), gs[0:1]))
    if n > 1:
        accel.batch_size = n
        results.append(compare(f"batched x{n}", accel.execute(xs), gs))
        accel.batch_size = 1

    print("\ntiming (throughput_test: DMA start -> done, accelerator only)")
    timing = {}
    for bs in sorted({1, n}):
        accel.batch_size = bs
        runs = [accel.throughput_test() for _ in range(args.runs)]
        ms = float(np.median([r["runtime[ms]"] for r in runs]))
        cyc = ms * 1e-3 * fclk * 1e6 / bs
        timing[bs] = dict(runtime_ms=ms, cycles_per_frame=cyc)
        print(f"  batch {bs:2d}: {ms:9.3f} ms  -> {cyc:,.0f} cycles/frame  ({bs / (ms * 1e-3):.2f} fps)")
    exp = cfg.get("get_exp_cycles")
    if exp:
        c = timing[max(timing)]["cycles_per_frame"]
        print(f"\n  get_exp_cycles()    : {exp:,}")
        print(f"  measured - expected : {c - exp:+,.0f} cycles ({100 * (c - exp) / exp:+.3f}%)")

    json.dump(dict(all_bit_exact=all(results), fclk_mhz=fclk, timing=timing, config=cfg),
              open("board_results.json", "w"), indent=2)
    print(f"\nALL BIT-EXACT: {all(results)}   (-> board_results.json)")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
