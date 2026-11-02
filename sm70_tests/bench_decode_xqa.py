#!/usr/bin/env python3
"""Offline bench for the PRODUCTION SM70 nvfp4 decode kernel (XQA shape).

Measures one layer of decode attention (the XQA kernel, 16-token pages,
120 parts) and reports ms + effective KV bandwidth. `--window` reproduces the
bounded path (`window_left` tokens) so the ceiling of "sink + recent pages"
can be read directly.

Usage:
  python sm70_tests/bench_decode_xqa.py --lens 8192,65536,131072,250000 \
      --windows -1,65536,32768
"""

from __future__ import annotations

import argparse
import statistics

import torch

D = 256
F = D // 2 + D // 16  # 144: nvfp4 codes + scales
NPG = 16
NPARTS = 120  # measured exact part count for the XQA kernel


def make_inputs(kv_len: int, H: int, HQ: int, dev: str):
    nb = (kv_len + NPG - 1) // NPG
    kv = torch.randint(0, 255, (nb, 2 * H, NPG, F), dtype=torch.uint8, device=dev)
    bt = torch.arange(nb, dtype=torch.int32, device=dev).reshape(1, -1)
    q = torch.randn(1, HQ, D, dtype=torch.float16, device=dev) * 0.1
    out = torch.zeros(1, HQ, D, dtype=torch.float16, device=dev)
    kl = torch.full((1,), kv_len, dtype=torch.int32, device=dev)
    return q, kv, bt, kl, out


def bench(fn, iters: int = 20) -> tuple[float, float]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="8192,65536,131072,250000")
    ap.add_argument("--windows", default="-1")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    from vllm.v1.attention.ops.sm70_nvfp4_decode_cuda import _mod_xqa

    torch.cuda.set_device(0)
    dev = "cuda:0"
    H, HQ = 2, 16  # TP1
    mod = _mod_xqa(H, HQ)
    mod.prepare()
    ws = torch.zeros(1 * H * 8 * NPARTS * (D + 2), dtype=torch.float32, device=dev)
    print(f"{'kv_len':>8s} {'window':>8s} {'min_ms':>8s} {'med_ms':>8s} "
          f"{'read_MB':>8s} {'GB/s':>7s}")
    for kv_len in [int(x) for x in args.lens.split(",")]:
        q, kv, bt, kl, out = make_inputs(kv_len, H, HQ, dev)
        for win in [int(x) for x in args.windows.split(",")]:
            def call():
                mod.run(q, kv, bt, kl, ws, out, 1.0, 1.0, D ** -0.5, NPARTS,
                        win, 1, None, None, 0, 0, NPG, 0, 1)

            mn, md = bench(call, args.iters)
            eff = kv_len if win < 0 else min(kv_len, win)
            mb = eff * H * D * 2 * (0.5 + 1 / 16) / 1e6
            print(
                f"{kv_len:8d} {win:8d} {mn:8.3f} {md:8.3f} {mb:8.1f} "
                f"{mb / mn:7.1f}",
                flush=True,
            )
        del q, kv, bt, kl, out
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
