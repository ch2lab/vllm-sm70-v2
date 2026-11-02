#!/usr/bin/env python3
"""Offline bench for the SM70 NVFP4 split-K decode attention kernel.

Reports ms/step and the effective KV bandwidth, so the gap to HBM peak is
visible per context length and split count.

KV bytes read per step (nvfp4, 4 bits + 1/16 scale byte):
    L * layers? no -- this is a single layer: L * Hkv * D * 2(K,V) * 0.5625

Usage:
  python sm70_tests/bench_decode_bw.py --lens 8192,65536,131072,250000 \
      --splits 40,80,160
"""

from __future__ import annotations

import argparse
import statistics

import torch

D = 256
NPG = 16  # production attention page size (decode kernel requires 16)


def make_inputs(kv_len: int, H: int, HQ: int, dev: str, nreq: int = 1):
    nb = (kv_len + NPG - 1) // NPG
    # [B, 2H, NPG, F] uint8, F = D + D/8
    kv = torch.randint(0, 255, (nb, 2 * H, NPG, D // 2 + D // 16),
                       dtype=torch.uint8, device=dev)
    bt = torch.arange(nb, dtype=torch.int32, device=dev).reshape(nreq, -1)
    q = torch.randn(nreq, HQ, D, dtype=torch.float16, device=dev) * 0.1
    out = torch.zeros(nreq, HQ, D, dtype=torch.float16, device=dev)
    kl = torch.full((nreq,), kv_len, dtype=torch.int32, device=dev)
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
    ap.add_argument("--splits", default="40,80,160")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    from vllm.v1.attention.ops.sm70_nvfp4_decode import sm70_nvfp4_paged_decode

    torch.cuda.set_device(0)
    dev = "cuda:0"
    H, HQ = 2, 16  # TP1 geometry
    G = HQ // H
    ratio_pad = 1 << (G - 1).bit_length()  # next pow2 of the GQA ratio
    print(f"{'kv_len':>8s} {'splits':>7s} {'min_ms':>8s} {'med_ms':>8s} "
          f"{'GB/s':>7s} {'ms/token@gpu':>13s}")
    for kv_len in [int(x) for x in args.lens.split(",")]:
        q, kv, bt, kl, out = make_inputs(kv_len, H, HQ, dev)
        for splits in [int(x) for x in args.splits.split(",")]:
            ws = torch.zeros(splits * 1 * H * ratio_pad * (D + 2),
                             dtype=torch.float32, device=dev)

            def call():
                sm70_nvfp4_paged_decode(
                    q, kv, bt, kl, out, 1.0, 1.0, D ** -0.5, -1, splits, ws,
                )

            mn, md = bench(call, args.iters)
            kv_bytes = kv_len * H * D * 2 * (0.5 + 1 / 16)  # K and V, nvfp4
            print(
                f"{kv_len:8d} {splits:7d} {mn:8.3f} {md:8.3f} "
                f"{kv_bytes / (mn * 1e-3) / 1e9:7.1f} {mn:13.3f}",
                flush=True,
            )
        del q, kv, bt, kl, out
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
