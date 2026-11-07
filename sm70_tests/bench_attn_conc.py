#!/usr/bin/env python3
"""Verify-attention cost at the concurrency shapes (loop path vs paired path).

The conc8 budget leaves the verify attention as the only remaining per-row
target (~1.1 ms/row estimated from 0.1355 ms per 2-row launch at R=1).  That
estimate has never been measured with R=8 requests in one batch, which is the
real conc8 shape (8 requests x 2 verify rows = 16 rows).  This measures it.

Shape: H=1 kv head, HQ=6 q heads, D=256, fp8 KV, page 1584, per-request
kv_len 2048 (the conc8 harness) and 8000 (single stream).

Usage: CUDA_VISIBLE_DEVICES=1 python sm70_tests/bench_attn_conc.py
"""
import os
import time

import torch

from vllm.v1.attention.ops.sm70_nvfp4_decode_cuda import paged_decode_fp8

H, HQ, D = 1, 6, 256


def bench(R: int, nqt: int, kv_len: int, page: int = 1584, splits: int = 40,
          iters: int = 100, pair: bool = False):
    dev = "cuda"
    rows = R * nqt
    n_blocks = (kv_len + page - 1) // page + 1
    kv = torch.randint(0, 256, (n_blocks, H, page, 2 * D),
                       dtype=torch.uint8, device=dev)
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev)
    bt = bt.unsqueeze(0).repeat(R, 1).contiguous()
    kv_lens = torch.full((R,), kv_len, dtype=torch.int32, device=dev)
    q = torch.randn(rows, HQ, D, dtype=torch.float16, device=dev) * 0.1
    out = torch.empty_like(q)
    G = HQ // H
    ws = torch.empty(rows * H * G * splits * (D + 2), dtype=torch.float32,
                     device=dev)

    os.environ["VLLM_SM70_XQA_PAIR"] = "1" if pair else "0"

    def call():
        paged_decode_fp8(q, kv, bt, kv_lens, out, 1.0, 1.0, D ** -0.5, -1,
                         splits, ws, kv_dtype="fp8_e4m3", nqt=nqt)

    for _ in range(10):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def main() -> None:
    print(f"verify attention | H={H} HQ={HQ} D={D} fp8 | 16 full-attn layers")
    print(f"{'R':>2} {'nqt':>4} {'kv':>6} {'loop ms':>9} {'pair ms':>9} "
          f"{'pair/loop':>10} {'x16 layers(ms)':>15}")
    for R, nqt, kv_len in ((1, 1, 8000), (1, 2, 8000), (8, 2, 2048),
                           (8, 2, 2048)):
        loop = bench(R, nqt, kv_len, pair=False)
        pair = bench(R, nqt, kv_len, pair=True)
        print(f"{R:2d} {nqt:4d} {kv_len:6d} {loop:9.4f} {pair:9.4f} "
              f"{pair/loop:10.2f} {pair*16:15.2f}")


if __name__ == "__main__":
    main()
