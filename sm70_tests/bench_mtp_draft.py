"""Standalone microbenchmark of the Qwen3_5MTP draft forward, block by block.

The draft costs 4.25 ms per pass (k=1 vs k=3 slope on the 27B), but its
internal split is unknown: shared lm_head vs the one transformer layer vs AR.
Both nsys attempts to decompose it failed, so this measures the pieces
directly at the 27B TP4 shapes.

The MTP weights are excluded from NVFP4 quantization (`ignore: re:.*mtp.*`),
so every block below is a plain fp16 weight read at M=1 rows; the floor is
weight bytes / 900 GB/s.

Per TP4 rank (hidden 5120, intermediate 17408, vocab/4 = 37984):
  fc          : 10240 -> 5120
  qkv         : 5120  -> (6 + 2*1) * 256 = 2048
  o           : 1536  -> 5120
  gate/up     : 5120  -> 17408  (x2)
  down        : 17408 -> 5120
  lm_head     : 5120  -> 37984  (shared with the target, vocab-sharded)

Usage: CUDA_VISIBLE_DEVICES=1 python sm70_tests/bench_mtp_draft.py
"""

import time

import torch

HBM_GBS = 900.0
H = 5120
INTER = 17408
VOCAB_TP = 37984
QO, KVH, HD = 6, 1, 256


def bench_gemm(n: int, k: int, m: int = 1, iters: int = 100):
    dev = "cuda"
    w = torch.randn(n, k, dtype=torch.float16, device=dev) * 0.02
    x = torch.randn(m, k, dtype=torch.float16, device=dev) * 0.1
    for _ in range(10):
        torch.matmul(x, w.t())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(x, w.t())
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters
    gbs = n * k * 2 / ms / 1e6
    return ms, gbs


def main() -> None:
    blocks = [
        ("fc (10240->5120)", 5120, 2 * H),
        ("qkv (5120->2048)", (QO + 2 * KVH) * HD, H),
        ("o   (1536->5120)", H, QO * HD),
        ("gate/up x2", 2 * INTER, H),
        ("down (17408->5120)", H, INTER),
        ("lm_head (5120->37984)", VOCAB_TP, H),
    ]
    print(f"MTP draft blocks | M=1 | HBM {HBM_GBS:.0f} GB/s")
    print(f"{'block':24s} {'ms':>8} {'GB/s':>8} {'%floor':>7}")
    total = 0.0
    for name, n, k in blocks:
        ms, gbs = bench_gemm(n, k)
        total += ms
        print(f"{name:24s} {ms:8.3f} {gbs:8.1f} {gbs/HBM_GBS*100:6.1f}%", flush=True)
    print(f"{'SUM (1 pass)':24s} {total:8.3f}")
    print(f"measured cost per draft pass: 4.25 ms -> overhead vs this sum: "
          f"{4.25 - total:+.3f} ms")


if __name__ == "__main__":
    main()
