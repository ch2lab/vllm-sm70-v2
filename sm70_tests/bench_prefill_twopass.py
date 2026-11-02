#!/usr/bin/env python3
"""Offline bench for the SM70 two-pass NVFP4 prefill attention kernel.

Measures an 8K-token prefill chunk at the END of a context of length kv_len
(i.e. the last chunk of a long prefill), so the KV allocation stays small while
the attention cost is representative. Reports ms and effective TFLOPS.

Attention FLOPs for a causal chunk of q tokens over L keys (all q rows are the
last `q` positions, so every row sees all L keys):
    2 * D * q * L * HQ * 2 (QK + PV)

Usage: python sm70_tests/bench_prefill_twopass.py [--qlens 8192] [--lens 8192,65536,131072,250000]
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch

D = 256
F = D + D // 8  # nvfp4 packed row: 256 codes + 32 scale bytes
NPG = 3728  # production attention page size (hybrid mamba alignment)


def make_inputs(T: int, kv_len: int, H: int, HQ: int, dev: str):
    n_blocks = (kv_len + NPG - 1) // NPG
    kv = torch.randint(
        0, 255, (n_blocks, 2 * H, NPG, F), dtype=torch.uint8, device=dev
    )
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).reshape(1, -1)
    q = torch.randn(T, HQ, D, dtype=torch.float16, device=dev) * 0.1
    out = torch.zeros(T, HQ, D, dtype=torch.float16, device=dev)
    qo_indptr = torch.tensor([0, T], dtype=torch.int32, device=dev)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=dev)
    return q, kv, bt, qo_indptr, kv_lens, out


def bench(fn, iters: int = 5) -> float:
    for _ in range(2):
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
    return min(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlens", default="8192")
    ap.add_argument("--lens", default="8192,65536,131072,250000")
    ap.add_argument("--bm", default="1024,2048")
    ap.add_argument("--bn", default="4096")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    os.environ["VLLM_SM70_PREFILL_TWOPASS"] = "1"
    from vllm.v1.attention.ops.sm70_nvfp4_prefill_twopass import (
        try_paged_prefill_twopass,
    )

    torch.cuda.set_device(0)
    dev = "cuda:0"
    H, HQ = 2, 16  # TP1 geometry
    print(f"{'q':>6s} {'kv_len':>8s} {'BM':>5s} {'BN':>5s} {'ms':>9s} {'TFLOPS':>8s}")
    for qlen in [int(x) for x in args.qlens.split(",")]:
        for kv_len in [int(x) for x in args.lens.split(",")]:
            q, kv, bt, qo, kl, out = make_inputs(qlen, kv_len, H, HQ, dev)
            for bm in [int(x) for x in args.bm.split(",")]:
                for bn in [int(x) for x in args.bn.split(",")]:
                    os.environ["VLLM_SM70_TWOPASS_BM"] = str(bm)
                    os.environ["VLLM_SM70_TWOPASS_BN"] = str(bn)

                    def call():
                        ok = try_paged_prefill_twopass(
                            q, kv, bt, qo, kl, out, 1.0, 1.0, D ** -0.5, -1,
                            qlen,
                        )
                        assert ok, "gate rejected the shape"

                    ms = bench(call, args.iters)
                    flops = 2 * D * qlen * kv_len * HQ * 2
                    print(
                        f"{qlen:6d} {kv_len:8d} {bm:5d} {bn:5d} {ms:9.2f} "
                        f"{flops / (ms * 1e-3) / 1e12:8.1f}",
                        flush=True,
                    )
            del q, kv, bt, qo, kl, out
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
