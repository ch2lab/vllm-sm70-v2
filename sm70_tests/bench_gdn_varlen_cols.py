#!/usr/bin/env python3
"""Time / compare the GDN varlen kernel's `cols` split at production shape.

The 250K-prefill trace shows gdn_forward_vlk_varlen_kernel at grid=(32,1,8) x
block=(32,4,1): 256 CTAs of 4 warps = 12.8 warps per SM, est. occupancy 20%.
The warp count is fixed by the algorithm at v_heads * (V/cols) / groups_per_warp
= 32 * 64 / 2 = 1024, so `cols` is the only knob that raises it; cols=1 would
give 2048 warps = 25.6/SM.  Each output element's recurrence is over tokens and
independent of how columns are grouped, so the two arms should agree bit for
bit - check it rather than assume it.

VLLM_SM70_GDN_COLS1 is read once into a static, so each arm runs in its own
process; run this file twice and diff the dumps.

Gate discipline: the gate is wall clock on the kernel at the production shape,
then the in-service prefill trace sum.  It is not the synthetic bench of a
different kernel.

    VLLM_SM70_GDN_COLS1=0 python sm70_tests/bench_gdn_varlen_cols.py --dump /tmp/c0
    VLLM_SM70_GDN_COLS1=1 python sm70_tests/bench_gdn_varlen_cols.py --dump /tmp/c1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_gdn_chunked_equiv import make  # noqa: E402


def graph_time(fn, reps: int = 8, attempts: int = 7) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    best = float("inf")
    for _ in range(attempts):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b))
    return best * 1e3 / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="8192")
    ap.add_argument("--seeds", default="9502")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    import vllm  # noqa: F401
    from vllm.model_executor.layers.mamba.gdn.sm70_gdn import (
        chunk_gated_delta_rule_sm70,
    )

    dev = "cuda"
    print(f"# VLLM_SM70_GDN_COLS1={os.environ.get('VLLM_SM70_GDN_COLS1','unset')}"
          f"  gpu={torch.cuda.get_device_name(0)}  (varlen path, log-gate)")
    for T in [int(x) for x in args.ts.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            # log_gate=True: the kernel expects a log-gate (feeding exp() NaNs it)
            q, k, v, gate, beta, cu = make(T, 1, seed, dev, torch.float32,
                                           16, 32, True)
            kw = dict(q=q, k=k, v=v, g=gate, beta=beta, initial_state=None,
                      output_final_state=True, cu_seqlens=cu,
                      use_qk_l2norm_in_kernel=False)
            o, fs = chunk_gated_delta_rule_sm70(**kw)
            torch.cuda.synchronize()
            t = graph_time(lambda: chunk_gated_delta_rule_sm70(**kw))
            print(f"  T={T:6d} seed={seed:<6} {t:9.1f} us   "
                  f"|o|sum={o.float().abs().sum().item():.6g}")
            if args.dump:
                for nm, tx in (("o", o), ("fs", fs)):
                    with open(f"{args.dump}.T{T}s{seed}.{nm}.bin", "wb") as fh:
                        fh.write(tx.detach().cpu().contiguous().numpy().tobytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
