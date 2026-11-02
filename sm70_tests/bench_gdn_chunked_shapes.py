#!/usr/bin/env python3
"""Time the serial vs chunked SM70 GDN prefill kernel per head geometry.

The chunked chain only pays off for some per-rank geometries, so
`sm70_gdn._CHUNKED_FAST_GEOMETRIES` is measurement driven. Rerun this before
adding a geometry to that set.

Usage: python sm70_tests/bench_gdn_chunked_shapes.py [T ...]
"""

from __future__ import annotations

import statistics
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_gdn_chunked_heads import make_inputs  # noqa: E402

GEOMETRIES = [
    (2, 6, "Qwen3.8-Flash-Next TP8"),
    (8, 16, "Qwen3.6-35B TP2"),
    (16, 32, "Qwen3.6-35B TP1"),
]


def bench(fn, q, k, v, gl, beta, iters: int = 20) -> tuple[float, float]:
    for _ in range(3):
        fn(q, k, v, gl, beta, None, True)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn(q, k, v, gl, beta, None, True)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return min(times), statistics.median(times)


def main() -> int:
    from vllm.model_executor.layers.mamba.gdn.sm70_gdn import (
        chunk_gated_delta_rule_sm70,
        chunk_gated_delta_rule_sm70_chunked,
    )

    ts = [int(a) for a in sys.argv[1:]] or [4096, 8192]
    torch.cuda.set_device(0)
    print(f"{'geometry':24s} {'HQ':>3s} {'HV':>3s} {'T':>6s} "
          f"{'serial min/med':>18s} {'chunked min/med':>18s} {'speedup':>8s}")
    for HQ, HV, label in GEOMETRIES:
        for T in ts:
            q, k, v, gl, beta = make_inputs(T, HQ, HV, seed=1, dev="cuda:0")
            s_min, s_med = bench(chunk_gated_delta_rule_sm70, q, k, v, gl, beta)
            c_min, c_med = bench(chunk_gated_delta_rule_sm70_chunked, q, k, v, gl, beta)
            print(
                f"{label:24s} {HQ:3d} {HV:3d} {T:6d} "
                f"{s_min:8.2f}/{s_med:7.2f} ms {c_min:8.2f}/{c_med:7.2f} ms "
                f"{s_min / c_min:7.2f}x",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
