#!/usr/bin/env python3
"""Is XQA_PARTS=240 the single-rank race the kernel comment claims, and does
the partition count cost decode time?

serve_qwen36.sh exports VLLM_SM70_DECODE_XQA_PARTS=240, while
sm70_nvfp4_decode_cuda.py:218-223 says 240 on one rank "is a RACE -- rel up to
8e+1 and NaN, non-deterministic, and it is what produced the 'ductduct...'
degenerate output in service", with 120 the safe figure.  That claim has never
been tested.  This drives the real kernel at the TP1 decode geometry and checks
(a) run-to-run bitwise stability at each partition count, (b) agreement between
partition counts, (c) cost.

NVFP4 codes are 4-bit e2m1 (all 16 values finite) but the group scales are
e4m3, whose 0x7F is NaN -- scale bytes are therefore drawn from [1,126].
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

H, HQ, D, PAGE = 2, 16, 256, 16


def graph_time(fn, reps=50, attempts=7):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    best = float("inf")
    for _ in range(attempts):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best * 1e3 / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-len", type=int, default=10240)
    ap.add_argument("--parts", default="120,240,180,60")
    args = ap.parse_args()
    import vllm  # noqa: F401
    import vllm._custom_ops  # noqa: F401
    from vllm.v1.attention.ops.sm70_nvfp4_decode_cuda import _mod_xqa

    dev = "cuda"
    kv_len = args.kv_len
    npages = (kv_len + PAGE - 1) // PAGE
    blocks = npages + 8
    torch.manual_seed(9502)
    kv = torch.zeros(blocks, 2 * H, PAGE, 144, dtype=torch.uint8, device=dev)
    # per (head, token) the cache holds D/2 = 128 nvfp4 code bytes then
    # D/16 = 16 e4m3 scale bytes, i.e. F = 144
    kv[:, :, :, :D // 2] = torch.randint(0, 255, (blocks, 2 * H, PAGE, D // 2),
                                         dtype=torch.uint8, device=dev)
    kv[:, :, :, D // 2:D // 2 + D // 16] = torch.randint(
        1, 126, (blocks, 2 * H, PAGE, D // 16), dtype=torch.uint8, device=dev)
    perm = torch.randperm(blocks, dtype=torch.int32, device=dev)[:npages]
    bt = torch.zeros(1, npages, dtype=torch.int32, device=dev)
    bt[0] = perm
    kvlens = torch.full((1,), kv_len, dtype=torch.int32, device=dev)
    q = torch.randn(1, HQ, D, dtype=torch.float16, device=dev)
    sm_scale = 1.0 / (D ** 0.5)
    # bytes actually read per layer: both K and V, 2*KV heads, nvfp4 + e4m3 scale
    mb = kv_len * 2 * H * D * (0.5 + 1.0 / 16) / 1e6

    print(f"# kv_len={kv_len} pages={npages} H={H} HQ={HQ} D={D} "
          f"gpu={torch.cuda.get_device_name(0)}")
    print(f"{'parts':>6} {'us/call':>9} {'GB/s':>7} {'%HBM':>6} "
          f"{'stable':>7} {'agree120':>9} {'finite':>7}")
    ref = None
    for parts in [int(x) for x in args.parts.split(",")]:
        mod = _mod_xqa(H, HQ)
        ws = torch.zeros(1 * HQ * max(parts, 240) * 258, dtype=torch.float32,
                         device=dev)
        out = torch.zeros(1, HQ, D, dtype=torch.float16, device=dev)

        def call(mod=mod, ws=ws, out=out, parts=parts):
            mod.run(q, kv, bt, kvlens, ws, out, 1.0, 1.0, sm_scale, parts, -1,
                    1, None, None, 0, 0, PAGE, 0, 1)

        outs = []
        for _ in range(6):
            out.zero_()
            call()
            torch.cuda.synchronize()
            outs.append(out.view(torch.int16).clone())
        stable = all(torch.equal(outs[0], o) for o in outs[1:])
        finite = bool(torch.isfinite(out.float()).all())
        if parts == 120:
            ref = outs[0]
            agree = None
        else:
            agree = None if ref is None else bool(torch.equal(ref, outs[0]))
        t = graph_time(call)
        print(f"{parts:>6} {t:>9.2f} {mb/1e3/(t/1e6):>7.0f} "
              f"{mb/1e3/(t/1e6)/900*100:>5.1f}% {str(stable):>7} "
              f"{str(agree):>9} {str(finite):>7}")
    print(f"\n# env in service: VLLM_SM70_DECODE_XQA_PARTS="
          f"{os.environ.get('VLLM_SM70_DECODE_XQA_PARTS', 'unset(->240 by script)')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
