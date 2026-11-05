#!/usr/bin/env python3
"""Reproduce the in-service decode_xqa16 timing (418 us/call at conc8,
span~2300) offline. The existing benches run R=1, warm L2, one KV buffer;
the service runs R=8, L2 flushed by MoE/GDN traffic between calls, and 12
QSA layers rotating over distinct KV tensors. This bench turns each
condition on/off to find which one explains the 5.2x gap (80.7 -> 418 us).

NVFP4 scale bytes drawn from [1,126] (0x7F is NaN in e4m3)."""
from __future__ import annotations

import argparse
import os
import sys

import torch

H, HQ, D, PAGE = 2, 16, 256, 16
NPARTS = int(os.environ.get("VLLM_SM70_DECODE_XQA_PARTS", "240"))


def time_calls(call, n_warm, n_iter, flush=None, l2buf=None):
    """Per-call GPU time via events; optional L2 flush / stream before each
    call (excluded from the timed window by recording after enqueue)."""
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n_iter)]
    for _ in range(n_warm):
        call(0)
    torch.cuda.synchronize()
    for i in range(n_iter):
        if flush is not None and l2buf is not None:
            flush(l2buf)
        s, e = ev[i]
        s.record()
        call(i)
        e.record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in ev)
    return ts[len(ts) // 2] * 1e3  # median us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-len", type=int, default=20069)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--parts", type=int, default=NPARTS)
    args = ap.parse_args()
    import vllm  # noqa: F401
    import vllm._custom_ops  # noqa: F401
    from vllm.v1.attention.ops.sm70_nvfp4_decode_cuda import _mod_xqa

    dev = "cuda"
    R, L = args.r, args.layers
    kv_len = args.kv_len
    npages = (kv_len + PAGE - 1) // PAGE
    blocks = R * npages + 64
    torch.manual_seed(9502)

    # one pool, L layer-slices (distinct tensors like the per-layer caches)
    pool = torch.zeros(L, blocks, 2 * H, PAGE, 144, dtype=torch.uint8, device=dev)
    pool[:, :, :, :, : D // 2] = torch.randint(
        0, 255, (L, blocks, 2 * H, PAGE, D // 2), dtype=torch.uint8, device=dev)
    pool[:, :, :, :, D // 2:D // 2 + D // 16] = torch.randint(
        1, 126, (L, blocks, 2 * H, PAGE, D // 16), dtype=torch.uint8, device=dev)

    mod = _mod_xqa(H, HQ)
    ws = torch.zeros(R * HQ * max(args.parts, 240) * 258, dtype=torch.float32, device=dev)
    out = torch.zeros(R, HQ, D, dtype=torch.float16, device=dev)
    q = torch.randn(R, HQ, D, dtype=torch.float16, device=dev)
    kvlens = torch.full((R,), kv_len, dtype=torch.int32, device=dev)
    sm_scale = 1.0 / (D**0.5)

    def make_bt(scatter):
        if scatter:
            bt = torch.stack([
                torch.randperm(blocks, dtype=torch.int32, device=dev)[:npages]
                for _ in range(R)])
        else:
            bt = (torch.arange(R * npages, dtype=torch.int32, device=dev)
                  .view(R, npages))
        return bt.contiguous()

    def flush_l2(buf):
        buf.zero_()  # 64 MB write >> 6 MB L2

    def stream_moe(buf):
        torch.sum(buf)  # read-only streaming ~ MoE weight pass

    l2 = torch.zeros(16, 1024 * 1024, dtype=torch.float32, device=dev)   # 64 MB
    moe = torch.randn(100 * 1024 * 1024 // 2, dtype=torch.float16, device=dev)  # 100 MB

    mb_per_call = R * kv_len * 2 * H * D * (0.5 + 1.0 / 16) / 1e6
    print(f"# kv_len={kv_len} R={R} layers={L} parts={args.parts} "
          f"gpu={torch.cuda.get_device_name(0)}  "
          f"KV read/call={mb_per_call:.1f} MB -> HBM floor "
          f"{mb_per_call/1e3/900*1e6:.1f} us")
    print(f"{'arm':>34} {'us/call':>9} {'%HBM':>6}")

    def run_arm(name, bt, rotate, between):
        def call(i, bt=bt, rotate=rotate, between=between):
            lay = (i % L) if rotate else 0
            mod.run(q, pool[lay], bt, kvlens, ws, out, 1.0, 1.0, sm_scale,
                    args.parts, -1, R, None, None, 0, 0, PAGE, 0, 1)
        if between == "flush":
            t = time_calls(call, 20, 200, flush=flush_l2, l2buf=l2)
        elif between == "moe":
            t = time_calls(call, 20, 200, flush=stream_moe, l2buf=moe)
        else:
            t = time_calls(call, 20, 200)
        print(f"{name:>34} {t:>9.1f} {mb_per_call/1e3/(t/1e6)/900*100:>5.1f}%")
        return t

    bt_sc, bt_seq = make_bt(True), make_bt(False)
    run_arm("warm seq-bt single-layer", bt_seq, False, None)
    run_arm("warm scat-bt single-layer", bt_sc, False, None)
    run_arm("warm scat-bt 12-layer rotate", bt_sc, True, None)
    run_arm("L2-flush scat-bt 12L", bt_sc, True, "flush")
    run_arm("MoE-stream scat-bt 12L", bt_sc, True, "moe")
    run_arm("MoE-stream seq-bt 12L", bt_seq, True, "moe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
