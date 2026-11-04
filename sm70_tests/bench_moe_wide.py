#!/usr/bin/env python3
"""Reproduce the prefill MoE expert GEMM offline and price it against the
123 TFLOPS HMMA ceiling (the kernel is 17.03 s / 11.8% of a 250K prefill at
only 24% of peak, per the 2026-11-05 attribution).

Shapes are the real ones: a full 8192-token chunk, top_k=8 over 256 experts,
w13 n=1024 k=2048 and w2 n=2048 k=512, route blocks of 16.  Routing is
synthetic but uniform (256 rows per expert, so no padding waste), which is what
the kernel's cost depends on; correctness of any change is proven separately by
an A/B bit-identity test, not by comparing against a reference GEMM here.

The env vars below are MEASUREMENT ARMS for the launcher's kernel selection,
mirroring what serve_qwen36.sh sets in production.  A winning arm becomes a
code default, not a knob the operator must set.

    VLLM_SM70_MOE_QPN2_WIDE=256 VLLM_SM70_MOE_QPN2_WARPS=16 \\
    VLLM_SM70_MOE_QPN2_NS=1 python sm70_tests/bench_moe_wide.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

TOPK = 8
E = 256
# Route-block width. 16 is production; 32 selects RowTiles=4 in the wide
# kernel (kMoeBlockM must equal moe_block_size), which halves the per-mma cost
# of the in-register e2m1 dequant at the price of double shared memory.
BLOCK = int(os.environ.get("SM70_BENCH_BLOCK", "16"))
PEAK_TFLOPS = 123.0


def graph_time(fn, reps=20, attempts=5):
    for _ in range(2):
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


def build(m, n, k, shards, dev, skew: float = 0.0, gscale_value=None,
          smin: int = 1):
    """Synthetic expert weights plus a route layout.

    ``skew`` controls how realistic the router is.  skew=0 spreads the
    m*top_k expanded slots evenly over the experts, which is what the original
    harness did -- and what flattered the RowTiles=4 arm, because an even count
    that is an exact multiple of the block size means *neither* path pays any
    padding waste.  skew>0 draws per-expert shares from a heavy-tailed
    distribution, so experts end on partial blocks and a wider route block
    costs real padded mma rows.

    ``gscale_value`` pins the fp32 global scale instead of drawing it from
    [0.001, 0.011).  Either is legal for the biased-scale arm -- with scale
    bytes <= 119 (e4m3 240) the default tops out at an effective scale of 2.64,
    so scale*2^14 = 43253 stays a normal fp16 -- but pinning it to the real
    checkpoint's value (max |effective scale| = 0.04 at max code 6, i.e.
    gscale ~ 1/150) makes the comparison happen at production magnitudes.

    ``smin`` is the floor of the random e4m3 group-scale byte.  It exists to
    isolate one effect: when the effective scale ``raw*gscale`` falls below
    2^-14, the *unbiased* dequant rounds the scale through an fp16 subnormal and
    loses mantissa bits, while the biased arm keeps it normal.  Raising smin so
    every effective scale is normal makes the two arms byte-identical, which is
    how that mechanism was confirmed.
    """
    torch.manual_seed(9502)
    codes = torch.randint(0, 255, (E, n, k // 2), dtype=torch.uint8, device=dev)
    scales = torch.randint(smin, 120, (E, n, k // 16), dtype=torch.uint8,
                           device=dev)
    if gscale_value is not None:
        gscale = torch.full(((E, shards) if shards > 1 else (E,)),
                            float(gscale_value), dtype=torch.float32,
                            device=dev)
    else:
        gscale = torch.rand((E, shards) if shards > 1 else (E,),
                            dtype=torch.float32, device=dev) * 0.01 + 0.001
    inp = torch.randn(m, k, dtype=torch.float16, device=dev)
    out = torch.zeros(m * TOPK, n, dtype=torch.float16, device=dev)

    slots = m * TOPK
    if skew <= 0.0:
        counts = torch.full((E,), slots // E, dtype=torch.long)
        counts[:slots - counts.sum().item()] += 1
    else:
        w = torch.rand(E, dtype=torch.float64).pow(1.0 / (1.0 + skew))
        w = w / w.sum()
        counts = torch.bincount(
            torch.multinomial(w.float(), slots, replacement=True),
            minlength=E)
    assert int(counts.sum()) == slots, int(counts.sum())

    # expanded slot s belongs to token s // TOPK; give each slot to an expert so
    # the per-expert counts match, then lay each expert out in B-wide blocks
    # with -1 in the padded slots (the kernel skips those on load and store).
    expert_of_slot = torch.empty(slots, dtype=torch.long)
    pos = 0
    for e in range(E):
        c = int(counts[e])
        expert_of_slot[pos:pos + c] = e
        pos += c
    blocks = int(sum((int(counts[e]) + BLOCK - 1) // BLOCK for e in range(E)))
    st = torch.full((blocks * BLOCK,), -1, dtype=torch.int32)
    eid = torch.empty(blocks, dtype=torch.int32)
    rows_by_expert = torch.argsort(expert_of_slot, stable=True)
    b = 0
    start = 0
    for e in range(E):
        c = int(counts[e])
        nb = (c + BLOCK - 1) // BLOCK
        eid[b:b + nb] = e
        rows = rows_by_expert[start:start + c]
        st[b * BLOCK:b * BLOCK + c] = rows.to(torch.int32)
        start += c
        b += nb
    assert b == blocks and start == slots
    npp = torch.full((1,), blocks * BLOCK, dtype=torch.int32, device=dev)
    tw = torch.rand(slots, dtype=torch.float32, device=dev)
    st, eid, npp = st.to(dev), eid.to(dev), npp.to(dev)
    return codes, scales, gscale, inp, out, st, eid, npp, tw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--dump", default="",
                    help="write the raw output bits to <dump>.<proj>.bin so two "
                         "launcher arms can be compared byte for byte")
    ap.add_argument("--skew", type=float, default=0.0,
                    help="heavy-tailed router (0 = even, which pays no padding "
                         "waste and therefore flatters wide route blocks)")
    ap.add_argument("--gscale", type=float, default=0.0,
                    help="pin the fp32 global scale (0 = draw from [0.001,0.011)); "
                         "use ~0.00667 for this checkpoint's production magnitude")
    ap.add_argument("--smin", type=int, default=1,
                    help="floor of the random e4m3 group-scale byte; >=8 keeps every "
                         "effective scale normal, which is where the biased arm is "
                         "provably bit-identical")
    ap.add_argument("--ab", action="store_true",
                    help="time the e2m1 dequant-feed arms (biased scale off/on) "
                         "interleaved inside THIS process.  Cross-process "
                         "absolutes are not comparable on this box, so a paired "
                         "same-process delta is the only readable number")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    import vllm  # noqa: F401
    import vllm._custom_ops  # noqa: F401
    from vllm.sm70_nvfp4_ops import qpn2_moe_gemm, qpn2_prepare_moe

    dev = "cuda"
    print(f"# gpu={torch.cuda.get_device_name(0)} chunk={args.chunk} "
          f"top_k={TOPK} experts={E} block={BLOCK} skew={args.skew} "
          f"WIDE={os.environ.get('VLLM_SM70_MOE_QPN2_WIDE','?')} "
          f"WARPS={os.environ.get('VLLM_SM70_MOE_QPN2_WARPS','?')} "
          f"NS={os.environ.get('VLLM_SM70_MOE_QPN2_NS','?')} "
          f"KSPLIT={os.environ.get('VLLM_SM70_MOE_QPN2_KSPLIT','?')}")
    print(f"{'proj':>5} {'n':>6} {'k':>6} {'TFLOP':>7} {'ms':>7} "
          f"{'TFLOPS':>7} {'%peak':>6}  finite")
    tot_us = 0.0
    for name, n, k, shards in (("w13", 1024, 2048, 2), ("w2", 2048, 512, 1)):
        codes_raw, scales_raw, gscale, inp, out, st, eid, npp, tw = build(
            args.chunk, n, k, shards, dev, args.skew,
            args.gscale if args.gscale > 0 else None, args.smin)
        codes, scales = qpn2_prepare_moe(codes_raw, scales_raw)

        def call():
            qpn2_moe_gemm(out, inp, codes, scales, gscale, st, eid, npp, tw,
                          TOPK, mul_topk_weights=True, num_shards=shards,
                          moe_block_size=BLOCK, fuse_act=False)

        call()
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(out.float()).all()) and bool(out.abs().sum() > 0)
        fl = 2 * args.chunk * TOPK * n * k
        if args.ab:
            per = {0: [], 1: []}
            for r in range(args.rounds):
                # alternate which arm goes first so a slow drift cannot favour
                # one of them
                for arm in ((0, 1) if r % 2 == 0 else (1, 0)):
                    os.environ["VLLM_SM70_MOE_QPN2_BIASED_SCALE"] = str(arm)
                    per[arm].append(graph_time(call))
            os.environ["VLLM_SM70_MOE_QPN2_BIASED_SCALE"] = "0"
            b0, b1 = min(per[0]), min(per[1])
            tf0 = fl / (b0 * 1e-6) / 1e12
            tf1 = fl / (b1 * 1e-6) / 1e12
            print(f"{name:>5} {n:>6} {k:>6} {fl/1e12:>7.3f} "
                  f"off={b0/1e3:7.3f}ms on={b1/1e3:7.3f}ms "
                  f"{tf0:>6.1f}/{tf1:>6.1f}TF "
                  f"{tf0/PEAK_TFLOPS*100:>5.1f}%/{tf1/PEAK_TFLOPS*100:>4.1f}% "
                  f"delta={100*(b1/b0-1):+6.2f}% "
                  f"spread={100*(max(per[0])/b0-1):.1f}%/"
                  f"{100*(max(per[1])/b1-1):.1f}% finite={ok}")
            if args.dump:
                for arm in (0, 1):
                    os.environ["VLLM_SM70_MOE_QPN2_BIASED_SCALE"] = str(arm)
                    out.zero_()
                    call()
                    torch.cuda.synchronize()
                    with open(f"{args.dump}.arm{arm}.{name}.bin", "wb") as fh:
                        fh.write(out.detach().cpu().numpy().tobytes())
                os.environ["VLLM_SM70_MOE_QPN2_BIASED_SCALE"] = "0"
            tot_us += b0
            continue
        if args.dump:
            # raw bits, so two launcher arms can be compared byte for byte
            with open(f"{args.dump}.{name}.bin", "wb") as fh:
                fh.write(out.detach().cpu().numpy().tobytes())
        t = graph_time(call)
        tot_us += t
        print(f"{name:>5} {n:>6} {k:>6} {fl/1e12:>7.3f} {t/1e3:>7.3f} "
              f"{fl/(t*1e-6)/1e12:>7.1f} {fl/(t*1e-6)/1e12/PEAK_TFLOPS*100:>5.1f}% "
              f" {str(ok):>5}")
    per_layer_ms = tot_us / 1e3
    print(f"\nper MoE layer (w13+w2): {per_layer_ms:.3f} ms; "
          f"x40 layers x {250000 // args.chunk} chunks = "
          f"{per_layer_ms * 40 * (250000 // args.chunk) / 1e3:.2f} s "
          f"(profile measured 17.03 s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
