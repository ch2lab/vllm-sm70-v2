#!/usr/bin/env python3
"""Compare the SM70 GDN chunked kernel against the incumbent varlen kernel at
this model's real geometry, and price the difference.

Verdict, measured 2026-11-05: the chunked kernel is numerically correct at
(hq,hv)=(16,32) -- max relative output difference 7.6e-4, final-state 3.6e-4 --
but 2.05x SLOWER than the incumbent (7.23 vs 14.87 ms at T=8192, the production
chunk size), and 3.8x slower at two sequences.  Its per-chunk state tensor
h_ws is [T/64 * N, Hv, 128, 128] fp32, i.e. 268 MB at T=8192 and Hv=32, which
is why it only pays off at the small head counts in _CHUNKED_FAST_GEOMETRIES
({(2,6),(8,16),(4,8)}; at (8,16) it measures 1.09x).  So excluding (16,32) is
correct and must stay; speeding GDN prefill needs the state kept in registers
per CTA (strata's src/prefill/gdn_chunk.cu does exactly that) rather than
materialised per chunk.

Two traps this harness exists to record:
  * the gate must be a **log**-gate.  Feeding exp(gate) makes both kernels
    return NaN, which looks like a kernel bug at first; --log-gate is the
    convention that matches production.
  * the two kernels are different algorithms (serial recurrence vs chunked
    cumsum/kkt/fwd), so agreement is tolerance-based, not bit-identity.  A
    change here must never be shipped on a bit-identical claim.

Geometry is the model's, read from the running service's own debug valve
(VLLM_SM70_GDNDBG=1): q/k = [1, T, 16, 128] fp16, v = [1, T, 32, 128] fp16,
g/beta = [1, T, 32].  The model applies L2-norm upstream
(fused_post_conv_prep) and calls with use_qk_l2norm_in_kernel=False, so the
harness pre-normalises q/k and passes the same flag -- it compares the two
production wrappers, not raw extension calls.

    python sm70_tests/test_gdn_chunked_equiv.py --log-gate
    python sm70_tests/test_gdn_chunked_equiv.py --ts 1024,8192 --seqs 1,3
"""
from __future__ import annotations

import argparse
import sys

import torch

HQ, HV, K, V = 16, 32, 128, 128


def l2n(x: torch.Tensor) -> torch.Tensor:
    return x / x.pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-12)


def make(T: int, nseq: int, seed: int, dev: str, gdtype: torch.dtype,
         hq: int = HQ, hv: int = HV, log_gate: bool = False):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = l2n(torch.randn(1, T, hq, K, generator=gen).to(
        device=dev, dtype=torch.float16))
    k = l2n(torch.randn(1, T, hq, K, generator=gen).to(
        device=dev, dtype=torch.float16))
    v = torch.randn(1, T, hv, V, generator=gen).to(
        device=dev, dtype=torch.float16)
    # gate is exp(negative decay) -> (0,1]; beta is a sigmoid -> (0,1)
    ggate = torch.rand(1, T, hv, generator=gen).to(
        device=dev, dtype=torch.float32)
    gate = -0.02 * ggate if log_gate else torch.exp(-0.02 * ggate)
    gate = gate.to(gdtype).contiguous()
    braw = torch.rand(1, T, hv, generator=gen).to(
        device=dev, dtype=torch.float32)
    beta = torch.sigmoid(2.0 * (braw - 0.5)).to(gdtype).contiguous()
    if nseq == 1:
        cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    else:
        # uneven splits: chunk boundaries must not all be 64-aligned
        step = T // nseq
        bounds = [0] + [step * i + (i * 7) % step for i in range(1, nseq)] + [T]
        cu = torch.tensor(bounds, dtype=torch.int32, device=dev)
    return q, k, v, gate, beta, cu


def graph_time(fn, reps=10, attempts=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(reps):
            fn()
    best = float("inf")
    for _ in range(attempts):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); gr.replay(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best * 1e3 / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="1024,2048,8192")
    ap.add_argument("--seqs", default="1,3")
    ap.add_argument("--seeds", default="9502,7")
    ap.add_argument("--gdtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--hq", type=int, default=HQ)
    ap.add_argument("--hv", type=int, default=HV)
    ap.add_argument("--log-gate", action="store_true",
                    help="feed g as a log-gate (negative) instead of exp()")
    ap.add_argument("--rel-tol", type=float, default=2e-2)
    args = ap.parse_args()

    import vllm  # noqa: F401
    from vllm.model_executor.layers.mamba.gdn.sm70_gdn import (
        chunk_gated_delta_rule_sm70,
        chunk_gated_delta_rule_sm70_chunked,
    )

    dev = "cuda"
    gd = torch.float32 if args.gdtype == "float32" else torch.float16
    bad = 0
    print(f"# geometry q[1,T,{args.hq},{K}] v[1,T,{args.hv},{V}] "
          f"g/beta dtype={args.gdtype} log_gate={args.log_gate} "
          f"gpu={torch.cuda.get_device_name(0)}")
    print(f"{'T':>6} {'nseq':>5} {'seed':>5} {'max|d|':>10} {'max rel':>10} "
          f"{'fs rel':>9} {'varlen_ms':>10} {'chunked_ms':>11} {'speedup':>8}")
    for T in [int(x) for x in args.ts.split(",")]:
        for nseq in [int(x) for x in args.seqs.split(",")]:
            if nseq > 1 and T // nseq < 64:
                continue
            for seed in [int(x) for x in args.seeds.split(",")]:
                q, k, v, g, beta, cu = make(
                    T, nseq, seed, dev, gd, args.hq, args.hv, args.log_gate)
                kw = dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=None,
                          output_final_state=True, cu_seqlens=cu,
                          use_qk_l2norm_in_kernel=False)
                try:
                    o_ref, s_ref = chunk_gated_delta_rule_sm70(**kw)
                    o_new, s_new = chunk_gated_delta_rule_sm70_chunked(**kw)
                except Exception as exc:  # noqa: BLE001
                    print(f"{T:>6} {nseq:>5} {seed:>5}  ERROR {type(exc).__name__}: "
                          f"{str(exc)[:80]}")
                    bad += 1
                    continue
                torch.cuda.synchronize()
                a, b = o_ref.float(), o_new.float()
                d = (a - b).abs()
                denom = a.abs().amax().clamp(min=1e-6)
                rel = float((d.max() / denom))
                fs = float("nan")
                if s_ref is not None and s_new is not None:
                    fs = float(((s_ref.float() - s_new.float()).abs().max()
                                / s_ref.float().abs().amax().clamp(min=1e-6)))
                t_ref = graph_time(lambda: chunk_gated_delta_rule_sm70(**kw))
                t_new = graph_time(
                    lambda: chunk_gated_delta_rule_sm70_chunked(**kw))
                ok = rel <= args.rel_tol and (fs != fs or fs <= args.rel_tol)
                bad += 0 if ok else 1
                print(f"{T:>6} {nseq:>5} {seed:>5} {float(d.max()):>10.2e} "
                      f"{rel:>10.2e} {fs:>9.2e} {t_ref/1e3:>10.3f} "
                      f"{t_new/1e3:>11.3f} {t_ref/t_new:>7.2f}x"
                      f"{'' if ok else '   <-- OVER TOL'}")
    print("RESULT:", "agrees within tolerance everywhere" if bad == 0
          else f"{bad} case(s) failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
