#!/usr/bin/env python3
"""Prototype: content-based page selection for the SM70 XQA decode kernel.

Proves the mechanism the plan's Phase 3 needs, without touching the engine:

  1. score every 16-token KV page with the query (max over the page's tokens of
     q.k, i.e. a Quest-style upper bound computed exactly for the prototype),
  2. keep the top-k pages (plus everything inside a forced sink/recent window),
  3. compact them into a short block table and run the *production* XQA kernel on
     that table (the same trick `try_paged_decode_cuda` already uses),
  4. report key-mass retained vs budget and ms/layer vs budget.

Two KV distributions are measured so the selector is validated, not just timed:
  - uniform: random K (attention is near-uniform, so mass ~= budget),
  - planted: 1% of pages hold high-norm K (a selector that works must recover
    far more mass than the budget).

Usage: python sm70_tests/proto_decode_pagesel.py --kv 250000 --budgets 0.015,0.03,0.06,0.125,0.25,0.5
"""

from __future__ import annotations

import argparse
import statistics

import torch

D = 256
F = D // 2 + D // 16
NPG = 16
NPARTS = 120



_LEVELS = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def encode_nvfp4(k: torch.Tensor, v: torch.Tensor, H: int, npg: int) -> torch.Tensor:
    """K/V [ntok, H, D] fp16 -> packed page tensor [nb, 2H, npg, F] uint8."""
    ntok, Dh = k.shape[0], k.shape[2]
    nb = (ntok + npg - 1) // npg
    dev = k.device
    pad = nb * npg - ntok
    if pad:
        k = torch.cat([k, torch.zeros(pad, H, Dh, dtype=k.dtype, device=dev)])
        v = torch.cat([v, torch.zeros(pad, H, Dh, dtype=v.dtype, device=dev)])

    def quant(x: torch.Tensor):
        # x [nb*npg, H, D] -> codes uint8 [.., D/2], scales uint8 [.., D/16]
        g = x.float().view(nb * npg, H, Dh // 16, 16)
        amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (amax / 6.0).squeeze(-1)                     # [.., D/16]
        idx = (g.abs() / (amax / 6.0)).round().clamp(0, 6)
        codes = idx.to(torch.uint8) | (g.sign() < 0).to(torch.uint8) << 3
        codes = codes.reshape(nb * npg, H, Dh)
        packed = (codes[..., 0::2] & 0xF) | ((codes[..., 1::2] & 0xF) << 4)
        su8 = scale.to(torch.float8_e4m3fn).view(torch.uint8)
        return packed.contiguous(), su8.contiguous()

    kc, ks = quant(k)
    vc, vs = quant(v)
    dd, sd = Dh // 2, Dh // 16
    flat = torch.zeros(nb, H * npg * dd + H * npg * sd + H * npg * dd + H * npg * sd,
                       dtype=torch.uint8, device=dev)
    o = 0
    flat[:, o:o + H * npg * dd] = kc.view(nb, H * npg * dd); o += H * npg * dd
    flat[:, o:o + H * npg * sd] = ks.view(nb, H * npg * sd); o += H * npg * sd
    flat[:, o:o + H * npg * dd] = vc.view(nb, H * npg * dd); o += H * npg * dd
    flat[:, o:o + H * npg * sd] = vs.view(nb, H * npg * sd)
    return flat.view(nb, 2 * H, npg, dd + sd)


def dequant_page_tokens(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[nb, 2H, 16, 144] uint8 -> fp16 K, V [ntok, H, D] (prototype only)."""
    nb = kv.shape[0]
    H = kv.shape[1] // 2
    npg, f = kv.shape[2], kv.shape[3]
    dd, sd = D // 2, D // 16
    # Page layout is flat: [K data H*N*DD][K scales H*N*SD][V data][V scales].
    flat = kv.view(nb, -1)
    kd = flat[:, : H * npg * dd].view(nb, H, npg, dd)
    o = H * npg * dd
    ks = flat[:, o: o + H * npg * sd].view(nb, H, npg, sd)
    o = H * npg * f
    vd = flat[:, o: o + H * npg * dd].view(nb, H, npg, dd)
    o = H * npg * f + H * npg * dd
    vs = flat[:, o: o + H * npg * sd].view(nb, H, npg, sd)

    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float16, device=kv.device)

    def unpack(codes: torch.Tensor) -> torch.Tensor:
        lo = lut[(codes & 0xF).long()]
        hi = lut[(codes >> 4).long()]
        out = torch.empty(codes.shape[:-1] + (codes.shape[-1] * 2,),
                          dtype=torch.float16, device=codes.device)
        out[..., 0::2] = lo
        out[..., 1::2] = hi
        return out

    def scales(s: torch.Tensor) -> torch.Tensor:
        # e4m3 -> fp16 approximation good enough for a score/selector prototype
        b = s.to(torch.int32)
        sign = torch.where((b & 0x80) != 0, -1.0, 1.0)
        e = (b >> 3) & 0xF
        m = (b & 7).float()
        val = torch.where(e == 0, m * 0.001953125, (1 + m * 0.125) * torch.pow(2.0, e - 7))
        return (sign * val).to(torch.float16)

    k = (unpack(kd) * scales(ks).repeat_interleave(16, dim=-1)).reshape(nb, H, npg, D)
    v = (unpack(vd) * scales(vs).repeat_interleave(16, dim=-1)).reshape(nb, H, npg, D)
    k = k.permute(0, 2, 1, 3).reshape(nb * npg, H, D)
    v = v.permute(0, 2, 1, 3).reshape(nb * npg, H, D)
    return k, v


def bench(fn, iters: int = 20) -> float:
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
    return min(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", type=int, default=250000)
    ap.add_argument("--budgets", default="0.015,0.03,0.06,0.125,0.25,0.5")
    ap.add_argument("--planted", type=float, default=0.005)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    from vllm.v1.attention.ops.sm70_nvfp4_decode_cuda import _mod_xqa

    torch.cuda.set_device(0)
    dev = "cuda:0"
    H, HQ = 2, 16
    G = HQ // H
    kv_len = args.kv
    nb = (kv_len + NPG - 1) // NPG
    mod = _mod_xqa(H, HQ)
    mod.prepare()
    ws = torch.zeros(H * 8 * NPARTS * (D + 2), dtype=torch.float32, device=dev)

    q = torch.randn(1, HQ, D, dtype=torch.float16, device=dev) / D ** 0.5
    out = torch.zeros(1, HQ, D, dtype=torch.float16, device=dev)
    bt_full = torch.arange(nb, dtype=torch.int32, device=dev).reshape(1, -1)
    kl_full = torch.tensor([kv_len], dtype=torch.int32, device=dev)

    for tag in ("uniform", "planted"):
        k_f = (torch.randn(nb * NPG, H, D, device=dev) * 0.5)
        v_f = (torch.randn(nb * NPG, H, D, device=dev) * 0.5)
        if tag == "planted":
            # 1% of pages carry K with 4x the norm: a working selector must find
            # them and keep far more mass than the budget.
            n_hot = max(1, int(nb * args.planted))
            hot = torch.randperm(nb, device=dev)[:n_hot]
            view = k_f.view(nb, NPG, H, D)
            view[hot] *= 20.0
        kv = encode_nvfp4(k_f.half(), v_f.half(), H, NPG)

        k, v = dequant_page_tokens(kv)          # [ntok, H, D]
        ntok = k.shape[0]
        # exact per-page score: max over the page of sum_h q_h . k  (upper bound)
        qh = q[0].view(H, G, D).mean(dim=1)     # [H, D]
        kk = k.view(nb, NPG, H, D)
        sc = torch.einsum("pnhd,hd->pnh", kk.float(), qh.float())
        page_score = sc.amax(dim=(1, 2))   # [nb] max over (token, head)
        # exact attention mass per page (for the quality proxy)
        logits = torch.einsum("thd,hd->th", k.float(), qh.float())
        p = torch.softmax(logits, dim=0)                          # [ntok, H]
        mass = p.view(nb, NPG, H).sum(dim=1)                      # [nb, H]

        def run_full():
            mod.run(q, kv, bt_full, kl_full, ws, out, 1.0, 1.0, D ** -0.5,
                    NPARTS, -1, 1, None, None, 0, 0, NPG, 0, 1)

        base = bench(run_full, args.iters)
        print(f"[{tag}] full  nb={nb:6d}  {base:6.3f} ms/layer  "
              f"{kv_len * H * D * 2 * 0.5625 / base / 1e6:6.1f} GB/s", flush=True)

        clean = kv.clone()
        for budget in [float(x) for x in args.budgets.split(",")]:
            keep_n = max(1, int(round(nb * budget)))
            # forced set: first 8 pages (sink) + last 8 pages (recent)
            forced = torch.cat([torch.arange(0, 8, device=dev),
                                torch.arange(nb - 8, nb, device=dev)])
            sel = page_score.topk(min(keep_n, nb)).indices
            idx = torch.unique(torch.cat([sel, forced]))
            idx, _ = torch.sort(idx)
            # compacted table: point the kernel at the selected pages
            bt_sel = torch.zeros(1, max(idx.numel(), 1), dtype=torch.int32, device=dev)
            bt_sel[0, : idx.numel()] = idx.to(torch.int32)
            kl_sel = torch.tensor([idx.numel() * NPG], dtype=torch.int32, device=dev)
            got = mass[idx].sum().item() / mass.sum().item()

            def run_sel():
                mod.run(q, clean, bt_sel, kl_sel, ws, out, 1.0, 1.0, D ** -0.5,
                        NPARTS, -1, 1, None, None, 0, 0, NPG, 0, 1)

            ms = bench(run_sel, args.iters)
            print(f"[{tag}] keep={budget*100:5.1f}%  pages={idx.numel():6d}  "
                  f"{ms:6.3f} ms/layer  mass={got*100:5.1f}%  "
                  f"speedup={base/ms:4.2f}x", flush=True)
        del kv, clean, k, v, mass, page_score
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
