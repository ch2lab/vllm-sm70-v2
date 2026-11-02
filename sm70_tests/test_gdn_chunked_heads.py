#!/usr/bin/env python3
"""GDN chunked kernel parity across head geometries (parameterized).

Compares the in-service serial kernel with the chunked (K1/K2/K3) kernel for
the geometries we actually serve:

  * Qwen3.8-Flash-Next TP8 -> per-rank Hq=2, Hv=6   (the reference deployment)
  * Qwen3.6-35B TP2        -> per-rank Hq=8, Hv=16
  * Qwen3.6-35B TP1        -> per-rank Hq=16, Hv=32

Metrics (fp64 reference computed with the same per-request geometry):
  rel_l2 = ||out - ref|| / ||ref||          (global; the one that matters)
  maxrel = max |out - ref| / max |ref|
  same for the final state.
"""

from __future__ import annotations

import sys

import torch

K = V = 128
RTOL_L2 = 5e-3


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def make_inputs(T: int, HQ: int, HV: int, seed: int, dev: str):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = l2norm(torch.randn(T, HQ, K, generator=g) * 0.5).to(dev, torch.half)
    k = l2norm(torch.randn(T, HQ, K, generator=g) * 0.5).to(dev, torch.half)
    v = (torch.randn(T, HV, V, generator=g) * 0.1).to(dev, torch.half)
    x = torch.randn(T, HV, generator=g) + 0.3
    gl = (-0.4493 * (x + torch.log1p(torch.exp(x)))).to(dev, torch.float32)
    beta = (0.2 + 0.6 * torch.rand(T, HV, generator=g)).to(dev, torch.float32)
    return (
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        gl.unsqueeze(0),
        beta.unsqueeze(0),
    )


def ref_fp64(q, k, v, gl, beta, HQ: int, HV: int):
    """Token-loop reference matching gdn_sm70_fwd.cu semantics."""
    T = q.shape[1]
    scale = K ** -0.5
    qf = q[0].double().cpu()
    kf = k[0].double().cpu()
    vf = v[0].double().cpu()
    gf = torch.exp(gl[0].double().cpu())
    bf = beta[0].double().cpu()
    S = torch.zeros(HV, V, K, dtype=torch.float64)
    o = torch.empty(1, T, HV, V, dtype=torch.float64)
    for t in range(T):
        for hv in range(HV):
            hq = hv // (HV // HQ)
            pred = S[hv] @ kf[t, hq]
            d = bf[t, hv] * (vf[t, hv] - gf[t, hv] * pred)
            S[hv] = gf[t, hv] * S[hv] + torch.outer(d, kf[t, hq])
            o[0, t, hv] = scale * (S[hv] @ qf[t, hq])
    return o, S


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().cpu().reshape(-1)
    b = b.double().cpu().reshape(-1)
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def maxrel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().cpu()
    b = b.double().cpu()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def main() -> int:
    from vllm.model_executor.layers.mamba.gdn.sm70_gdn import (
        chunk_gated_delta_rule_sm70,
        chunk_gated_delta_rule_sm70_chunked,
    )

    torch.cuda.set_device(0)
    dev = "cuda:0"
    configs = [
        (2, 6, "Qwen3.8 TP8", (1024, 2048, 4096, 8192)),
        (8, 16, "Qwen3.6 TP2", (1024, 2048, 4096)),
        (16, 32, "Qwen3.6 TP1", (1024, 2048, 4096)),
    ]
    failures = []
    for HQ, HV, label, ts in configs:
        for T in ts:
            q, k, v, gl, beta = make_inputs(T, HQ, HV, seed=T + HQ, dev=dev)
            ser, ser_fs = chunk_gated_delta_rule_sm70(q, k, v, gl, beta, None, True)
            chk, chk_fs = chunk_gated_delta_rule_sm70_chunked(
                q, k, v, gl, beta, None, True
            )
            r_l2 = rel_l2(chk, ser)
            r_max = maxrel(chk, ser)
            ok = r_l2 <= RTOL_L2
            extra = ""
            if T <= 2048:  # fp64 reference is O(T*Hv) python matmuls
                ref, _ = ref_fp64(q, k, v, gl, beta, HQ, HV)
                s_l2 = rel_l2(ser, ref)
                c_l2 = rel_l2(chk, ref)
                extra = f" | serial-ref {s_l2:.2e} chunked-ref {c_l2:.2e}"
                ok = ok and s_l2 <= RTOL_L2 and c_l2 <= RTOL_L2
            print(
                f"{label:12s} HQ={HQ:2d} HV={HV:2d} T={T:5d} "
                f"chunked-vs-serial rel_l2={r_l2:.2e} maxrel={r_max:.2e} "
                f"{'PASS' if ok else 'FAIL'}{extra}",
                flush=True,
            )
            if not ok:
                failures.append(f"{label} HQ={HQ} HV={HV} T={T}: rel_l2={r_l2:.2e}")

    print("\nRESULT:", "ALL PASS" if not failures else f"FAIL ({len(failures)})")
    for f in failures:
        print("  -", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
