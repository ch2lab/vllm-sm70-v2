#!/usr/bin/env python3
"""P3 GDN shadow-kernel parity test (TDD RED for the chunked SM70 kernel).

Ground truth = FP64 token loop of the CURRENT in-service kernel semantics
(gdn_sm70_fwd.cu L291-372: pred = S@k; d = b*(v - g*pred);
S = g*S + d*k^T; o = scale*(S_new @ q); gate = exp(raw)).
NOT the FLA Triton math (algorithm-equivalent only, see
/data/src/sm70_docs/p3_gdn_math_sheet.md).

Checks (all on the same fp16 inputs; reference runs in fp64 from them):
  A. current kernel vs FP64 ref (validates the ref itself)
  B. shadow (chunked) kernel vs FP64 ref
  C. shadow vs current
  D. final-state parity (shadow / current vs ref)
Shapes: Hq=2 Hv=6 K=V=128 (TP8 per-rank); T in {256, 100(partial chunk), 8192(prod)};
h0 in {zero, nonzero}@T=256.

RED contract: the shadow entry
  vllm.model_executor.layers.mamba.gdn.sm70_gdn.chunk_gated_delta_rule_sm70_chunked
does not exist yet -> probe fails first, before any CUDA work, so this file is
safe to run while the machine is busy (e.g. during the G0 nsys window).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

HQ, HV, K, V = 2, 6, 128, 128
T_LIST = [256, 100, 8192]
RTOL, ATOL = 5e-3, 2e-3


def l2norm_last(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def make_inputs(T: int, seed: int, h0: torch.Tensor | None, dev: str):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = l2norm_last(torch.randn(T, HQ, K, generator=g) * 0.5).to(dev, torch.half)
    k = l2norm_last(torch.randn(T, HQ, K, generator=g) * 0.5).to(dev, torch.half)
    v = (torch.randn(T, HV, V, generator=g) * 0.1).to(dev, torch.half)
    # Production gate form (gdn_sm70_fwd.cu:720): g = -exp(A_log)*softplus(x),
    # per-token per-head; A_log=-0.8 -> mean ~ -0.27, weak corner ~ -0.05.
    x = torch.randn(T, HV, generator=g) + 0.3
    gl = (-0.4493 * (x + torch.log1p(torch.exp(x)))).to(dev, torch.float32)
    beta = (0.2 + 0.6 * torch.rand(T, HV, generator=g)).to(dev, torch.float32)
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    gl = gl.unsqueeze(0)
    beta = beta.unsqueeze(0)
    return q, k, v, gl, beta, h0


def gdn_ref_fp64(q, k, v, gl, beta, h0=None):
    """FP64 token loop on CPU; q/k/v fp16 inputs (any device) upcast; returns
    (o [1,T,HV,V] cpu f64, S [Hv,V,K] cpu f64)."""
    T = q.shape[1]
    scale = K ** -0.5
    qf = q[0].double().cpu()
    kf = k[0].double().cpu()
    vf = v[0].double().cpu()
    gf = torch.exp(gl[0].double().cpu())
    bf = beta[0].double().cpu()
    S = torch.zeros(HV, V, K, dtype=torch.float64)
    if h0 is not None:
        S = h0[0].double().cpu()
    o = torch.empty(1, T, HV, V, dtype=torch.float64)
    for t in range(T):
        for hv in range(HV):
            hq = hv // (HV // HQ)
            pred = S[hv] @ kf[t, hq]
            d = bf[t, hv] * (vf[t, hv] - gf[t, hv] * pred)
            S[hv] = gf[t, hv] * S[hv] + torch.outer(d, kf[t, hq])
            o[0, t, hv] = scale * (S[hv] @ qf[t, hq])
    return o, S


def probe_shadow():
    from vllm.model_executor.layers.mamba.gdn import sm70_gdn

    return getattr(sm70_gdn, "chunk_gated_delta_rule_sm70_chunked", None)


def max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().cpu()
    b = b.double().cpu()
    denom = b.abs().clamp_min(1e-6)
    return float(((a - b).abs() / denom).max().item())


def main() -> int:
    shadow = probe_shadow()
    if shadow is None:
        print("RED (expected): shadow entry "
              "sm70_gdn.chunk_gated_delta_rule_sm70_chunked not implemented yet")
        return 1

    from vllm.model_executor.layers.mamba.gdn.sm70_gdn import (
        chunk_gated_delta_rule_sm70,
    )

    torch.cuda.set_device(0)
    dev = "cuda:0"
    failures = []

    for T in T_LIST:
        for h0tag, make_h0 in [
            ("zero", lambda: None),
            ("nonzero", lambda: torch.randn(1, HV, V, K, device=dev,
                                            dtype=torch.float32) * 0.05),
        ]:
            h0 = make_h0()
            q, k, v, gl, beta, _ = make_inputs(T, seed=T + 7, h0=h0, dev=dev)

            # A. validate the reference against the current kernel
            cur_out, cur_fs = chunk_gated_delta_rule_sm70(
                q, k, v, gl, beta, h0, True)
            ref_out, ref_fs = gdn_ref_fp64(q, k, v, gl, beta, h0)
            a_err = max_rel_err(cur_out, ref_out)
            if not torch.allclose(cur_out.double().cpu(), ref_out,
                                  rtol=RTOL, atol=ATOL):
                failures.append(f"A(T={T},{h0tag}): current vs ref rel={a_err:.2e}")

            # B/C. shadow
            sh_out, sh_fs = shadow(q, k, v, gl, beta, h0, True)
            b_err = max_rel_err(sh_out, ref_out)
            c_err = max_rel_err(sh_out, cur_out)
            if not torch.allclose(sh_out.double().cpu(), ref_out,
                                  rtol=RTOL, atol=ATOL):
                failures.append(f"B(T={T},{h0tag}): shadow vs ref rel={b_err:.2e}")
            if not torch.allclose(sh_out.double().cpu(), cur_out.double().cpu(),
                                  rtol=RTOL, atol=ATOL):
                failures.append(f"C(T={T},{h0tag}): shadow vs current rel={c_err:.2e}")

            # D. final state (reuse A's reference)
            if h0 is not None and ref_fs.max().abs().item() > 1e-6:
                fs_err_c = max_rel_err(cur_fs, ref_fs)
                fs_err_s = max_rel_err(sh_fs, ref_fs)
                if not torch.allclose(cur_fs.double().cpu(), ref_fs,
                                      rtol=RTOL, atol=ATOL):
                    failures.append(f"D-cur(T={T},{h0tag}) rel={fs_err_c:.2e}")
                if not torch.allclose(sh_fs.double().cpu(), ref_fs,
                                      rtol=RTOL, atol=ATOL):
                    failures.append(f"D-sh(T={T},{h0tag}) rel={fs_err_s:.2e}")

            print(f"T={T} h0={h0tag}: A={a_err:.2e} B={b_err:.2e} "
                  f"C={c_err:.2e}")

    # MANDATORY gate since 2026-10-18 — real-gate dynamic range.
    # Production gates are log-gates with g <= 0 and a huge dynamic range
    # (measured on a serving dump: g in [-61.883, -0.000], mean -13.28), so the
    # in-chunk cumulative gcs reaches ~-830 inside a 64-token chunk. The old
    # "INFO-only weak-decay corner" used a *constant* g = -0.05, whose in-chunk
    # cumulative only reaches ~-17, so neither exp() factor overflowed and the
    # K3 factorisation bug (exp(gcs_t)*exp(-gcs_j) -> 0*inf = NaN, 62% of the
    # output poisoned in production) slipped through every green gate. This
    # case must PASS/FAIL, not inform.
    for T in (256, 1024, 4096):
        q, k, v, _, beta, _ = make_inputs(T, seed=T + 31, h0=None, dev=dev)
        ggen = torch.Generator(device=dev).manual_seed(T)
        gl = (torch.rand(1, T, HV, device=dev, generator=ggen) * -62.0).float()
        cur_out, _ = chunk_gated_delta_rule_sm70(q, k, v, gl, beta, None, True)
        sh_out, _ = shadow(q, k, v, gl, beta, None, True)
        finite = bool(torch.isfinite(sh_out).all().item())
        close = bool(torch.allclose(sh_out.double().cpu(), cur_out.double().cpu(),
                                    rtol=RTOL, atol=ATOL))
        err = max_rel_err(sh_out, cur_out)
        print(f"REAL-G T={T} (g~U(-62,0)): finite={finite} "
              f"allclose={close} rel={err:.2e}")
        if not finite:
            failures.append(f"REAL-G(T={T}): shadow output has NaN/inf")
        elif not close:
            failures.append(f"REAL-G(T={T}): shadow vs current rel={err:.2e}")

    # INFO-only stress corner: constant weak decay g=-0.05 (WY inverse
    # amplification bound; documented in spec — superseded by REAL-G above).
    T = 1024
    q, k, v, _, beta, _ = make_inputs(T, seed=T + 99, h0=None, dev=dev)
    gl = (-0.05 * torch.ones(1, T, HV, device=dev, dtype=torch.float32))
    cur_out, _ = chunk_gated_delta_rule_sm70(q, k, v, gl, beta, None, True)
    sh_out, _ = shadow(q, k, v, gl, beta, None, True)
    nref = cur_out.cpu().double().norm().item()
    stress = (sh_out.cpu().double() - cur_out.cpu().double()).norm().item() / max(nref, 1e-12)
    print(f"STRESS weak-decay g=-0.05 T={T}: sh-vs-cur norm-rel={stress:.3e} "
          f"(WY amplification corner; informational)")

    if failures:
        print("FAIL:")
        for f in failures:
            print("  " + f)
        return 2
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
