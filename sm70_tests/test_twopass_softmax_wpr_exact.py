"""Equivalence test for the WPR (warps-per-row) split of fused_softmax_reg.

BN=8192 made the single-warp row spill (~256 registers for r[]), costing
+3.8 s of in-service time over BN=4096 despite processing the same number of
elements.  The fix splits a row over WPR warps so NH stays at the un-spilled
level, which moves the max/sum reduction across a shared-memory combine.  That
combine is new code, so it must be checked against (a) an fp32 oracle and (b)
the single-warp path on BN values where both are legal.

Rows are chosen not to divide by ROWS so the clamped (non-early-returning)
warps are exercised, and lim is varied so some rows are fully live, some
partially masked, and some empty.
"""
import torch

from vllm.v1.attention.ops.sm70_nvfp4_prefill_twopass import _mod

mod = _mod("nvfp4")
dev = "cuda"


def reference(S, mrow0, lrow0, qpos, k0, kv_len, sm):
    """One streaming-softmax chunk, in fp32, matching the kernel's semantics."""
    n, bn = S.shape
    x = S.float() * sm
    lim = (torch.minimum(qpos + 1, torch.tensor(kv_len, device=dev)) - k0).clamp(min=0)
    cols = torch.arange(bn, device=dev).unsqueeze(0)
    live = cols < lim.unsqueeze(1)
    xm = torch.where(live, x, torch.full_like(x, float("-inf")))
    mx = xm.max(dim=1).values
    mx = torch.where(torch.isinf(mx), torch.full_like(mx, -1e30), mx)
    newm = torch.maximum(mrow0, mx)
    a = torch.exp(mrow0 - newm)
    e = torch.where(live, torch.exp(xm - newm.unsqueeze(1)), torch.zeros_like(x))
    e = torch.nan_to_num(e, nan=0.0, neginf=0.0, posinf=0.0)
    return e.half(), newm, lrow0 * a + e.sum(dim=1), a


def run(S0, mrow0, lrow0, qpos, k0, kv_len, sm, wpr):
    S = S0.clone()
    mrow = mrow0.clone(); lrow = lrow0.clone(); alpha = torch.ones_like(mrow)
    mod.bench_softmax_wpr(S, mrow, lrow, alpha, qpos, k0, kv_len, sm, wpr)
    torch.cuda.synchronize()
    return S, mrow, lrow, alpha


def check(bn, nrows, wpr):
    g = torch.Generator(device=dev).manual_seed(7 * bn + nrows + wpr)
    S0 = (torch.randn(nrows, bn, generator=g, device=dev, dtype=torch.float32)
          * 4.0).half()
    mrow0 = (torch.randn(nrows, generator=g, device=dev) * 2.0).float()
    lrow0 = torch.rand(nrows, generator=g, device=dev) * 50.0 + 1.0
    # vary lim: fully live, partially masked, and fully empty rows
    qpos = torch.empty(nrows, dtype=torch.int32, device=dev)
    k0, kv_len = 4096, 12000
    qpos[: nrows // 3] = k0 + bn + 5                 # fully live
    qpos[nrows // 3: 2 * (nrows // 3)] = k0 + bn // 2  # half masked
    qpos[2 * (nrows // 3):] = k0 - 3                  # lim <= 0
    sm = 0.0625

    eP, em, el, ea = reference(S0, mrow0, lrow0, qpos, k0, kv_len, sm)
    P, mrow, lrow, alpha = run(S0, mrow0, lrow0, qpos, k0, kv_len, sm, wpr)
    perr = (P.float() - eP.float()).abs().max().item()
    merr = (mrow - em).abs().max().item()
    lerr = ((lrow - el).abs() / el.abs().clamp(min=1e-3)).max().item()
    aerr = (alpha - ea).abs().max().item()

    # single-warp path is the incumbent: the split must match it closely
    P1, m1, l1, a1 = run(S0, mrow0, lrow0, qpos, k0, kv_len, sm, 1)
    dp = (P.float() - P1.float()).abs().max().item()
    dl = ((lrow - l1).abs() / l1.abs().clamp(min=1e-3)).max().item()
    ok = perr < 6e-4 and merr < 1e-5 and lerr < 1e-4 and aerr < 1e-5 and dl < 1e-4
    print(f"  BN={bn:5} nrows={nrows:4} wpr={wpr}: vs oracle P {perr:.2e} "
          f"m {merr:.1e} l {lerr:.2e} a {aerr:.1e} | vs wpr=1 P {dp:.2e} l {dl:.2e}"
          f"  {'ok' if ok else '*** FAIL ***'}")
    return ok


allok = True
for bn in (2048, 4096, 8192):
    for nrows in (5, 7, 13, 33, 64, 129):
        for wpr in (1, 2, 4):
            allok &= check(bn, nrows, wpr)
print("VERDICT:", "PASS" if allok else "FAIL")
