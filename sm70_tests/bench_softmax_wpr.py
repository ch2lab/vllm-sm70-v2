"""Same-process interleaved timing of fused_softmax_reg across WPR.

The register table says BN=8192 should prefer WPR=4 (REG:149, 3 CTAs/SM) over
the incumbent WPR=1 (REG:255, 2 CTAs/SM) and over WPR=2 (REG:255 + STACK:88).
This checks the direction offline with the pairing this campaign requires; the
in-service kernel us/call remains the actual judge.

Every BN processes the same total element count (nrows*BN per launch times
launches proportional to 1/BN), so at fixed BN the WPR comparison is pure.
"""
import torch

from vllm.v1.attention.ops.sm70_nvfp4_prefill_twopass import _mod

mod = _mod("nvfp4")
dev = "cuda"


def timed(nrows, bn, wpr, reps=6):
    g = torch.Generator(device=dev).manual_seed(11)
    S = (torch.randn(nrows, bn, generator=g, device=dev) * 4.0).half()
    mrow = (torch.randn(nrows, generator=g, device=dev) * 2.0).float()
    lrow = torch.rand(nrows, generator=g, device=dev) * 50 + 1
    alpha = torch.ones(nrows, device=dev)
    qpos = torch.full((nrows,), bn + 4096, dtype=torch.int32, device=dev)
    kv_len = bn + 8192
    for _ in range(2):
        mod.bench_softmax_wpr(S, mrow, lrow, alpha, qpos, 4096, kv_len, 0.0625, wpr)
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        mod.bench_softmax_wpr(S, mrow, lrow, alpha, qpos, 4096, kv_len, 0.0625, wpr)
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3)
    del S, mrow, lrow, alpha, qpos
    torch.cuda.empty_cache()
    return best


print("# one launch of fused_softmax_reg, interleaved min-of-6, same process")
for bn, wprs in ((8192, (1, 2, 4)), (4096, (1, 2, 4))):
    nrows = 32768
    res = {w: [] for w in wprs}
    for r in range(3):
        for w in (wprs if r % 2 == 0 else wprs[::-1]):
            res[w].append(timed(nrows, bn, w))
    base = min(res[1]) if 1 in res else min(res.values())
    print(f"  BN={bn} nrows={nrows}  (S = {nrows*bn*2/2**20:.0f} MiB, "
          f"{nrows*bn*2*2/2**20:.0f} MiB of traffic per launch)")
    for w in wprs:
        t = min(res[w])
        gbs = nrows * bn * 2 * 2 / (t * 1e-6) / 1e9
        print(f"    wpr={w}: {t:8.1f} us   {gbs:6.0f} GB/s ({100*gbs/900:3.0f}% HBM)"
              f"   vs wpr=1 {100*(t/base-1):+6.1f}%   spread "
              f"{100*(max(res[w])/t-1):.1f}%")
