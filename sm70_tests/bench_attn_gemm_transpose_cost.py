"""Is the QK^T "contiguous B" gain real, or an artefact of hoisting the transpose?

The orientation screen showed QK^T at 63.7 TF/s vs 66.3 TF/s if B is materialised
contiguous first (+4.1%).  But that measurement produced Kc = K.t().contiguous()
ONCE, outside the timed loop.  In the served pipeline that is not available:
bridge_kv writes Kb as [BN,256] fresh for every k-block, so any contiguous-B gemm
must pay for the transpose every iteration.  This measures the honest totals.

Also localises the real limit, since the two sweeps disagree with my prior:
  PV (N=256) runs at 83.5 TF/s = 85% of the box ceiling -> narrow-N is NOT the
  problem.  QK^T is K-bound: K=256 -> 63 TF, K=512 -> 73, K=1024 -> 80.  K is
  head_dim, fixed at 256 by the model, so most of the "70-72 TF vs 98.5 TF" gap is
  inherent to d=256 attention on Volta rather than a defect left unfixed.
"""
import torch

dev = "cuda"
M, BN, D = 32768, 8192, 256
PEAK = 98.5
FL = 2 * M * BN * D


def timed(fn, reps=9):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    v = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        v.append(a.elapsed_time(b))
    v.sort()
    return v[0], v[len(v) // 2]


def line(name, ms):
    print(f"  {name:38} {ms:8.2f} ms   {FL/(ms*1e-3)/1e12:6.1f} TF/s"
          f"  = 峰值 {100*FL/(ms*1e-3)/1e12/PEAK:5.1f}%")


Q = torch.randn(M, D, device=dev, dtype=torch.float16) * 0.1
Kb = torch.randn(BN, D, device=dev, dtype=torch.float16) * 0.1
S = torch.randn(M, BN, device=dev, dtype=torch.float16) * 0.1
Vb = torch.randn(BN, D, device=dev, dtype=torch.float16) * 0.1
C = torch.empty(M, BN, device=dev, dtype=torch.float16)
Kc = torch.empty(D, BN, device=dev, dtype=torch.float16)
O = torch.empty(M, D, device=dev, dtype=torch.float16)
Ot = torch.empty(D, M, device=dev, dtype=torch.float16)
St = torch.empty(BN, M, device=dev, dtype=torch.float16)
Vt = torch.empty(D, BN, device=dev, dtype=torch.float16)

print("=== QK^T：把转置算进成本（bridge_kv 每个 k 块都要重写 Kb）===")
best, med = timed(lambda: torch.mm(Q, Kb.t(), out=C))
line("现状  mm(Q, Kb.t())  [视图转置]", med)
best2, med2 = timed(lambda: torch.mm(Q, Kc, out=C))
line("上轮那个（Kc 预先烘好，不合法）", med2)


def with_transpose():
    torch.transpose(Kb, 0, 1).contiguous(out=None) if False else Kc.copy_(Kb.t())
    torch.mm(Q, Kc, out=C)


best3, med3 = timed(with_transpose)
line("诚实版 Kc.copy_(Kb.t()) 然后 mm", med3)
print(f"  -> 转置本身花了 {med3-med2:.2f} ms；相对现状 "
      f"{100*(med3/med-1):+.1f}%  {'值得' if med3 < med else '不值得'}")

print("\n=== PV：同样把转置算进成本 ===")
best, med = timed(lambda: torch.mm(S, Vb, out=O))
line("现状  mm(S, Vb)", med)


def pv_transposed():
    Vt.copy_(Vb.t())
    St.copy_(S.t())
    torch.mm(Vt, St, out=Ot)


best2, med2 = timed(pv_transposed)
line("转置版 Vt/Sr 各拷一次 然后 mm", med2)
print(f"  -> 相对现状 {100*(med2/med-1):+.1f}%  {'值得' if med2 < med else '不值得'}"
      f"（注意 S^T 是 512 MB 的转置拷贝，几乎肯定亏）")

print("\n=== 上限估计：若 softmax 直接把 S 写成转置布局，能否白拿 PV 的 +4.8% ===")
print("  PV 转置版要求 S 以 [BN,M] 存放。让 fused_softmax_reg 就地转置写回可省掉")
print("  上面那次 512 MB 拷贝，但 softmax 已在 88% HBM，连续写会变跨步写。")
St2 = torch.empty(BN, M, device=dev, dtype=torch.float16)
m_cont = timed(lambda: St2.copy_(St2))[1]          # same-bytes reference point
m_strided = timed(lambda: St2.copy_(S.t()))[1]     # [M,BN] -> [BN,M] strided write
print(f"  同字节数的连续拷贝 {m_cont:.2f} ms vs 转置拷贝 {m_strided:.2f} ms"
      f"  -> 转置写代价 {100*(m_strided/max(m_cont,1e-6)-1):+.0f}%")
print(f"  PV 从 81.0 提升到约 87.5 TF/s 只省 {33.0*(1-81.0/87.5):.1f} s，"
      f"而 softmax(28.2 s) 只要退化 >{100*33.0*(1-81.0/87.5)/28.2:.0f}% 就净亏")
print("  => 判定：不划算，PV 转置路线关闭（除非有免拷贝的转置布局来源）")
