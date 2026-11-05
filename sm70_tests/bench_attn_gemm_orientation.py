"""Why do the attention GEMMs sit at 70-72 TF when cublas reaches 98.5-99.5 here?

The two-pass attention bucket is ~78% of prefill and its GEMMs are the single
largest unbroken number in the campaign (77.3 s of 140.4 s GPU-busy). Prior work
swept the M slope (BN/BM) and swapped cublasLt algorithms; nobody checked the
ORIENTATION, which is the thing that decides whether cublas picks a good kernel
for these particular shapes.

The shapes come from the live code, not from a guess:
  QK^T   at::mm_out(S,  Qb, Kb.transpose(0,1))   A=[M,256] x B=[BN,256]^T   -> [M,BN]
  PV     at::mm_out(PV, S,  Vb)                  A=[M,BN]  x B=[BN,256]     -> [M,256]
with M = BM*G = 4096*8 = 32768 (verified from the trace grid (BN/128, M/128, 1)).

Two structural suspicions, both testable here and both exact (a transpose changes
no arithmetic):
  1. QK^T hands cublas a transposed B (a TN gemm). Making B contiguous first may
     pick a better kernel, if the transpose's bandwidth cost is smaller than the
     kernel gain.
  2. PV has N=256, a very narrow output, and K=8192. Narrow-N gemms are the classic
     weak spot. Computing the transposed problem O^T = V^T x S^T turns the wide M
     into the N dimension, which many kernels handle better.

Nothing here is a claim about the served pipeline; it is an offline screen for
whether an exact reformulation buys TF/s. If a winner appears it gets gated
properly (same-period control, in-service us/call, quality gates).
"""
import torch

dev = "cuda"
M, BN, D = 32768, 8192, 256      # live shapes at BN=8192 / BM=4096 / G=8 (TP1)
PEAK = 98.5                      # measured cublas square ceiling on this box


def tf(flops, ms):
    return flops / (ms * 1e-3) / 1e12


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
    return v[0], v[len(v) // 2], 100 * (v[-1] / v[0] - 1)


def report(name, flops, best, med, spread):
    print(f"  {name:34} {best:8.2f} ms (med {med:8.2f}, spread {spread:4.1f}%)"
          f"  {tf(flops,best):6.1f} TF/s  = 峰值 {100*tf(flops,best)/PEAK:4.1f}%")


print(f"device={torch.cuda.get_device_name(0)}  M={M} BN={BN} D={D} fp16")
print(f"参照：本机 cublas 方阵实测上限 {PEAK} TF/s；两遍注意力 GEMM 现值 70-72 TF/s\n")

Q = torch.randn(M, D, device=dev, dtype=torch.float16) * 0.1
K = torch.randn(BN, D, device=dev, dtype=torch.float16) * 0.1
S = torch.randn(M, BN, device=dev, dtype=torch.float16) * 0.1
V = torch.randn(BN, D, device=dev, dtype=torch.float16) * 0.1
C_MN = torch.empty(M, BN, device=dev, dtype=torch.float16)
O_MT = torch.empty(D, M, device=dev, dtype=torch.float16)

flops_qk = 2 * M * BN * D
flops_pv = 2 * M * BN * D

print("=== QK^T: [M,D] x [BN,D]^T -> [M,BN]  （现网是 TN，B 是转置视图）===")
best, med, sp = timed(lambda: torch.mm(Q, K.t(), out=C_MN))
report("现状  mm(Q, K.t())", flops_qk, best, med, sp)
Kc = K.t().contiguous()                      # [D, BN] materialised once
best, med, sp = timed(lambda: torch.mm(Q, Kc, out=C_MN))
report("NN    mm(Q, Kc)  (B 预转置)", flops_qk, best, med, sp)
Qt = Q.t().contiguous()
Ct = torch.empty(BN, M, device=dev, dtype=torch.float16)
best, med, sp = timed(lambda: torch.mm(K, Qt, out=Ct))
report("反向  mm(K, Q^T) -> [BN,M]", flops_qk, best, med, sp)

print("\n=== PV: [M,BN] x [BN,D] -> [M,D]   （N=256 窄输出，怀疑是弱点）===")
O = torch.empty(M, D, device=dev, dtype=torch.float16)
best, med, sp = timed(lambda: torch.mm(S, V, out=O))
report("现状  mm(S, V)", flops_pv, best, med, sp)
Vt = V.t().contiguous()                      # [D, BN]
best, med, sp = timed(lambda: torch.mm(Vt, S.t(), out=O_MT))
report("转置  mm(V^T, S^T) -> [D,M]", flops_pv, best, med, sp)
St = S.t().contiguous()
best, med, sp = timed(lambda: torch.mm(Vt, St, out=O_MT))
report("转置+St预置 mm(V^T, St)", flops_pv, best, med, sp)

print("\n=== 参照：同 K 深度下把 N 放宽，看 N=256 是不是真瓶颈 ===")
for n in (256, 512, 1024, 2048, 4096):
    B2 = torch.randn(BN, n, device=dev, dtype=torch.float16) * 0.1
    C2 = torch.empty(M, n, device=dev, dtype=torch.float16)
    best, med, sp = timed(lambda: torch.mm(S, B2, out=C2), reps=5)
    print(f"  N={n:5}  {best:8.2f} ms  {tf(2*M*BN*n, best):6.1f} TF/s"
          f"  = 峰值 {100*tf(2*M*BN*n,best)/PEAK:4.1f}%")
    del B2, C2
    torch.cuda.empty_cache()

print("\n=== 参照：把 K 减短（QK^T 的 K=256 是否偏短）==="
      )
for kk in (128, 256, 512, 1024):
    A2 = torch.randn(M, kk, device=dev, dtype=torch.float16) * 0.1
    B2 = torch.randn(BN, kk, device=dev, dtype=torch.float16) * 0.1
    C2 = torch.empty(M, BN, device=dev, dtype=torch.float16)
    best, med, sp = timed(lambda: torch.mm(A2, B2.t(), out=C2), reps=5)
    print(f"  K={kk:5}  {best:8.2f} ms  {tf(2*M*BN*kk, best):6.1f} TF/s"
          f"  = 峰值 {100*tf(2*M*BN*kk,best)/PEAK:4.1f}%")
    del A2, B2, C2
    torch.cuda.empty_cache()
