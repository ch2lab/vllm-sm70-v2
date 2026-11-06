"""Standalone microbenchmark of the SM70 GDN decode kernel.

The decode attribution could not be trusted from nsys windows, but the kernel
list still named `gdn_forward_vlk_varlen` as the third-largest non-GEMM item
(192 calls x 2.1 ms).  This measures the kernel directly at the 27B TP4
shapes so its cost is known without any profiler windowing.

27B (Qwen3.8-27B-QUASAR-NVFP4) linear-attention config, per TP4 rank:
  Hq = 16/4 = 4, Hv = 48/4 = 12, K = V = 128, 48 linear layers
  state layout [N, Hv, V, K] fp32

Floor: the state is read and written once per call ->
  2 * N * Hv * V * K * 4 bytes at 900 GB/s.

Usage: CUDA_VISIBLE_DEVICES=1 python sm70_tests/bench_gdn_decode.py
"""

import time

import torch

from vllm.model_executor.layers.mamba.gdn.sm70_gdn import _load_ext

HBM_GBS = 900.0
HQ, HV, K, V = 4, 12, 128, 128


def bench(n_req: int, nqt: int, iters: int = 50):
    ext = _load_ext()
    dev = "cuda"
    T = n_req * nqt
    q = torch.randn(1, T, HQ, K, dtype=torch.float16, device=dev) * 0.1
    k = torch.randn(1, T, HQ, K, dtype=torch.float16, device=dev) * 0.1
    v = torch.randn(1, T, HV, V, dtype=torch.float16, device=dev) * 0.1
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, T, HV, dtype=torch.float16, device=dev))
    beta = torch.rand(1, T, HV, dtype=torch.float16, device=dev)
    state = torch.randn(n_req, HV, V, K, dtype=torch.float32, device=dev) * 0.01
    cu = torch.arange(0, T + 1, nqt, dtype=torch.int32, device=dev)

    def call():
        ext.gdn_forward_vlk_varlen(q, k, v, g, beta, state, cu,
                                   K ** -0.5, True, False, False, None)

    for _ in range(10):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters
    state_bytes = 2 * n_req * HV * V * K * 4
    gbs = state_bytes / ms / 1e6
    return ms, gbs


def main() -> None:
    print(f"GDN decode kernel | Hq={HQ} Hv={HV} K={K} V={V} | HBM {HBM_GBS:.0f} GB/s")
    print(f"{'reqs':>5} {'nqt':>4} {'T':>4} {'ms':>8} {'GB/s':>8} {'%floor':>7} {'per-layer x48':>14}")
    for n_req, nqt in ((1, 4), (2, 4), (8, 4), (16, 4), (1, 1), (8, 2)):
        ms, gbs = bench(n_req, nqt)
        print(f"{n_req:>5} {nqt:>4} {n_req*nqt:>4} {ms:8.3f} {gbs:8.1f} "
              f"{gbs/HBM_GBS*100:6.1f}% {ms*48:12.1f} ms", flush=True)


if __name__ == "__main__":
    main()
