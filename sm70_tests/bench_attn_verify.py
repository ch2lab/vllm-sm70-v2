"""Standalone microbenchmark of the SM70 verify attention (fp8 KV, nqt=4).

The decode attribution from nsys is unusable, but it named
`BatchPrefillWithPagedKVCacheKernel` at 12.8 ms/call.  This measures the
kernel the serving path actually uses for spec verification
(`paged_decode_fp8`, the XQA fp8 variant with nqt > 1) at the 27B TP4 shapes,
so its cost is known without any profiler.

27B TP4 per rank: H = 1 kv head, HQ = 6 q heads, D = 256, fp8 KV,
page = 1584 tokens, kv_len ~= 8000.

Floor: one pass over the request's KV -> kv_len * H * 2D bytes.

Usage: CUDA_VISIBLE_DEVICES=1 python sm70_tests/bench_attn_verify.py
"""

import time

import torch

from vllm.v1.attention.ops.sm70_nvfp4_decode_cuda import paged_decode_fp8

HBM_GBS = 900.0
H, HQ, D, PAGE = 1, 6, 256, 1584


def bench(kv_len: int, nqt: int, splits: int = 10, iters: int = 100):
    dev = "cuda"
    n_blocks = (kv_len + PAGE - 1) // PAGE + 1
    kv = torch.randint(0, 256, (n_blocks, H, PAGE, 2 * D),
                       dtype=torch.uint8, device=dev)
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).unsqueeze(0)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=dev)
    q = torch.randn(nqt, HQ, D, dtype=torch.float16, device=dev) * 0.1
    out = torch.empty(nqt, HQ, D, dtype=torch.float16, device=dev)
    rp = 1 << (HQ // H - 1).bit_length()
    ws = torch.empty(splits * rp * (D + 2), dtype=torch.float32, device=dev)

    def call():
        paged_decode_fp8(q, kv, bt, kv_lens, out, 1.0, 1.0, D ** -0.5,
                         -1, splits, ws, kv_dtype="fp8_e4m3", nqt=nqt)

    for _ in range(10):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters
    kv_bytes = kv_len * H * 2 * D
    return ms, kv_bytes / ms / 1e6


def main() -> None:
    print(f"verify attention (fp8 XQA) | H={H} HQ={HQ} D={D} page={PAGE} | HBM {HBM_GBS:.0f} GB/s")
    print(f"{'kv_len':>7} {'nqt':>4} {'ms':>8} {'GB/s':>9} {'%floor':>7} {'x16 layers':>11}")
    for kv_len in (2048, 8000, 32000):
        for nqt in (4,):
            ms, gbs = bench(kv_len, nqt)
            print(f"{kv_len:>7} {nqt:>4} {ms:8.4f} {gbs:9.1f} "
                  f"{gbs/HBM_GBS*100:6.1f}% {ms*16:9.2f} ms", flush=True)


if __name__ == "__main__":
    main()
