#!/usr/bin/env python3
"""Dump the FP8 QPN GEMM's output so two warp-count arms can be compared.

Widening the CTA from 16 to 32 warps changes how many K groups each warp sums
(k/16 -> k/32) and how many partials the final shared reduction adds (16 -> 32),
so the fp32 accumulation is **reassociated**: this class of change cannot be
validated by bit identity, only by bounding the difference and passing the
service quality gates.  Each arm must run in its own process because
VLLM_SM70_FP8_QPN_WARPS is read once into a static.

    VLLM_SM70_FP8_QPN_WARPS=16 python sm70_tests/test_fp8_qpn_warps_exact.py --dump /tmp/a
    VLLM_SM70_FP8_QPN_WARPS=32 python sm70_tests/test_fp8_qpn_warps_exact.py --dump /tmp/b
    python sm70_tests/test_moe_nacc_exact.py /tmp/a.o_proj.bin /tmp/b.o_proj.bin
"""
from __future__ import annotations

import argparse

import torch

# the four production decode shapes, as in bench_fp8_qpn.py
SHAPES = [
    ("gdn_in_proj", 12288, 2048),
    ("attn_qkv", 9216, 2048),
    ("o_proj", 2048, 4096),
    ("lm_head", 248320, 2048),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="")
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--seed", type=int, default=9502)
    ap.add_argument("--split-k", type=int, default=16)
    ap.add_argument("--only", default="",
                    help="comma-separated shape names (default: all)")
    args = ap.parse_args()

    import vllm  # noqa: F401
    import vllm._custom_ops  # noqa: F401

    dev = "cuda"
    wanted = set(x for x in args.only.split(",") if x)
    for name, n, k in SHAPES:
        if wanted and name not in wanted:
            continue
        torch.manual_seed(args.seed)
        # valid e4m3 codes only: 0x7F is NaN, so quantise real values
        w = torch.randn(k, n, device=dev, dtype=torch.float32).clamp_(-4.0, 4.0)
        w8 = w.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        packed = torch.ops._C.sm70_fp8_qpn_pack(w8)
        scale = torch.ones(1, dtype=torch.float32, device=dev)
        a = torch.randn(args.m, k, dtype=torch.float16, device=dev)
        out = torch.empty(args.m, n, dtype=torch.float16, device=dev)
        torch.ops._C.sm70_fp8_qpn_dispatch(out, a, packed, scale, args.split_k)
        torch.cuda.synchronize()
        if args.dump:
            p = f"{args.dump}.m{args.m}s{args.seed}.{name}.bin"
            with open(p, "wb") as fh:
                fh.write(out.detach().cpu().numpy().tobytes())
            print(f"dumped {p} ({out.numel()*2} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
