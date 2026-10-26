#!/usr/bin/env python3
"""Bit-exactness gate for candidate H (VLLM_SM70_QSA_POS_FUSE).

The fused kernel must reproduce the torch chain exactly, for both the decode
shapes (qlen == 1 per request, the production decode step) and the chunked
prefill shapes (ragged qlens up to max-num-batched-tokens = 8192), because the
captured graph bakes one of them.

    local = arange(n) - qsl[token_to_req]
    qp    = (seq_lens[token_to_req] - (qsl[1:] - qsl[:-1])[token_to_req]
             + local).to(int64)
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/data/src/vllm-v2")

from vllm.models.qwen4_exp.nvidia.ops.qsa_pos_fuse import (  # noqa: E402
    qsa_positions_fused,
)


def torch_chain(qsl, seq_lens, token_to_req, num_tokens):
    local = torch.arange(num_tokens, device=qsl.device) - qsl[token_to_req]
    return (
        seq_lens[token_to_req] - (qsl[1:] - qsl[:-1])[token_to_req] + local
    ).to(torch.int64)


def make_case(qlens, seq_lens, device="cuda"):
    qlens_t = torch.tensor(qlens, dtype=torch.int32, device=device)
    qsl = torch.zeros(len(qlens) + 1, dtype=torch.int32, device=device)
    qsl[1:] = torch.cumsum(qlens_t, dim=0)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    num_tokens = int(sum(qlens))
    token_to_req = torch.repeat_interleave(
        torch.arange(len(qlens), dtype=torch.int32, device=device), qlens_t
    )
    return qsl, seq_lens_t, token_to_req, num_tokens, qlens_t


CASES = {
    # production decode step: one token per request, distinct long contexts
    "decode_uniform_8": ([1] * 8, [65536, 32768, 4096, 1024, 2, 1, 262144, 17]),
    "decode_uniform_64": ([1] * 64, list(range(1, 65))),
    "decode_single": ([1], [65536]),
    # chunked prefill tails (ragged qlens, n <= max_num_batched_tokens)
    "prefill_ragged": ([1024, 2048, 512, 4096], [65536, 40960, 8192, 1024]),
    "prefill_single_8192": ([8192], [65536]),
    "prefill_mixed_tiny": ([1, 1, 8190], [70000, 4096, 99999]),
    "prefill_two": ([4096, 4096], [8192, 262144]),
}


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    ok = True
    for name, (qlens, seq_lens) in CASES.items():
        qsl, sl, t2r, n, _ = make_case(qlens, seq_lens)
        ref = torch_chain(qsl, sl, t2r, n)
        got = qsa_positions_fused(t2r, qsl, sl, n)
        same_dtype = got.dtype == ref.dtype == torch.int64
        same_shape = got.shape == ref.shape
        equal = bool(torch.equal(got, ref))
        bad = 0 if equal else int((got != ref).sum())
        status = "PASS" if (equal and same_dtype and same_shape) else "FAIL"
        ok &= status == "PASS"
        print(
            f"{status} {name:<22} n={n:<6} reqs={len(qlens):<3} "
            f"dtype={got.dtype}/{ref.dtype} shape={tuple(got.shape)} "
            f"mismatch={bad}"
            + ("" if equal else f"  first_ref={ref[:4].tolist()} "
                                f"first_got={got[:4].tolist()}")
        )
    print("VERDICT:", "BIT-EXACT-OK" if ok else "BIT-EXACT-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
