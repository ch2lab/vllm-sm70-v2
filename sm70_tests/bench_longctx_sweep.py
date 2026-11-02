#!/usr/bin/env python3
"""Long-context prefill/decode sweep for the SM70 service.

One request per length, streaming SSE; prefill = time to first token, decode =
steady inter-token interval. A unique salt token at the *front* of every prompt
keeps the prefix cache cold so the numbers are real prefill costs.

Usage:
  python sm70_tests/bench_longctx_sweep.py --base http://127.0.0.1:8001 \
      --lengths 10000,65000,130000,250000 --out /data/sm70_results/sweep.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

MODEL = "/data/models/Qwen3.6-35B-A3B-NVFP4-Fast"
PASSAGE = (
    "人工智能与机器学习在当代科学和工程中的广泛应用正在快速改变各个行业的发展轨迹，"
    "并且推动着技术革命向前发展。大语言模型服务在预填充和解码两个阶段表现出截然不同的"
    "性能特征：预填充阶段需要处理成批的输入 token，解码阶段则逐 token 生成，受显存带宽"
    "与内核启动开销的限制。"
)


def api_key() -> str:
    try:
        return Path("/root/.vllm_api_key").read_text().strip()
    except FileNotFoundError:
        return ""


def build_prompt(tk, length: int) -> list[int]:
    salt = tk.encode(f"SALT-{time.time_ns()} ", add_special_tokens=False)
    ids = list(salt)
    body = tk.encode(PASSAGE * 60, add_special_tokens=False)
    while len(ids) < length:
        ids.extend(body)
    return ids[:length]


def run_one(base: str, key: str, tk, model: str, length: int, out_tokens: int):
    ids = build_prompt(tk, length)
    body = {
        "model": model,
        "prompt": ids,
        "max_tokens": out_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    t0 = time.perf_counter()
    first = last = None
    nchunk = 0
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("text") or ch[0].get("delta")):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                nchunk += 1
    wall = time.perf_counter() - t0
    n_out = (usage or {}).get("completion_tokens") or nchunk
    n_prompt = (usage or {}).get("prompt_tokens") or len(ids)
    prefill_ms = (first - t0) * 1000 if first else float("nan")
    decode_ms = (last - first) * 1000 / max(n_out - 1, 1) if first and last else float("nan")
    return {
        "length": length,
        "prompt_tokens": n_prompt,
        "output_tokens": n_out,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": n_prompt / (prefill_ms / 1000) if prefill_ms == prefill_ms else None,
        "decode_ms_per_tok": decode_ms,
        "decode_tok_s": 1000 / decode_ms if decode_ms == decode_ms else None,
        "wall_s": wall,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="Qwen3.6-35B-A3B")
    ap.add_argument("--lengths", default="10000,65000,130000,250000")
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2048)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    key = api_key()

    if args.warmup:
        run_one(args.base, key, tk, args.model, args.warmup, 8)

    rows = []
    for length in [int(x) for x in args.lengths.split(",") if x.strip()]:
        row = run_one(args.base, key, tk, args.model, length, args.output_tokens)
        rows.append(row)
        print(
            f"len={row['length']:7d} pt={row['prompt_tokens']:7d} "
            f"prefill={row['prefill_ms']:9.1f}ms ({row['prefill_tok_s']:7.1f} tok/s) "
            f"decode={row['decode_ms_per_tok']:6.2f}ms ({row['decode_tok_s']:6.1f} tok/s)",
            flush=True,
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
