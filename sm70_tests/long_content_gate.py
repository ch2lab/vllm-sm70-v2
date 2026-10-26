#!/usr/bin/env python3
"""Long-context content gate — the T >= 4096 envelope.

`correctness.py`'s prompts are all <= 2k tokens, so it cannot see a candidate
that only activates at larger T. That is exactly how the chunked-GDN NaN
reached production (2026-10-18): the shadow ran for the >= 4096-token prefill
ubatches (production sizes observed: 3136 ... 7840) and emitted "ductduct...",
while every short-prompt check passed.

A needle is placed at the head of a very long prompt and the model must return
it, so the whole context has to be processed correctly.

Usage:
  python sm70_tests/long_content_gate.py [--tokens 58282] [--needle 391]
Exit 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

FILLER = "The quick brown fox jumps over the lazy dog. "


def build(target_tokens: int, needle: int) -> str:
    # ~4 chars/token for this filler; the exact token count is verified from
    # usage.prompt_tokens after the request.
    return (
        f"记住这个数字：{needle}。\n"
        + FILLER * (target_tokens * 4 // len(FILLER) + 1)
        + "\n问题：我一开始让你记住的数字是多少？只输出数字。"
    )


def ask(prompt: str, max_tokens: int = 16) -> tuple[str, dict, dict]:
    body = {
        "model": common.model_name(),
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "thinking_token_budget": 0,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8001/v1/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + common.api_key(),
        },
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["text"], d.get("usage") or {}, d.get("metrics") or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=58282)
    ap.add_argument("--needle", type=int, default=391)
    args = ap.parse_args()

    text, usage, metrics = ask(build(args.tokens, args.needle))
    pt = usage.get("prompt_tokens")
    ok = str(args.needle) in text
    ttft = metrics.get("time_to_first_token_ms")
    tag = "PASS" if ok else "FAIL"
    print(
        f"[{tag}] prompt_tokens={pt} (target {args.tokens}) "
        f"ttft_ms={round(ttft, 1) if ttft else None} out={text[:40]!r}"
    )
    if pt is not None and pt < 4096:
        print(
            "WARNING: prompt_tokens < 4096 — this gate is meant to exercise "
            "the T >= 4096 envelope"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
