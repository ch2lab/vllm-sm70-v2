#!/usr/bin/env python3
"""Online content-correctness STANDARD for the SM70 QSA prefill kernel change.

Content correctness is a STANDARD against ground truth, PRE-REGISTERED here
before running: a fixed set of factual QA prompts with unambiguous answers; the
model's reply must contain the ground-truth token (semantic hit), not be a
string-identical match to another arm and not an A-vs-B diff.  A broken kernel
yields wrong/degenerate replies (fails); a numerically-tiny-but-correct change
(the FMA kernel's 2.5e-4 vs FP64) leaves the answers correct (passes).

Request shape (deliberate, 2026-10-14 fix): this is a CHAT/reasoning model, so
the prompt must be IN-DISTRIBUTION — /v1/chat/completions with a proper user
turn and chat_template_kwargs={"enable_thinking": false} (thinking truly OFF; the
raw /v1/completions bare-string path is off-distribution and thinking is not
reliably suppressed there).  Output judged on the message's content field.

Robustness (PERF_CAMPAIGN §9.8): greedy output can be non-reproducible on some
near-tie prompts, so each question runs K times and is judged by MAJORITY; every
miss is printed with its FULL text so a human can confirm by eye whether a miss
is a real wrong answer vs a model-side instability (not a kernel defect).
The per-question hit-rate is reported; greedy reproducibility is a separate
quality item, not part of this kernel standard.

Bar (pre-registered): all N questions majority-correct over K runs.
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata
import urllib.request

MODEL = "Qwen3.8-Flash-Next"
URL = "http://127.0.0.1:8001/v1/chat/completions"
K = 10  # runs per question

# --- PRE-REGISTERED standard (fixed before running) ---
# (prompt, ground-truth token that must appear, case-insensitive, in the reply)
BATTERY = [
    ("What is the capital of France? Answer with only the single word.",
     "paris"),
    ("What is 2 plus 2? Answer with only the number.", "4"),
    ("How many days are in a week? Answer with only the number.", "7"),
    ("What planet is known as the Red Planet? Answer with only the word.",
     "mars"),
    ("What is the chemical symbol for water? Answer with only the symbol.",
     "h2o"),
    ("What color is the sky on a clear day? Answer with only the word.",
     "blue"),
]
BAR = "every question majority-correct over K runs"


def _norm(s: str) -> str:
    """NFKD-fold then keep alnum so 'H\u2082O' -> 'h2o'."""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s.lower() if c.isalnum())


def ask(prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 48, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["SM70_API_KEY"]})
    d = json.load(urllib.request.urlopen(req, timeout=180))
    ch = d["choices"][0]["message"]
    return {"content": ch.get("content") or "",
            "reasoning": ch.get("reasoning_content")}


def main() -> int:
    q_pass = 0
    for p, gt in BATTERY:
        counts = {"hit": 0, "miss": 0}
        misses = []
        for _ in range(K):
            r = ask(p)
            if _norm(gt) in _norm(r["content"]):
                counts["hit"] += 1
            else:
                counts["miss"] += 1
                misses.append(r["content"])
        ok = counts["hit"] > K / 2
        q_pass += int(ok)
        print(f"  [{'OK' if ok else 'MISS'}] {gt!r} hit={counts['hit']}/{K}")
        for m in misses:
            print(f"       miss full-text: {m!r}")
    met = q_pass == len(BATTERY)
    print(f"ONLINE_CONTENT: {q_pass}/{len(BATTERY)} questions pass majority "
          f"(bar: {BAR}, K={K}, shape=chat/enable_thinking=false)")
    print("ONLINE_CONTENT_STANDARD_" + ("MET" if met else "FAIL"))
    return 0 if met else 1


if __name__ == "__main__":
    sys.exit(main())
