#!/usr/bin/env python3
"""quality_harness.py — 质量守门（阶段 A）：长文检索 / 多位算术 / 指令跟随。

Why: canary(6/6) 与 long_content_gate(→"391") 只能抓"明显坏掉"，抓不到"减 FLOP/字节"
类优化带来的轻度退化（长文检索变差、算术随机错、格式崩）。本工具给优化一个可判的守门：
固定任务集 + 多数表决 + 退化指标（输出长度 / EOS / 5-gram 重复），并落基线 JSON。

用法：
  source /data/src/sm70_env.sh
  python sm70_tests/quality_harness.py [--k 5] [--tag baseline]

判据（预登记）：每题的多数表决通过率 > K/2 才算该题 PASS；全部题目 PASS 才算标准达成。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

MODEL = "Qwen3.8-Flash-Next"
URL = "http://127.0.0.1:8001/v1/chat/completions"

# ---------------------------------------------------------------- long context
FACTS = [
    ("access code for the blue vault", "7412"),
    ("backup server hostname", "ORION-9"),
    ("reserved meeting room", "3B-217"),
]
LONG_Q = ("Using only the document above, answer all three questions on one line "
          "separated by semicolons: (1) What is the access code for the blue vault? "
          "(2) What is the backup server hostname? (3) What is the reserved meeting room?")

ARITH = [
    ("What is 8473 multiplied by 29? Answer with only the number.", "245717"),
    ("What is 123456 plus 789012? Answer with only the number.", "912468"),
    ("What is 9123 minus 4567? Answer with only the number.", "4556"),
]

FORMAT = [
    ("Reply with exactly the single word PARIS and nothing else.", "exact:PARIS"),
    ("List exactly three colors separated by semicolons, lowercase, no other text.",
     "three_semicolon_items"),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s.lower() if c.isalnum())


def build_long_prompt(target_words: int = 22000, seed: int = 7) -> str:
    """~32K-token filler with the three facts planted at 15%/50%/85%."""
    rng = random.Random(seed)
    vocab = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
             "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
             "xray yankee zulu system report segment baseline window metric buffer").split()
    total = target_words
    at = {int(total * f): FACTS[i] for i, f in enumerate((0.15, 0.50, 0.85))}
    out: list[str] = ["# Operations handbook (internal)\n"]
    for i in range(total):
        if i in at:
            name, val = at[i]
            out.append(f"\nNOTE: the {name} is {val}.\n")
        else:
            out.append(rng.choice(vocab) + ("." if i % 17 == 0 else " "))
    out.append("\n# End of handbook.\n")
    return "".join(out)


def ask(prompt: str, max_tokens: int = 96) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["SM70_API_KEY"]})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    ch = d["choices"][0]
    msg = ch["message"]
    usage = d.get("usage") or {}
    return {
        "content": (msg.get("content") or "").strip(),
        "reasoning": msg.get("reasoning_content"),
        "finish": ch.get("finish_reason"),
        "out_tokens": usage.get("completion_tokens"),
        "secs": round(time.perf_counter() - t0, 2),
    }


def max_ngram_repeat(text: str, n: int = 5) -> int:
    toks = norm(text)
    if len(toks) < n:
        return 1
    grams = [toks[i:i + n] for i in range(len(toks) - n + 1)]
    return max(Counter(grams).values()) if grams else 1


def judge(kind: str, ans: str) -> bool:
    if kind == "SEMI3":
        return all(norm(v) in norm(ans) for _, v in FACTS)
    if kind.startswith("exact:"):
        want = kind.split(":", 1)[1]
        return ans.strip() == want
    if kind == "three_semicolon_items":
        parts = [p.strip() for p in ans.split(";")]
        return (len(parts) == 3 and all(parts)
                and all(p.replace(" ", "").isalpha() and p.islower() for p in parts)
                and "\n" not in ans.strip())
    return norm(kind) in norm(ans)


def run_question(name: str, prompt: str, judge_spec, runs: list, k: int,
                 max_tokens: int = 96) -> dict:
    hits, samples = 0, []
    for _ in range(k):
        r = ask(prompt, max_tokens=max_tokens)
        ok = judge(judge_spec, r["content"])
        hits += int(ok)
        samples.append({**r, "ok": ok,
                        "repeat5": max_ngram_repeat(r["content"])})
    passed = hits > k / 2
    # degenerate signals across samples
    lens = [s["out_tokens"] or 0 for s in samples]
    repeats = max(s["repeat5"] for s in samples)
    eos = sum(1 for s in samples if s["finish"] == "stop")
    rec = {"name": name, "pass": passed, "hits": hits, "k": k,
           "out_tokens": lens, "max_repeat5": repeats, "eos": eos,
           "samples": samples}
    runs.append(rec)
    flag = "OK  " if passed else "MISS"
    print(f"  [{flag}] {name:<28} hit={hits}/{k} eos={eos}/{k} "
          f"repeat5={repeats} out={lens}", flush=True)
    for s in samples:
        if not s["ok"]:
            print(f"        miss[{s['secs']}s,{s['finish']}]: {s['content']!r}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--long-words", type=int, default=22000)
    a = ap.parse_args()
    outdir = Path("/data/sm70_results/quality_baseline")
    outdir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    t0 = time.perf_counter()
    lp = build_long_prompt(a.long_words)
    print(f"  long prompt: {len(lp)} chars (~{len(lp)//4} tokens)", flush=True)
    run_question("longctx-retrieval(3 facts)", lp + "\n\n" + LONG_Q,
                 "SEMI3", runs, a.k, max_tokens=160)

    for i, (p, gt) in enumerate(ARITH, 1):
        run_question(f"arith-{i}", p, gt, runs, a.k)
    for i, (p, spec) in enumerate(FORMAT, 1):
        run_question(f"format-{i}", p, spec, runs, a.k)

    total = len(runs)
    passed = sum(1 for r in runs if r["pass"])
    met = passed == total
    worst_rep = max(r["max_repeat5"] for r in runs)
    summary = {
        "tag": a.tag, "k": a.k, "questions": total, "passed": passed,
        "met": met, "worst_repeat5": worst_rep,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "runs": runs,
    }
    path = outdir / f"quality_{a.tag}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(f"QUALITY_HARNESS: {passed}/{total} questions pass majority "
          f"(bar: every question majority-correct over K={a.k}); worst_repeat5={worst_rep}")
    print("QUALITY_STANDARD_" + ("MET" if met else "FAIL") + f"  -> {path}")
    return 0 if met else 1


if __name__ == "__main__":
    sys.exit(main())
