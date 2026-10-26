#!/usr/bin/env python3
"""SM70 correctness suite — content gates on top of the perf plumbing.

Input construction uses the same deterministic token-ladder prompts as
perf_bench (llm_speedtest-style), plus fixed reference questions. Checks:

  1. known_answer     — fixed math/reasoning prompts, output must contain
                        the known answer (content gate)
  2. determinism      — same input twice (greedy) => identical text
  3. stream_equal     — streaming text == non-streaming text (same input)
  4. length_respect   — completion_tokens <= max_tokens and >= 1
  5. prefix_cache     — repeat a long prompt: second request must report
                        cached > 0 in X-Request-Perf AND return identical
                        text (cache must not change content)
  6. needle           — marker buried in a 2k-token prompt is retrieved
  7. usage_sane       — every request above has positive usage counts

Usage:
    python3 sm70_tests/correctness.py
    SM70_RESULTS_DIR=/data/sm70_results python3 sm70_tests/correctness.py

Durable outputs: /data/sm70_results/correctness.log (VERDICT line).
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.__str__())
import common  # noqa: E402

KNOWN_ANSWERS = [
    ("17*23等于?", "391"),
    ("56乘以12等于?", "672"),
    ("把字符串 hello 反转后是什么?只输出结果。", "olleh"),
]


def check_known_answers(timeout: float) -> list[tuple[str, bool, str]]:
    out = []
    for q, expect in KNOWN_ANSWERS:
        # 256: this model thinks before answering — 64 tokens can all go
        # to reasoning and content stays null (round-2 gate failure).
        r = common.chat_nonstream(q, 256, timeout)
        ok = r["status"] == 200 and expect in common.full_text(r)
        out.append((f"known_answer[{expect}]", ok, common.full_text(r)[:80]))
    return out


def check_determinism(timeout: float) -> list[tuple[str, bool, str]]:
    prompt = common.make_prompt(512)
    a = common.chat_nonstream(prompt, 64, timeout)
    b = common.chat_nonstream(prompt, 64, timeout)
    fa, fb = common.full_text(a), common.full_text(b)
    ok = a["status"] == b["status"] == 200 and fa == fb and bool(fa)
    return [("determinism", ok, f"len={len(fa)}/{len(fb)}")]


def check_stream_equal(timeout: float) -> list[tuple[str, bool, str]]:
    prompt = common.make_prompt(256)
    # 128: leaves room for the reasoning phase so content is generated on
    # both sides; content (not reasoning) is compared across modes.
    a = common.chat_nonstream(prompt, 128, timeout)
    b = common.chat_stream(prompt, 128, timeout)
    ok = a["status"] == 200 and b["status"] == 200 and a["text"] == b["text"]
    return [("stream_equal", ok,
             f"nonstream={len(a['text'])} stream={len(b['text'])}")]


def check_length(timeout: float) -> list[tuple[str, bool, str]]:
    r = common.chat_nonstream(common.make_prompt(256), 32, timeout)
    ok = r["status"] == 200 and 1 <= (r["completion_tokens"] or 0) <= 32
    return [("length_respect", ok, f"completion_tokens={r['completion_tokens']}")]


def check_prefix_cache(timeout: float) -> list[tuple[str, bool, str]]:
    # 4096-token prompt: a long shared prefix gives an unambiguous reuse signal.
    # Under speculative decoding the per-request `cached` counter is NOT credited
    # for short prompts (measured 2026-10-19 MTP k=1: a 1024-token prompt reports
    # cached=0 in both requests, while the same server reuses 7200 of 8244 tokens
    # on an 8192-token prompt and drops prefill 1354ms -> 291ms). So assert the
    # behaviour, not the metric: reuse must show up either as cached>0 or as a
    # collapsed second prefill.
    prompt = common.make_prompt(4096)
    a = common.chat_nonstream(prompt, 128, timeout)
    b = common.chat_nonstream(prompt, 128, timeout)
    pa, pb = a["perf"], b["perf"]
    cached = pb.cached if pb else None
    ms_a = pa.prefill_ms if pa else None
    ms_b = pb.prefill_ms if pb else None
    fa, fb = common.full_text(a), common.full_text(b)
    reused = bool(cached and cached > 0) or bool(
        ms_a and ms_b and ms_b < 0.5 * ms_a
    )
    ok = (a["status"] == 200 and b["status"] == 200
          and fa == fb and bool(fa) and reused)
    return [("prefix_cache", ok,
             f"cached={cached} prefill={ms_a}->{ms_b}ms same_text={fa == fb}")]


def check_needle(timeout: float) -> list[tuple[str, bool, str]]:
    marker = "N4X7Q9K2"
    base = common.make_prompt(2048)
    lines = base.split("\n")
    prompt = "\n".join(lines[: len(lines) // 2]
                       + [f"请记住这个唯一编号: {marker}。"]
                       + lines[len(lines) // 2:])
    r = common.chat_nonstream(f"{prompt}\n\n请只输出我之前给你的那个唯一编号。", 256, timeout)
    ok = r["status"] == 200 and marker in common.full_text(r)
    return [("needle", ok, common.full_text(r)[:60])]


def check_usage(timeout: float) -> list[tuple[str, bool, str]]:
    r = common.chat_nonstream(common.make_prompt(128), 16, timeout)
    ok = (r["status"] == 200
          and (r["prompt_tokens"] or 0) > 0
          and (r["completion_tokens"] or 0) > 0
          and r["perf"] is not None)
    return [("usage_sane", ok,
             f"prompt={r['prompt_tokens']} completion={r['completion_tokens']} "
             f"perf={'yes' if r['perf'] else 'NO'}")]


def main() -> int:
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    checks = [
        check_known_answers,
        check_determinism,
        check_stream_equal,
        check_length,
        check_prefix_cache,
        check_needle,
        check_usage,
    ]
    results: list[tuple[str, bool, str]] = []
    for fn in checks:
        t0 = time.time()
        try:
            results.extend(fn(timeout))
        except Exception as e:  # noqa: BLE001 — report and continue
            results.append((fn.__name__, False, f"EXC: {type(e).__name__}: {e}"))
        print(f"  {fn.__name__} done in {time.time() - t0:.1f}s")

    failed = [r for r in results if not r[1]]
    verdict = "CORRECT-OK" if not failed else "CORRECT-FAIL"
    line = f"{verdict} checks={len(results)} failed={len(failed)} " + \
        (f" details={[(n, d) for n, _, d in failed]}" if failed else "")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(line)
    log = common.record(
        "correctness",
        [f"{name} {'PASS' if ok else 'FAIL'} {detail}" for name, ok, detail in results]
        + [line],
    )
    print(f"results: {log}")
    return 0 if verdict == "CORRECT-OK" else 1


if __name__ == "__main__":
    sys.exit(main())
