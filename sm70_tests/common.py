#!/usr/bin/env python3
"""Shared helpers for the SM70 perf / correctness test scripts.

Location contract: this directory (/data/src/vllm-v2/sm70_tests) is
versioned with the tree and survives reboots — test scripts must NEVER
live in /tmp (volatile).

Metrics source: the server-side X-Request-Perf header (nested format:
{"prefill":{ms,tokens,cached,tps},"decode":{ms,tokens,tps}}) is
authoritative; client wall time is auxiliary.

Results go to /data/sm70_results (override with SM70_RESULTS_DIR).
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("SM70_RESULTS_DIR", "/data/sm70_results"))
ENV_FILE = Path("/data/src/sm70_env.sh")
MODEL_PATH = os.environ.get("SM70_MODEL_PATH", "/data/models/Qwen3.8-Flash-Next-FP4-Local-NVFP4")


def base_url() -> str:
    return os.environ.get("SM70_BASE", "http://127.0.0.1:8001")


def model_name() -> str:
    return os.environ.get("SM70_MODEL", "Qwen3.8-Flash-Next")


def api_key() -> str:
    key = os.environ.get("SM70_API_KEY")
    if key:
        return key
    try:
        m = re.search(r"SM70_API_KEY=(\S+)", ENV_FILE.read_text())
    except OSError:
        m = None
    if not m:
        raise SystemExit(
            "SM70_API_KEY not set — export it or set it in /data/src/sm70_env.sh"
        )
    return m.group(1)


# ---------------------------------------------------------------------------
# Prompt construction (deterministic, token-count controlled)
#
# Mirrors llm_speedtest: a seeded sequence of single-token words (its
# PROMPT_TOKEN_WORDS bank) + a fixed instruction suffix. Repeated natural
# prose BPE-compresses away and would collapse the length ladder, so a
# word bank is used (each word = 1 token). Each request's seed makes its
# prefix unique, so the server's prefix cache cannot inflate perf numbers;
# and all metrics use the server-reported actual prompt_tokens, so
# residual construction error is absorbed — the same principles as
# llm_speedtest v1.8.
# ---------------------------------------------------------------------------

# Single-token word bank mirroring llm_speedtest's PROMPT_TOKEN_WORDS:
# each entry is a common word that tokenizes to exactly one token, so the
# filler is ~1 token/word and does NOT BPE-compress (repeated sentences
# do — that is why llm_speedtest uses a word bank, not repeated prose).
_WORDS = [
    'a', 'the', 'and', 'of', 'to', 'in', 'is', 'it', 'that', 'for', 'with', 'as',
    'on', 'by', 'from', 'this', 'be', 'are', 'or', 'not', 'we', 'you', 'they',
    'can', 'will', 'if', 'all', 'one', 'time', 'world', 'life', 'work', 'data',
    'model', 'token', 'text', 'idea', 'mind', 'story', 'light', 'space',
    'future', 'human', 'system', 'simple', 'clear', 'reason', 'change',
    'value', 'truth',
]
# Fixed instruction tail (llm_speedtest's DEFAULT_BENCHMARK_SUFFIX): the
# filler must read like a plausible prompt, not token soup.
_SUFFIX = (
    "\nBased on the words above, write a short philosophical essay "
    "discussing the meaning of existence, the nature of consciousness, "
    "and humanity's place in the universe. Use clear, coherent sentences."
)


def make_prompt(target_tokens: int, seed: int = 0) -> str:
    """Deterministic prompt with ~target_tokens.

    Construction mirrors llm_speedtest: a seeded sequence of single-token
    words + a fixed instruction suffix. ``seed`` makes each request's
    prefix unique (llm_speedtest does this so vLLM prefix caching cannot
    inflate the numbers — different seeds share no prefix); determinism
    is preserved because the seed is caller-controlled (perf_bench uses
    the task index). Tokens-per-word is calibrated empirically, and all
    downstream metrics use the server-reported actual prompt_tokens
    (X-Request-Perf header), so residual construction error is absorbed —
    the same principle as llm_speedtest v1.8 (use the API's real
    prompt_tokens, not the estimate).
    """
    rng = random.Random(seed * 1000 + target_tokens * 7)
    sample = " ".join(rng.choice(_WORDS) for _ in range(64))
    tpp = max(0.5, count_tokens(sample) / 64)
    suffix_tokens = count_tokens(_SUFFIX)
    need = max(0, target_tokens - suffix_tokens)
    n_words = max(1, int(need / tpp))
    text = ""
    for _ in range(6):
        text = " ".join(rng.choice(_WORDS) for _ in range(n_words)) + _SUFFIX
        n = count_tokens(text)
        if abs(n - target_tokens) <= max(8, int(0.02 * target_tokens)):
            break
        denom = max(1, n - suffix_tokens)
        n_words = max(1, int(n_words * target_tokens / max(1, denom)))
    return text


def _tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise SystemExit("transformers not found in /data/vllm-dev venv") from e
    return AutoTokenizer.from_pretrained(MODEL_PATH)


_TOK = None


def count_tokens(text: str) -> int:
    global _TOK
    if _TOK is None:
        _TOK = _tokenizer()
    return len(_TOK.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _post(body: dict, timeout: float) -> tuple[int, dict, bytes, float]:
    """Non-streaming POST /v1/chat/completions -> (status, headers, body, secs)."""
    req = urllib.request.Request(
        base_url() + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key(),
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        hdrs = {k.lower(): v for k, v in r.headers.items()}
    return r.status, hdrs, data, time.time() - t0


@dataclass
class PerfInfo:
    """Parsed X-Request-Perf header (nested format). None group => unknown."""

    prefill_ms: float | None = None
    prefill_tokens: int | None = None
    cached: int | None = None
    prefill_tps: float | None = None
    decode_ms: float | None = None
    decode_tokens: int | None = None
    decode_tps: float | None = None

    @classmethod
    def from_header(cls, raw: str | None) -> PerfInfo | None:
        if not raw:
            return None
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            return None
        p, de = d.get("prefill") or {}, d.get("decode") or {}
        return cls(
            prefill_ms=p.get("ms"),
            prefill_tokens=p.get("tokens"),
            cached=p.get("cached"),
            prefill_tps=p.get("tps"),
            decode_ms=de.get("ms"),
            decode_tokens=de.get("tokens"),
            decode_tps=de.get("tps"),
        )


def chat_nonstream(prompt: str, max_tokens: int, timeout: float = 300.0):
    """Non-streaming chat. Returns dict with text/usage/perf/wall."""
    body = {
        "model": model_name(),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    status, hdrs, data, wall = _post(body, timeout)
    d = json.loads(data.decode())
    usage = d.get("usage") or {}
    choices = d.get("choices") or [{}]
    msg = (choices[0].get("message") or {}) if choices else {}
    return {
        "status": status,
        "text": msg.get("content") or "",
        "reasoning": msg.get("reasoning") or "",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "perf": PerfInfo.from_header(hdrs.get("x-request-perf")),
        "wall": wall,
    }


def full_text(r: dict) -> str:
    """reasoning + content — this model thinks before it answers, so
    content gates must search both (content is null when the token budget
    runs out inside the reasoning phase)."""
    return (r.get("reasoning") or "") + (r.get("text") or "")


def chat_stream(prompt: str, max_tokens: int, timeout: float = 300.0):
    """Streaming chat. Returns text, header perf (prefill group only), and
    client-side first/last byte times (decode cross-check)."""
    body = {
        "model": model_name(),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        base_url() + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key(),
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        hdrs = {k.lower(): v for k, v in r.headers.items()}
        status = r.status
        parts: list[str] = []
        reason_parts: list[str] = []
        t_first = t_last = None
        n_chunks = 0
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            t_last = time.time() - t0
            if t_first is None:
                t_first = time.time() - t0
                n_chunks += 1
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices") or []:
                delta = (choice.get("delta") or {})
                content = delta.get("content")
                if content:
                    parts.append(content)
                    n_chunks += 1
                reasoning = delta.get("reasoning") or delta.get(
                    "reasoning_content"
                )
                if reasoning:
                    reason_parts.append(reasoning)
    wall = time.time() - t0
    return {
        "status": status,
        "text": "".join(parts),
        "reasoning": "".join(reason_parts),
        "perf": PerfInfo.from_header(hdrs.get("x-request-perf")),
        "wall": wall,
        "t_first": t_first,
        "t_last": t_last,
        "n_chunks": n_chunks,
    }


# ---------------------------------------------------------------------------
# Results (durable — /tmp is volatile and forbidden for test artifacts)
# ---------------------------------------------------------------------------

def record(tag: str, lines: list[str]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    f = RESULTS_DIR / f"{tag}.log"
    with open(f, "a") as fh:
        for line in lines:
            fh.write(f"{time.strftime('%F %T')} {line}\n")
    return f


def write_csv(tag: str, header: list[str], rows: list[list]) -> Path:
    import csv

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    f = RESULTS_DIR / f"{tag}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return f


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": 0, "med": 0, "p95": 0}
    xs = sorted(xs)
    return {
        "mean": sum(xs) / len(xs),
        "med": xs[len(xs) // 2],
        "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))],
    }
