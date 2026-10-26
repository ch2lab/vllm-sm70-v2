#!/usr/bin/env python3
"""SM70 baseline-window ruler — ported from the 1Cat contract ruler
(/data/src/vllm/bench_1cat_contract.py).

Contract (ported): single-request ladder over prompt lengths; exact
token-id prompts; APC-immune (per-length salt + per-run rotation);
thinking-OFF; greedy (temperature=0); ignore-EOS; max_tokens=513 (512
decode intervals); warm x1 + timed x2 per bucket.

Differences vs the old ruler (all deliberate, 2026-10-13):
- Primary metrics come from server-side sources (common.py contract:
  server perf data authoritative, client wall time auxiliary):
    * prefill tok/s = X-Request-Perf header `prefill.tps` (present on every
      run, including the streaming timed runs — see below);
    * decode tps = the service journal perf line's decode `tok/s` (engine
      rate = TARGETS §4's caliber) from each timed run's own line.
  Client-side TTFT/TPOT and the warm run's server-header decode tps are
  kept as cross-check columns (divergence > 5% -> anomaly row, never a
  service change).
- The BAR comparison table / gap columns are NOT ported (bar reference
  was stopped).
- thinking-OFF caliber, established by live probe 2026-10-13
  (/data/sm70_results/baseline_window/think_probe*.json):
    * raw /v1/completions prompts think by DEFAULT (a 256-token prompt
      spent its whole 64-token budget inside `
...`
      reasoning and never reached the answer — finish=length);
    * empty-think-block suffix and "/no_think" in the prompt do NOT
      suppress it; chat_template_kwargs is a no-op on /v1/completions;
    * the working switch is the per-request sampling parameter
      `thinking_token_budget: 0` (engine-side: as soon as the `
`
      start marker appears, the forced end sequence yields an EMPTY
      think block, then the model answers — probes P8/P9: output
      `
4`, 6 tokens, finish=stop).
  Every request in this ruler sends it.
- The warm run is NON-streaming: its response header (emitted at
  completion) carries the FULL perf record (prefill AND decode groups),
  so it doubles as the bucket's server-side decode cross-check. The two
  timed runs are STREAMING: the first SSE chunk's response header carries
  the prefill group (decode group omitted by design of the perf middleware
  — decode unknown at first-token time), and the client measures TTFT/TPOT
  for the cross-check. The journal perf line is the decode-tps source
  (primary) for those timed runs.

APC-immune prompt construction (ported, unchanged): exactly `n` tokens =
per-length salt `SALT-{n} ` (prompts of different lengths are
prefix-disjoint — 2026-10-10: rotate-only still hit 13-50% APC across
rows because the ladder lengths are integer multiples) + per-run
rotation of the fixed passage (consecutive runs of the same length are
prefix-disjoint). With the live service's --enable-prefix-caching ON, no
run can be served from cache — enforced on EVERY run (warm + both timed):
if header prefill.cached != 0 OR journal perf line `cache hit` > 0, the
run is INVALID and excluded from all means (dual gate).

Guards (ported): completion_tokens == 513 AND prompt_tokens == n;
otherwise the run is INVALID and excluded from all aggregates.

Outputs (durable, never /tmp): /data/sm70_results/baseline_window/
  baseline_<date>.jsonl  — one line per run (warm + timed + errors)
  baseline_table.md      — per bucket: prefill tok/s and decode tps
                           (min/med over the timed runs) + caliber label

Usage:
  /data/vllm-dev/bin/python sm70_tests/baseline.py --selftest
  /data/vllm-dev/bin/python sm70_tests/baseline.py --lens 256 --limit 1
  /data/vllm-dev/bin/python sm70_tests/baseline.py \
      --lens 256 512 1024 2048 4096 8192 16384 32768 65536

Exit code: 0 when every bucket has all REPEATS valid timed runs, else 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parent.__str__())
import common  # noqa: E402

MODEL_DIR = "/data/models/Qwen3.8-Flash-Next-NVFP4"
NOUT = 513  # 512 decode intervals (ported contract)
WARMUP = 1  # protocol: warm x1 + timed x2 per bucket (old ruler 1/2)
REPEATS = 2
SERVE_UNIT = "sm70-serve"
OUT_DIR = common.RESULTS_DIR / "baseline_window"

# thinking-OFF for this tree's raw /v1/completions caliber — see module
# docstring for the probe trail (2026-10-13).
THINKING_OFF = {"thinking_token_budget": 0}

# Fixed passage: verbatim from the old ruler (realistic bilingual
# technical prose so the PLE n-gram access pattern is realistic, not
# random tokens).
PASSAGE = (
    "人工智能与机器学习在当代科学和工程中的广泛应用正在快速改变各个行业"
    "的发展轨迹，并且推动着技术革命向前发展。大语言模型服务在预填充和解码"
    "两个阶段表现出截然不同的性能特征：预填充阶段需要处理成批的输入 token，"
    "解码阶段则逐 token 生成，受显存带宽与内核启动开销的限制。在高并发场景"
    "下，连续批处理调度器通过动态重排请求来提高 GPU 利用率，前缀缓存则让共"
    "享前缀的请求跳过重复计算，而张量并行把每个矩阵运算分摊到多张卡上。"
)


# ---------------------------------------------------------------------------
# Prompt construction (ported from bench_1cat_contract.py, unchanged)
# ---------------------------------------------------------------------------

def build_prompt_ids(n: int, tokenizer, rotate: int = 0,
                     salt_seed: int = 0) -> list[int]:
    """Exactly `n` tokens, deterministic per (n, rotate, salt_seed).

    Leading per-row salt (``SALT-{n} `` — ``SALT-{n}-{salt_seed} `` when
    salt_seed > 0) + `rotate` cyclic shift. The salt makes prompts of
    different lengths prefix-disjoint; rotation keeps consecutive runs of
    the same row prefix-disjoint. With both, a service with prefix caching
    ON cannot serve a timed run from cache. A nonzero salt_seed makes the
    FIRST token of every window unique, so a same-boot re-run cannot hit
    a previously cached body either: APC is prefix-based, so any leading
    token that differs from the cached one breaks the hit entirely
    (finding 2026-10-13: a 65536 re-run got a 99.3% cache hit off the
    65K body the earlier trace-window run had left cached).
    """
    enc = tokenizer.encode(PASSAGE, add_special_tokens=False)
    base = enc["input_ids"] if isinstance(enc, dict) else enc
    if not base:
        raise RuntimeError("tokenizer returned no ids")
    salt_suffix = f"-{salt_seed}" if salt_seed else ""
    salt_enc = tokenizer.encode(f"SALT-{n}{salt_suffix} ",
                                add_special_tokens=False)
    salt = salt_enc["input_ids"] if isinstance(salt_enc, dict) else salt_enc
    if not salt:
        raise RuntimeError("tokenizer returned no salt ids")
    m = len(base)
    r = rotate % m
    base = base[r:] + base[:r]
    body = (base * (n // m + 1))[: max(0, n - len(salt))]
    return (salt + body)[:n]


def fingerprint(ids: list[int]) -> str:
    h = hashlib.sha256()
    h.update(len(ids).to_bytes(4, "big"))
    h.update(",".join(map(str, ids[:512])).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# /v1/completions (common.py has no completions helper — only chat)
# ---------------------------------------------------------------------------

def _req(path: str, body: dict) -> urllib.request.Request:
    return urllib.request.Request(
        common.base_url() + path,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + common.api_key(),
        },
        method="POST",
    )


def _base_body() -> dict:
    return {
        "model": common.model_name(),
        "max_tokens": NOUT,
        "temperature": 0,
        "ignore_eos": True,
    }


def completions_stream(prompt_ids: list[int], timeout: float = 600.0) -> dict:
    """One streaming run. None-ish record with 'err' on failure.

    The X-Request-Perf header is emitted WITH the first SSE chunk (the
    middleware holds the start message until first-token time), so
    urlopen() returning is first-token time: client TTFT = t0 -> return.
    The header carries the prefill group only (perf middleware design).
    """
    body = _base_body()
    body.update(THINKING_OFF)
    body.update({"prompt": prompt_ids, "stream": True,
                 "stream_options": {"include_usage": True}})
    t0_unix = time.time()
    t0 = time.perf_counter()
    usage: dict = {}
    rid = None
    n_text_chunks = 0
    try:
        with urllib.request.urlopen(_req("/v1/completions", body),
                                    timeout=timeout) as r:
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            t_first = time.perf_counter()
            status = r.status
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if rid is None:
                    rid = chunk.get("id")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("choices"):
                    n_text_chunks += 1
            t_end = time.perf_counter()
    except urllib.error.HTTPError as e:
        return {"err": f"HTTP {e.code}: {e.read()[:200]!r}", "t0_unix": t0_unix}
    except Exception as e:  # noqa: BLE001 - report, don't crash the ladder
        return {"err": f"{type(e).__name__}: {e}", "t0_unix": t0_unix}
    if n_text_chunks == 0:
        return {"err": "no tokens", "usage": usage, "t0_unix": t0_unix}
    return {
        "mode": "stream",
        "status": status,
        "request_id": rid,
        "usage": usage,
        "perf": common.PerfInfo.from_header(hdrs.get("x-request-perf")),
        "perf_raw": hdrs.get("x-request-perf"),
        "ttft_s": t_first - t0,
        "decode_s": t_end - t_first,
        "wall_s": t_end - t0,
        "t0_unix": t0_unix,
    }


def completions_nonstream(prompt_ids: list[int],
                          timeout: float = 600.0) -> dict:
    """One non-streaming run: header at completion carries the FULL perf
    record (prefill + decode groups)."""
    body = _base_body()
    body.update(THINKING_OFF)
    body.update({"prompt": prompt_ids, "stream": False})
    t0_unix = time.time()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(_req("/v1/completions", body),
                                    timeout=timeout) as r:
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            d = json.loads(r.read().decode())
            status = r.status
        t_end = time.perf_counter()
    except urllib.error.HTTPError as e:
        return {"err": f"HTTP {e.code}: {e.read()[:200]!r}", "t0_unix": t0_unix}
    except Exception as e:  # noqa: BLE001
        return {"err": f"{type(e).__name__}: {e}", "t0_unix": t0_unix}
    return {
        "mode": "nonstream",
        "status": status,
        "request_id": d.get("id"),
        "usage": d.get("usage") or {},
        "perf": common.PerfInfo.from_header(hdrs.get("x-request-perf")),
        "perf_raw": hdrs.get("x-request-perf"),
        "wall_s": t_end - t0,
        "t0_unix": t0_unix,
    }


# ---------------------------------------------------------------------------
# Guards + journal dual-check
# ---------------------------------------------------------------------------

def row_from_run(r: dict, n_prompt: int) -> dict | None:
    """Guard + arithmetic. None if the run is not a valid timed sample."""
    u = r.get("usage") or {}
    if u.get("completion_tokens") != NOUT:
        return None
    if u.get("prompt_tokens") != n_prompt:
        return None
    if r["mode"] == "stream":
        if r.get("decode_s") is None or r["decode_s"] <= 0:
            return None
        return {
            "prefill_tok_s": (n_prompt / r["ttft_s"]
                              if r.get("ttft_s") else None),
            "decode_tok_s": (NOUT - 1) / r["decode_s"],
            "tpot_ms": r["decode_s"] / (NOUT - 1) * 1000,
            "ttft_ms": r["ttft_s"] * 1000,
        }
    p = r.get("perf")
    if p is None or not p.decode_ms or p.decode_ms <= 0:
        return None
    return {
        "prefill_tok_s": p.prefill_tps,
        "decode_tok_s": p.decode_tps,
        # decode_ms is already ms: ms / tokens = ms per token.
        "tpot_ms": (p.decode_ms / p.decode_tokens
                    if p.decode_tokens else None),
        "ttft_ms": p.prefill_ms,
    }


def journal_perf(request_id: str | None, t0_unix: float) -> dict | None:
    """Parse this run's service perf line (journal half of the cache dual
    gate + the decode-tps primary source):
        ... input N tokens(prefill X ms, Y tok/s, cache hit: Z%),
            output M tokens(decode W ms, V tok/s)
    Returns {"cache_hit_pct", "prefill_tps", "decode_tps"} (fields may be
    None) or None when the line is unavailable (unit missing, not found)."""
    if not request_id:
        return None
    since = time.strftime("%F %T", time.localtime(t0_unix - 5))
    try:
        out = subprocess.run(
            ["journalctl", "-u", SERVE_UNIT, "-o", "cat", "--since", since],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        if request_id in line and "cache hit:" in line:
            hit = re.search(r"cache hit: ([0-9.]+)%", line)
            pre = re.search(r"prefill [0-9.]+ ms, ([0-9.]+) tok/s", line)
            dec = re.search(r"output [0-9]+ tokens\(decode [0-9.]+ ms, "
                            r"([0-9.]+) tok/s\)", line)
            return {
                "cache_hit_pct": float(hit.group(1)) if hit else None,
                "prefill_tps": float(pre.group(1)) if pre else None,
                "decode_tps": float(dec.group(1)) if dec else None,
            }
    return None


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

def _run_jsonl(length: int, tag: str, rotate: int, fp: str, rec: dict,
               row: dict | None, jperf: dict | None, cache_status: str,
               client: dict | None, ttft_div_pct: float | None) -> dict:
    """One jsonl row. `metrics` = the guard-checked row_from_run numbers
    (client-side for the streaming timed runs; the warm run's nonstream
    response header for the warm run); `journal` = the primary server-side
    numbers (service journal perf line; None when unavailable); `client`
    = the auxiliary client-side cross-check block (timed stream runs only;
    same values as `metrics` for stream runs)."""
    u = rec.get("usage") or {}
    perf = rec.get("perf")
    return {
        "ts": time.strftime("%F %T"),
        "length": length,
        "run": tag,
        "rotate": rotate,
        "fingerprint": fp,
        "mode": rec.get("mode"),
        "status": rec.get("status"),
        "request_id": rec.get("request_id"),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "valid": row is not None,
        "perf": asdict(perf) if perf else None,
        "cached": perf.cached if perf else None,
        "cache_status": cache_status,
        "journal": jperf,
        "client_ttft_ms": (rec["ttft_s"] * 1000 if "ttft_s" in rec else None),
        "ttft_div_pct": ttft_div_pct,
        "metrics": row,
        "client": client,
        "err": rec.get("err"),
    }


def run_bucket(length: int, tok, fh, salt_seed: int = 0) -> dict:
    """One bucket: warm x1 (nonstream) + timed x2 (stream). Returns the
    table row dict."""
    base_ids = build_prompt_ids(length, tok, salt_seed=salt_seed)
    assert len(base_ids) == length
    fp = fingerprint(base_ids)
    print(f"# bucket {length}: prompt fingerprint {fp}")
    timed: list[tuple[dict, dict, dict | None]] = []  # (rec, row, jperf)
    warm_row: dict | None = None
    problems: list[str] = []
    no_journal: list[str] = []  # runs whose journal perf line was not found

    for run_i, tag in [(0, "warm"), (1, "timed1"), (2, "timed2")]:
        ids = build_prompt_ids(length, tok, rotate=run_i,
                               salt_seed=salt_seed)
        assert len(ids) == length
        rec = (completions_nonstream(ids) if run_i == 0
               else completions_stream(ids))
        if "err" in rec:
            print(f"  {tag} FAILED: {rec['err']}", flush=True)
            fh.write(json.dumps(
                _run_jsonl(length, tag, run_i, fp, rec, None, None, "unknown",
                           None, None),
                ensure_ascii=False) + "\n")
            problems.append(f"{tag}: {rec['err']}")
            break  # 不硬跑: a failing bucket stops here (413/OOM case)
        row = row_from_run(rec, length)
        if row is None:
            print(f"  {tag} INVALID: usage={rec['usage']} "
                  f"(need completion={NOUT}, prompt={length})", flush=True)
            fh.write(json.dumps(
                _run_jsonl(length, tag, run_i, fp, rec, None, None, "unknown",
                           None, None),
                ensure_ascii=False) + "\n")
            problems.append(f"{tag}: guard")
            continue
        # Journal perf line (line lands at request end; give journald a
        # moment, one retry) — decode-tps primary source + journal half of
        # the per-run cache gate.
        time.sleep(1.0)
        jperf = journal_perf(rec.get("request_id"), rec["t0_unix"])
        if jperf is None:
            time.sleep(2.0)
            jperf = journal_perf(rec.get("request_id"), rec["t0_unix"])
        # Per-run cache gate (ALL runs — warm and both timed): header
        # cached or journal cache hit > 0 -> run INVALID, excluded from all
        # means. APC-immune design: this should never fire; if it does it
        # is a real finding, surfaced loudly, not papered over.
        hdr_cached = rec["perf"].cached if rec["perf"] else None
        j_hit = jperf["cache_hit_pct"] if jperf else None
        cache_status = "ok"
        if (hdr_cached or 0) > 0 or (j_hit or 0) > 0:
            cache_status = (f"hit hdr_cached={hdr_cached} "
                           f"journal={j_hit}%")
        # Client TTFT vs header prefill ms (stream runs only) — cross-check.
        ttft_div = None
        if rec["mode"] == "stream" and rec["perf"] and rec["perf"].prefill_ms:
            ttft_div = (abs(rec["ttft_s"] * 1000 - rec["perf"].prefill_ms)
                       / rec["perf"].prefill_ms * 100.0)
        client = None
        if rec["mode"] == "stream":
            client = {
                "ttft_ms": row["ttft_ms"],
                "prefill_tok_s": row["prefill_tok_s"],
                "decode_tps": row["decode_tok_s"],
                "tpot_ms": row["tpot_ms"],
            }
        fh.write(json.dumps(
            _run_jsonl(length, tag, run_i, fp, rec, row, jperf, cache_status,
                       client, ttft_div),
            ensure_ascii=False) + "\n")
        if cache_status != "ok":
            print(f"  {tag} CACHE HIT -> run INVALID, excluded from means "
                  f"(hdr cached={hdr_cached}, journal={j_hit}%) "
                  f"— surfacing, not averaging in", flush=True)
            problems.append(f"{tag}: cache {cache_status}")
        elif jperf is None:
            no_journal.append(tag)
            print(f"  {tag} NOTE: journal perf line not found "
                  f"(unit={SERVE_UNIT}?); journal half of the cache check "
                  f"unavailable for this run", flush=True)
        if rec["mode"] == "stream" and run_i >= 1:
            timed.append((rec, row, jperf))
            jd = jperf["decode_tps"] if jperf else None
            print(f"  {tag}: prefill={rec['perf'].prefill_tps if rec['perf'] else None} tok/s (hdr) "
                  f"cached={rec['perf'].cached if rec['perf'] else None} "
                  f"decode={jd if jd is not None else 'n/a'} tps (journal) "
                  f"client_decode={row['decode_tok_s']:.2f} tps "
                  f"client_tpot={row['tpot_ms']:.3f}ms "
                  f"ttft_div={'n/a' if ttft_div is None else format(ttft_div, '.1f')}%"
                  f" journal_hit={j_hit if j_hit is not None else 'n/a'}% "
                  f"cache={cache_status}", flush=True)
        else:
            warm_row = row
            print(f"  {tag}: prefill={row['prefill_tok_s']} tok/s (hdr) "
                  f"decode={row['decode_tok_s']} tps (hdr) "
                  f"cached={rec['perf'].cached if rec['perf'] else None} "
                  f"journal={jperf} cache={cache_status}", flush=True)

    # Bucket aggregates (timed runs only, per the old ruler's protocol).
    # Primary sources: prefill tok/s = X-Request-Perf header (per timed
    # run); decode tps = journal perf line (per timed run). Cross-checks:
    # warm run's server-header decode tps, client TTFT/TPOT.
    out: dict = {"length": length, "fingerprint": fp}
    srv_decode = warm_row["decode_tok_s"] if warm_row else None
    ttfts = []
    for rec, row, _jp in timed:
        if rec["perf"] and rec["perf"].prefill_ms:
            ttfts.append(abs(rec["ttft_s"] * 1000 - rec["perf"].prefill_ms)
                         / rec["perf"].prefill_ms * 100.0)
    if len(timed) == REPEATS and not problems:
        p1 = timed[0][0]["perf"].prefill_tps if timed[0][0]["perf"] else None
        p2 = timed[1][0]["perf"].prefill_tps if timed[1][0]["perf"] else None
        d1 = timed[0][2]["decode_tps"] if timed[0][2] else None
        d2 = timed[1][2]["decode_tps"] if timed[1][2] else None
        pp = [p for p in (p1, p2) if p is not None]
        dd = [d for d in (d1, d2) if d is not None]
        repeat_diff = (abs(d1 - d2) / (statistics.fmean([d1, d2]) or 1) * 100
                       if d1 is not None and d2 is not None else None)
        # Cross-check: warm run's server-header decode vs journal decode.
        xcheck = []
        if srv_decode:
            xcheck = [abs(d - srv_decode) / srv_decode * 100
                      for d in (d1, d2) if d is not None]
        out.update({
            "status": "OK" if not no_journal else "OK*",
            "prefill_min": min(pp) if pp else None,
            "prefill_med": statistics.median(pp) if pp else None,
            "decode_min": min(dd) if dd else None,
            "decode_med": statistics.median(dd) if dd else None,
            "client_decode": [timed[0][1]["decode_tok_s"],
                              timed[1][1]["decode_tok_s"]],
            "srv_decode": srv_decode,
            "journal_missing": sorted(no_journal),
            "repeat_diff_pct": repeat_diff,
            "ttft_div_max": max(ttfts) if ttfts else None,
            "decode_xcheck_max": max(xcheck) if xcheck else None,
        })
        if repeat_diff is not None and repeat_diff >= 5.0:
            print(f"  !! decode tps between repeats differs {repeat_diff:.1f}% "
                  f"(>= 5%) — check service state (do not relax the gate)",
                  flush=True)
            out["anomaly"] = "repeat-diff"
        if xcheck and max(xcheck) > 5.0:
            print(f"  !! journal vs warm-header decode tps diverges "
                  f"{max(xcheck):.1f}% (> 5%) — anomaly row", flush=True)
            out["anomaly"] = out.get("anomaly", "") + "+decode-xcheck"
        if ttfts and max(ttfts) > 5.0:
            print(f"  !! client TTFT vs header prefill ms diverges "
                  f"{max(ttfts):.1f}% (> 5%) — anomaly row", flush=True)
            out["anomaly"] = out.get("anomaly", "") + "+ttft"
        if no_journal:
            print(f"  NOTE: journal perf line missing for {', '.join(no_journal)}"
                  f" — decode-tps source and journal cache half unverified "
                  f"for those runs", flush=True)
    else:
        out["status"] = "INVALID"
        out["problems"] = problems
    return out


def _fmt(v, spec: str) -> str:
    return format(v, spec) if v is not None else "-"


def write_table(rows: list[dict], salt_seed: int = 0) -> Path:
    caliber = (
        "Caliber: thinking-OFF via per-request thinking_token_budget=0 "
        "(raw /v1/completions thinks by default — probe 2026-10-13, "
        "think_probe*.json) | APC-immune prompt: SALT-{n} + per-run "
        "rotation | warm x1 nonstream + timed x2 stream | "
        f"max_tokens={NOUT} (512 decode intervals), greedy, ignore-eos | "
        "guards: completion==513, prompt==n | per-run cache gate: header "
        "cached==0 AND journal 'cache hit: 0%' (any hit -> run INVALID) "
        "| column sources: prefill tok/s = X-Request-Perf header "
        "(server, per timed run); decode tps = journal perf line (engine, "
        "per timed run); srv decode = warm run's nonstream header "
        "(cross-check); client decode / TTFT div = client-side "
        "(cross-check, >5% divergence -> anomaly); repeat diff = journal "
        "decode tps between the two timed runs (engine, >=5% -> anomaly)"
    )
    if salt_seed:
        caliber += f" | salt-seed={salt_seed}"
    md = [
        f"# SM70 baseline window — {time.strftime('%F %T')}",
        "",
        f"Ruler: sm70_tests/baseline.py (ported 1Cat contract ruler, "
        f"exact token-id prompts)",
        f"Service: {common.model_name()} @ {common.base_url()} "
        "(vllm-v2 clean tree, 8xV100 TP8, live — zero restarts)",
        caliber,
        "",
        "| len | prefill tok/s min/med (hdr) | decode tps min/med (journal) "
        "| srv decode tps (warm hdr) | client decode tps (t1/t2) "
        "| TTFT div max % | repeat diff % | cache | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["status"] in ("OK", "OK*"):
            cache = ("0/0/0" if not r["journal_missing"]
                     else "hdr-only (journal missing)")
            md.append(
                f"| {r['length']} "
                f"| {_fmt(r['prefill_min'], '.0f')}/"
                f"{_fmt(r['prefill_med'], '.0f')} "
                f"| {_fmt(r['decode_min'], '.1f')}/"
                f"{_fmt(r['decode_med'], '.1f')} "
                f"| {_fmt(r['srv_decode'], '.1f')} "
                f"| {r['client_decode'][0]:.1f}/{r['client_decode'][1]:.1f} "
                f"| {_fmt(r['ttft_div_max'], '.1f')} "
                f"| {_fmt(r['repeat_diff_pct'], '.1f')} "
                f"| {cache} "
                f"| {r['status']}"
                f"{(' (' + r['anomaly'] + ')') if r.get('anomaly') else ''} "
                f"|")
        else:
            md.append(f"| {r['length']} | - | - | - | - | - | - | - "
                      f"| INVALID ({'; '.join(r.get('problems', []))}) |")
    f = OUT_DIR / "baseline_table.md"
    f.write_text("\n".join(md) + "\n")
    return f


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", nargs="+", type=int, default=[256],
                    help="prompt lengths (tokens), one bucket each")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N buckets (safety cap)")
    ap.add_argument("--salt-seed", type=int, default=0,
                    help="APC immunity for same-boot re-runs: nonzero "
                         "changes the salt to SALT-{n}-{seed} so the "
                         "prompt's first token differs per window "
                         "(any APC hit on a previously cached body "
                         "impossible); 0 keeps the legacy SALT-{n} salt "
                         "(byte-identical prompts)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    lens = args.lens[: args.limit] if args.limit else args.lens
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date = time.strftime("%Y-%m-%d")
    fh = open(OUT_DIR / f"baseline_{date}.jsonl", "a", encoding="utf-8")

    print(f"# ruler: lens={lens} nout={NOUT} "
          f"warmup={WARMUP} repeats={REPEATS} "
          f"thinking=OFF(thinking_token_budget=0)"
          + (f" salt_seed={args.salt_seed}" if args.salt_seed else ""))
    rows = []
    with fh:
        for length in lens:
            rows.append(run_bucket(length, tok, fh, args.salt_seed))
    table = write_table(rows, args.salt_seed)
    n_ok = sum(1 for r in rows if r["status"] in ("OK", "OK*"))
    print(f"# baseline: buckets={len(rows)} ok={n_ok} "
          f"invalid={len(rows) - n_ok}")
    print(f"jsonl: {OUT_DIR / f'baseline_{date}.jsonl'}")
    print(f"table: {table}")
    return 0 if n_ok == len(rows) else 1


# ---------------------------------------------------------------------------
# Offline selftest (ported 4 guard cases + prompt-construction checks)
# ---------------------------------------------------------------------------

class _DummyTok:
    """Stands in for AutoTokenizer in the offline selftest only."""

    def __init__(self):
        self.base = list(range(1000, 1000 + 50))

    def encode(self, text, add_special_tokens=False):
        return {"input_ids": self.base * 3}


class _HashTok:
    """Selftest-only: deterministic ids that VARY with the input text
    (unlike _DummyTok's constant block), so a salt-text change is
    observable in the produced prompt ids."""

    def encode(self, text, add_special_tokens=False):
        d = hashlib.sha256(text.encode("utf-8")).digest()
        return {"input_ids": [int.from_bytes(d[i:i + 4], "big") % 40000
                              + 100000 for i in range(0, 16, 4)]}


def selftest() -> int:
    """Guard logic against cases that must pass and cases that must
    fail, plus prompt-construction invariants (lengths, determinism,
    and salt-seed behavior: seed=0 byte-identical to the legacy salt,
    seed>0 changes the first token and the fingerprint)."""
    failures = []
    valid = {
        "mode": "stream",
        "ttft_s": 2.0,
        "decode_s": 7.0,
        "usage": {"completion_tokens": NOUT, "prompt_tokens": 8192},
    }
    if row_from_run(valid, 8192) is None:
        failures.append("valid run rejected")
    if row_from_run({**valid, "usage": {"completion_tokens": NOUT - 1,
                                        "prompt_tokens": 8192}},
                    8192) is not None:
        failures.append("truncated run accepted")
    if row_from_run({**valid, "usage": {"completion_tokens": NOUT,
                                        "prompt_tokens": 8193}},
                    8192) is not None:
        failures.append("prompt-length mismatch accepted")
    if row_from_run({**valid, "decode_s": 0.0}, 8192) is not None:
        failures.append("zero decode window accepted")
    ids = build_prompt_ids(64, _DummyTok())
    if len(ids) != 64:
        failures.append(f"build_prompt_ids wrong length {len(ids)}")
    if fingerprint(ids) != fingerprint(ids):
        failures.append("fingerprint not deterministic")
    if fingerprint(ids) == fingerprint([1] * 64):
        failures.append("fingerprint ignores content")
    # salt-seed (APC immunity for same-boot re-runs). _HashTok maps each
    # text to distinct deterministic ids, so the salt text itself is
    # observable in the output:
    htok = _HashTok()
    ids0 = build_prompt_ids(64, htok)
    # (a) seed=0 must be byte-identical to the pre-change construction:
    #     salt prefix == the encoding of the literal legacy salt text.
    legacy_salt = htok.encode("SALT-64 ")["input_ids"]
    if ids0[:len(legacy_salt)] != legacy_salt:
        failures.append("seed=0 salt prefix differs from legacy "
                       "'SALT-{n} ' construction")
    # (b) seed=1 changes the first prompt token (APC prefix broken).
    ids1 = build_prompt_ids(64, htok, salt_seed=1)
    if len(ids1) != 64:
        failures.append(f"seed=1 prompt wrong length {len(ids1)}")
    if ids1[0] == ids0[0]:
        failures.append("seed=1 first token identical to seed=0")
    # (c) fingerprints differ between seeds.
    if fingerprint(ids0) == fingerprint(ids1):
        failures.append("seed=1 fingerprint identical to seed=0")
    if failures:
        print(f"SELFTEST FAIL: {failures}")
        return 1
    print("SELFTEST PASS (guards reject invalid runs, accept valid; "
          "salt-seed: seed=0 legacy-identical, seed>0 changes first "
          "token + fingerprint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
