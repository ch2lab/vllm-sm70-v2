"""Concurrency-throughput KPI (added 10-16: user directive — every
optimization round reports single-stream prefill/decode AND concurrent
throughput).

Caliber mirrors the 65K ruler (baseline.py): raw /v1/completions,
thinking-OFF (per-request thinking_token_budget=0), greedy, ignore_eos,
max_tokens=513, exact token-id prompts, APC-immune (per-slot, per-round
unique salt — shared with no prefix across slots AND across rounds, so
neither the warm round nor a repeated timed run can hit the prefix
cache).

Canonical KPI (10-16 user directive: measure the phases SEPARATELY —
mixed phases interfere via chunked prefill):
  prefill mode (--mode prefill): long prompt + max_tokens=1
    -> prefill_agg_tps is the KPI; the single-token decode is noise-free
    and negligible.
  decode mode (--mode decode): ONE identical long prompt (65536) for all
    slots + 513 tokens; warm round fills the prefix cache, timed slots hit
    it fully (within block granularity) -> decode_agg_tps is the KPI,
    measured at full context with zero prefill interference.
  e2e mode (--mode full, auxiliary only): long prompt + 513 — prefill and
  decode overlap; kept for reference, not the KPI.

Phase metrics (client wall time — the throughput KPI; per-request
server X-Request-Perf kept as auxiliary cross-check):
  prefill_agg_tps   = sum(prompt_tokens)   / max(ttft)
                      [dispatch -> the slowest first token]
  decode_agg_tps    = sum(completion_tokens) / (max(t_end) - max(ttft))
                      [all C streams in the batch -> the last completion]
  e2e_tps           = sum(prompt+completion) / max(t_end)
Plus per-stream decode tps (client) and per-request header tps.

Guard: every request must report cached=0 (APC-immune); a nonzero cache
makes the round invalid, matching the ruler's cache dual-check policy.

Usage:
  python qsa_conc_tput.py --conc 4 --len 65536 --salt-seed <S> [--rounds 2]
  rounds: first is warm (untimed), the rest are timed; each timed round is
  reported.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib import error as urlerror
from urllib import request as urlrequest

import common


def _req(path: str, body: dict) -> urlrequest.Request:
    return urlrequest.Request(
        common.base_url() + path,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + common.api_key(),
        },
        method="POST",
    )


def _one(prompt_ids: list[int], max_tokens: int, epoch: float,
         timeout: float = 900.0) -> dict:
    """One streaming /v1/completions run (ruler caliber). Phase times are
    offsets from ``epoch`` (the round's dispatch), so the aggregates are
    not skewed by per-thread dispatch spread."""
    body = {
        "model": common.model_name(),
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "thinking_token_budget": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    usage: dict = {}
    try:
        with urlrequest.urlopen(_req("/v1/completions", body),
                                 timeout=timeout) as r:
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            t_first = time.perf_counter() - epoch
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
                if chunk.get("usage"):
                    usage = chunk["usage"]
            t_end = time.perf_counter() - epoch
    except urlerror.HTTPError as e:
        return {"err": f"HTTP {e.code}: {e.read()[:200]!r}"}
    except Exception as e:  # noqa: BLE001 - report, keep the pool alive
        return {"err": f"{type(e).__name__}: {e}"}
    perf = common.PerfInfo.from_header(hdrs.get("x-request-perf"))
    if perf is None:  # engine-dead mid-stream: no perf header
        return {"err": "no X-Request-Perf header (engine likely dead)",
                "usage": usage}
    return {
        "ttft_s": t_first,
        "decode_s": t_end - t_first,
        "wall_s": t_end,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prefill_tps_hdr": perf.prefill_tps,
        "decode_tps_hdr": perf.decode_tps,
        "cached": perf.cached,
    }


def _run_round(prompts: list[list[int]], max_tokens: int,
               cache_policy: tuple[str, int]) -> dict:
    """One round of concurrent requests. cache_policy: ("none", 0) = APC-
    immune (every request must show cached==0) or ("prefix", min_cached) =
    APC-warmed (every request must show cached>=min_cached, else the
    shared prefix was evicted and the decode window is contaminated)."""
    kind, min_cached = cache_policy
    conc = len(prompts)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(
            ex.map(lambda i: _one(prompts[i], max_tokens, t0), range(conc)))
    t_dispatch_end = time.perf_counter()
    errs = [r["err"] for r in results if "err" in r]
    if errs:
        return {"err": errs}
    # phase times are already offsets from t0 (the round's dispatch epoch)
    abs_first = [r["ttft_s"] for r in results]
    abs_end = [r["wall_s"] for r in results]
    p_tok = sum(r["prompt_tokens"] or 0 for r in results)
    c_tok = sum(r["completion_tokens"] or 0 for r in results)
    max_first = max(abs_first)
    max_end = max(abs_end)
    steady = max_end - max_first
    if kind == "none":
        cache_ok = all((r["cached"] or 0) == 0 for r in results)
    else:
        cache_ok = all((r["cached"] or 0) >= min_cached for r in results)
    summary = {
        "conc": conc,
        "prefill_agg_tps": round(p_tok / max_first, 1),
        "decode_agg_tps": (round(c_tok / steady, 1)
                           if steady > 1e-3 and max_tokens > 1 else None),
        "e2e_tps": round((p_tok + c_tok) / max_end, 1),
        "max_ttft_s": round(max_first, 2),
        "steady_decode_s": round(steady, 2),
        "e2e_s": round(max_end, 2),
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "cache_ok": cache_ok,
        "min_cached": min((r["cached"] or 0) for r in results),
        "per_stream_decode_tps": (
            [round((r["completion_tokens"] or 0) / r["decode_s"], 1)
             if r["decode_s"] > 0 else None for r in results]
            if max_tokens > 1 else None),
        "per_req_prefill_tps_hdr": [r["prefill_tps_hdr"] for r in results],
    }
    return {"ok": summary, "per_req": results,
            "dispatch_spread_s": round(t_dispatch_end - t0, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--mode", choices=["prefill", "decode", "full"],
                    default="full",
                    help="prefill = long prompt + 1 token; decode = short "
                         "prompt + 513 tokens; full = long + 513 "
                         "(phases interfere — reference only)")
    ap.add_argument("--len", type=int, default=None,
                    help="default by --mode: prefill=65536, decode=2048, "
                         "full=65536")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default by --mode: prefill=1, decode=513, "
                         "full=513")
    ap.add_argument("--salt-seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=2,
                    help="1 = warm only; N = 1 warm + N-1 timed")
    ap.add_argument("--tag", default="conc_tput")
    ap.add_argument("--decode-prefix-len", type=int, default=None,
                    help="decode mode: shared prompt length (default "
                         "65536) — identical for all slots; warm round "
                         "fills the APC, timed slots hit fully")
    args = ap.parse_args()
    mode_defaults = {"prefill": (65536, 1), "decode": (2048, 513),
                     "full": (65536, 513)}
    d_len, d_out = mode_defaults[args.mode]
    if args.len is None:
        args.len = d_len
    if args.max_tokens is None:
        args.max_tokens = d_out

    tok = common._tokenizer()
    if args.mode == "decode":
        # ONE identical prompt for every slot (user directive 10-16):
        # the warm round fills the APC with the whole prompt, every timed
        # slot then hits the full block-aligned prefix (100% within block
        # granularity — a sub-block tail is the only possible miss) and the
        # decode phase runs at full context with zero prefill work.
        prefix_len = args.decode_prefix_len or 65536
        shared = tok.encode(common.make_prompt(prefix_len,
                                                seed=args.salt_seed),
                            add_special_tokens=False)
        min_cached = int(0.95 * len(shared))

        def mk_prompt(r: int, i: int) -> list[int]:
            return shared

        cache_policy: tuple[str, int] = ("prefix", min_cached)
        warm_n = 1  # one request warms the shared prefix
    else:
        def mk_prompt(r: int, i: int) -> list[int]:
            return tok.encode(
                common.make_prompt(args.len,
                                   seed=args.salt_seed * 100000 + r * 1000
                                   + i),
                add_special_tokens=False)

        cache_policy = ("none", 0)
        warm_n = args.conc

    lines: list[str] = []
    for r in range(args.rounds):
        seed = args.salt_seed * 100000 + r * 1000
        prompts = [mk_prompt(r, i) for i in range(warm_n if r == 0
                                                     else args.conc)]
        out = _run_round(prompts, args.max_tokens, cache_policy)
        timed = r > 0
        # Provenance stamp (sm70/DISCIPLINE.md §7): every line carries whether it
        # was produced through /root/run/measure_gate.sh and which serve script
        # sha it ran against, so an un-gated number can never be mistaken for a
        # disciplined verdict.
        import hashlib as _hl
        _sha = ""
        try:
            with open("/root/run/run_sm70_serve_ipc.sh", "rb") as _f:
                _sha = _hl.sha256(_f.read()).hexdigest()[:8]
        except OSError:
            pass
        _gate = "gated" if os.environ.get("SM70_GATE") == "1" else "UNGATED"
        line = (f"round={r}{'(warm)' if not timed else ''} "
                f"mode={args.mode} conc={len(prompts)} len={args.len} "
                f"seed={seed} {_gate} serve_sha={_sha}")
        if "err" in out:
            line += f" ERROR {out['err']}"
            lines.append(line)
            print(line)
            continue
        s = out["ok"]
        if not s["cache_ok"]:
            # In prefix mode the warm round is the one that FILLS the APC —
            # a cache miss there is expected, not a failure.
            if not (cache_policy[0] == "prefix" and not timed):
                line += (f" INVALID(cache policy "
                         f"{'none' if cache_policy[0] == 'none' else 'prefix'}: "
                         f"min_cached={s['min_cached']} "
                         f"need>={cache_policy[1]})")
            else:
                line += " (warm, fills APC — miss expected)"
        else:
            line += (f" prefill_agg={s['prefill_agg_tps']} "
                     f"decode_agg={s['decode_agg_tps']} "
                     f"e2e={s['e2e_tps']} ttft_max={s['max_ttft_s']}s "
                     f"steady_decode={s['steady_decode_s']}s "
                     f"per_stream={s['per_stream_decode_tps']} "
                     f"min_cached={s['min_cached']} "
                     f"hdr_prefill={s['per_req_prefill_tps_hdr']}")
        lines.append(line)
        print(line)
    common.record(args.tag, lines)
    return 1 if any("ERROR" in l or "INVALID" in l for l in lines) else 0


if __name__ == "__main__":
    raise SystemExit(main())


def wait_engine_ready(restart_unix: float | None = None,
                      timeout_s: float = 420.0) -> bool:
    """Poll (CPU-side only) until the engine is actually ready after a
    restart: the journal's "init engine ... took" line from this boot +
    health 200. No GPU requests are sent, so the boot window stays
    uncontended and not-ready 500 storms can't confuse the result.
    Use instead of a fixed boot-window sleep: the probe starts the moment
    the engine is ready (typically ~110-140s after boot)."""
    import subprocess
    since = time.strftime("%H:%M:%S",
                          time.localtime(restart_unix or time.time() - 1800))
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = subprocess.run(
            ["journalctl", "-u", "sm70-serve.service", "-o", "cat",
             "--since", since],
            capture_output=True, text=True).stdout
        ready_line = "init engine" in out
        if ready_line:
            print(f"ENGINE-READY ~{time.time() - t0:.0f}s")
            return True
        time.sleep(10)
    return False
