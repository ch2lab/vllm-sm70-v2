#!/usr/bin/env python3
"""Per-kernel GPU attribution for a batch=1 decode, via vLLM's torch profiler.

Decode at batch=1 sits ~6x above the weight-bandwidth floor on TP1, and the
number that decides what to fix is the per-kernel split of one step.  This
drives the server-side torch profiler (no .item(), no --enforce-eager, both of
which are unusable under CUDA-graph capture) over a prefix-cached request so
the captured window is decode steps only, then aggregates kernel self-time.

    python sm70_tests/bench_decode_profile.py --base http://127.0.0.1:8001 \
        --prompt-len 10000 --tokens 256 --out /data/sm70_results/tp1_prof

Requires the server to have been started with VLLM_TORCH_PROFILER_DIR set
(serve_qwen36.sh turns that env into --profiler-config).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
import urllib.request

API_KEY_PATH = "/root/.vllm_api_key"


def _key() -> str:
    try:
        with open(API_KEY_PATH) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _post(base: str, path: str, timeout: int = 600) -> None:
    req = urllib.request.Request(
        base + path,
        data=b"{}",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_key()}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()


def chat(base: str, prompt: str, tokens: int, timeout: int = 3600) -> float:
    body = json.dumps({
        "model": "Qwen3.6-35B-A3B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_key()}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read())
    ch = j["choices"][0]
    used = j.get("usage", {}).get("prompt_tokens")
    return time.perf_counter() - t0, used


def n_tokens(base: str, text: str) -> int:
    body = json.dumps({"model": "Qwen3.6-35B-A3B", "prompt": text}).encode()
    req = urllib.request.Request(
        base + "/tokenize", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_key()}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return int(json.loads(r.read())["count"])


def haystack(base: str, tokens: int, salt: int = 9502) -> str:
    """Unique lines so nothing is dedupable, cut to ``tokens`` **measured** by the
    server's tokenizer.  Estimating from words/chars has been wrong three times
    here (13 vs 14.7 tokens per line overran max_model_len and the request was
    rejected), so the length is fitted, not assumed."""
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
             "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega "
             "copper silver cobalt nickel zinc iron quartz basalt granite "
             "marble slate").split()
    import random
    rnd = random.Random(salt)
    lines = [f"record {i:06d} " + " ".join(rnd.choice(words) for _ in range(11))
             for i in range(tokens // 12 + 2)]
    lo, hi = 1, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_tokens(base, "\n".join(lines[:mid])) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    return "\n".join(lines[:lo])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--prompt-len", type=int, default=10000)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--prof-salt", type=int, default=0,
                    help="use a different document for the profiled request "
                         "(0 = same as warm, i.e. prefix-cache friendly)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # leave headroom: the chat template adds tokens on top of the content
    target = args.prompt_len - 64
    prompt = haystack(args.base, target)
    prof_prompt = (prompt if args.prof_salt == 0
                   else haystack(args.base, target, args.prof_salt))

    # Warm: compile/graph replay plus the prefix cache, so the profiled window
    # is steady-state decode rather than prefill.
    _, p_warm = chat(args.base, prompt, 8)
    _, p_warm2 = chat(args.base, prompt, 8)
    print(f"[prof] warm prompt_tokens={p_warm} (2nd call: {p_warm2})", flush=True)

    before = set(glob.glob(os.path.join(args.out, "*.trace.json*")))
    t, p_un = chat(args.base, prompt, args.tokens)
    print(f"[prof] unprofiled run: {t:.2f} s for {args.tokens} tokens "
          f"({args.tokens / t:.1f} tok/s), prompt_tokens={p_un}", flush=True)

    _post(args.base, "/start_profile")
    t, p_prof = chat(args.base, prof_prompt, args.tokens)
    print(f"[prof] profiled run:   {t:.2f} s ({args.tokens / t:.1f} tok/s), "
          f"prompt_tokens={p_prof}", flush=True)
    _post(args.base, "/stop_profile", timeout=1200)

    print("[prof] waiting for trace files...", flush=True)
    new: list[str] = []
    for _ in range(120):
        new = sorted(set(glob.glob(os.path.join(args.out, "*.trace.json*")))
                     - before)
        if new:
            time.sleep(3)
            new = sorted(set(glob.glob(os.path.join(args.out, "*.trace.json*")))
                         - before)
            if new:
                break
        time.sleep(2)
    if not new:
        print("[prof] NO TRACE PRODUCED", file=sys.stderr)
        return 1

    for path in new:
        print(f"\n=== trace {path} ({os.path.getsize(path)/1e6:.1f} MB)")
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            data = json.load(fh)
        evs = data["traceEvents"] if isinstance(data, dict) else data
        tot: dict[str, float] = {}
        cnt: dict[str, int] = {}
        gpu_total = 0.0
        cpu_total = 0.0
        cpu_tot: dict[str, float] = {}
        for e in evs:
            cat = e.get("cat")
            dur = float(e.get("dur", 0) or 0)
            name = e.get("name", "?")
            if cat in ("kernel", "gpu_kernel", "cuda_kernel"):
                tot[name] = tot.get(name, 0.0) + dur
                cnt[name] = cnt.get(name, 0) + 1
                gpu_total += dur
            elif cat in ("cpu_op", "cuda_runtime", "gpu_user_annotation"):
                cpu_tot[name] = cpu_tot.get(name, 0.0) + dur
                cpu_total += dur
        if not tot:
            print("[prof] trace has no kernel events "
                  f"(cats seen: {sorted({e.get('cat') for e in evs if e.get('cat')})[:8]})")
            continue
        print(f"[prof] GPU busy total {gpu_total/1e3:.2f} s over ~{args.tokens} "
              f"decode steps -> {gpu_total/1e3/args.tokens*1000:.2f} ms/step of "
              f"kernel time")
        print(f"{'self_ms':>9} {'%/step':>7} {'n':>8} {'us/call':>8}  name")
        for name, dur in sorted(tot.items(), key=lambda kv: -kv[1])[:args.top]:
            print(f"{dur/1e3:>9.2f} {dur/gpu_total*100/1:>6.1f}% "
                  f"{cnt[name]:>8} {dur/max(1,cnt[name]):>8.1f}  {name[:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
