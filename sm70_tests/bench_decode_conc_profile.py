"""conc8 decode-dominated profiled capture.

Fires 8 concurrent streaming completions (2048-token salted prompts, 512
tokens). Once all 8 have emitted their first token (prefill done, steady
decode), /start_profile, sample ~4 s, /stop_profile. Reports per-stream ITL
and aggregate tok/s from stream timings, then the trace dir for attribution.
Requires VLLM_TORCH_PROFILER_DIR in the service env (else /start_profile 404).
"""
import json
import statistics
import threading
import os
import time
import urllib.request

BASE = "http://127.0.0.1:8001"
KEY = open("/root/.vllm_api_key").read().strip()
CONC = int(os.environ.get("CONC", "8"))
PROMPT_TOK = 2048
MAX_TOK = 512
WINDOW_S = 4.0


def post(path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def haystack(tokens, salt):
    rng = (salt * 2654435761) & 0xFFFFFFFF
    words = []
    for _ in range(int(tokens * 1.4)):
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        words.append("tok%06d" % (rng % 1000000))
    return " ".join(words)


res = {}
firsts = {}
lock = threading.Lock()
all_first = threading.Event()


def worker(i, prompt):
    body = json.dumps({"model": "Qwen3.6-35B-A3B", "prompt": prompt,
                       "max_tokens": MAX_TOK, "temperature": 0,
                       "stream": True}).encode()
    req = urllib.request.Request(
        BASE + "/v1/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    first = last = None
    n = 0
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            s = line.decode().strip()
            if not s.startswith("data: "):
                continue
            if s[6:] == "[DONE]":
                break
            j = json.loads(s[6:])
            ch = j.get("choices") or [{}]
            if ch[0].get("text"):
                now = time.time()
                if first is None:
                    first = now
                    with lock:
                        firsts[i] = now
                        if len(firsts) == CONC:
                            all_first.set()
                last = now
                n += 1
    res[i] = (first, last, n)


prompts = [haystack(PROMPT_TOK, 9502 + 17 * i) for i in range(CONC)]
ths = [threading.Thread(target=worker, args=(i, prompts[i])) for i in range(CONC)]
t_start = time.time()
for t in ths:
    t.start()
ok = all_first.wait(timeout=120)
print("all-first after %.2fs (ok=%s)" % (time.time() - t_start, ok))
post("/start_profile", {}, timeout=30)
time.sleep(WINDOW_S)
try:
    post("/stop_profile", {}, timeout=60)
except Exception as e:
    print("stop_profile warn:", repr(e)[:120])
for t in ths:
    t.join(timeout=600)

itls = []
for i in sorted(res):
    f, l, n = res[i]
    if f and l and n > 2:
        itls.append((l - f) / (n - 1))
itl = statistics.median(itls)
print("per-stream ITL median=%.3f ms  aggregate=%.1f tok/s  (n=%d streams)"
      % (itl * 1000, CONC / itl, len(itls)))
print("DONE_CONC_PROFILE")
