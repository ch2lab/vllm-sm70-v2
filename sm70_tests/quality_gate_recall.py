"""Long-context retrieval gate on a REAL corpus (no synthetic repetition).

Both earlier probes degenerated (`ductduct...`) because the filler was a small
set of sentences looped hundreds of times -- the model loops, and the measurement
says nothing about the window.  Here the haystack is the repo's own sm70 docs
(1.27M chars of varied technical Chinese, paragraphs shuffled with a seed).

Probe: a distinct city <-> 5-digit code record is planted at each sampled
position; the question asks for that city's code.  A dropped needle is
unanswerable (the code appears nowhere else), and an answer that repeats a
4-char n-gram more than 8 times is reported INVALID rather than FAIL, so model
degeneration is not confused with retrieval loss.
"""
import glob
import json
import os
import random
import sys
import urllib.request

from transformers import AutoTokenizer

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
TARGET_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 90000
KEY = open("/root/.vllm_api_key").read().strip()
TK = AutoTokenizer.from_pretrained("/data/models/Qwen3.6-35B-A3B-NVFP4-Fast",
                                   trust_remote_code=True)

CITIES = ["兰州", "宜昌", "柳州", "保定", "舟山", "绵阳", "赣州", "临汾"]


def corpus_paragraphs():
    paras = []
    for f in sorted(glob.glob("/data/src/vllm-v2/sm70/*.md")):
        for p in open(f, encoding="utf-8", errors="ignore").read().split("\n\n"):
            p = p.strip()
            if 80 <= len(p) <= 1200:
                paras.append(p)
    return paras


def build_doc(fracs, target_tokens, seed=11):
    rng = random.Random(seed)
    paras = corpus_paragraphs()
    rng.shuffle(paras)
    cities = rng.sample(CITIES, len(fracs))
    codes = rng.sample(range(10000, 99999), len(fracs))
    needles = [(f, c, k) for f, c, k in zip(fracs, cities, codes)]
    pending = list(needles)
    parts, ntok, i = [], 0, 0
    while ntok < target_tokens and i < len(paras):
        if pending and pending[0][0] * target_tokens <= ntok:
            frac, city, code = pending.pop(0)
            part = f"【附录】{city}的校验码是 {code}。"
        else:
            part = paras[i]
            i += 1
        parts.append(part)
        ntok += len(TK.encode(part, add_special_tokens=False))
    for frac, city, code in pending:          # flush anything left
        part = f"【附录】{city}的校验码是 {code}。"
        parts.append(part)
        ntok += len(TK.encode(part, add_special_tokens=False))
    return "\n\n".join(parts), needles, ntok


def degenerate(text: str) -> bool:
    for n in (4, 5):
        for i in range(len(text) - 8 * n):
            if text[i:i + n] * 8 == text[i:i + n * 8]:
                return True
    return False


def chat(prompt, max_tokens=16):
    body = {"model": os.environ.get("SM70_MODEL", "Qwen3.6-35B-A3B"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""


def main():
    print(f"base={BASE} target={TARGET_TOKENS}", flush=True)
    ok = bad = invalid = 0
    for label, fracs, tgt in (("control 8K", [0.05, 0.50, 0.92], 8000),
                              ("long", [0.05, 0.20, 0.35, 0.50, 0.65, 0.92],
                               TARGET_TOKENS)):
        doc, needles, ntok = build_doc(fracs, tgt)
        print(f"-- {label}: {ntok} tokens", flush=True)
        for frac, city, code in needles:
            ans = chat(doc + f"\n\n问题：{city}的校验码是多少？只回答数字。")
            if degenerate(ans):
                invalid += 1
                print(f"  [INVALID] {frac*100:5.1f}% {city} -> {ans.strip()[:34]!r}",
                      flush=True)
                continue
            hit = str(code) in ans
            ok += hit
            print(f"  [{'PASS' if hit else 'FAIL'}] {frac*100:5.1f}% {city} "
                  f"期望{code} -> {ans.strip()[:34]!r}", flush=True)
    n = ok + bad + invalid
    print(f"QUALITY recall={ok}/{n - invalid} invalid={invalid} (of {n})",
          flush=True)


if __name__ == "__main__":
    main()
