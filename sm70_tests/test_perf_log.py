#!/usr/bin/env python3
"""ASGI harness for RequestPerfMiddleware + perf_track (no GPU needed).

Simulates the engine-yield shapes each endpoint actually sees and asserts
header/log behavior:

1. json_cum  — non-streaming chat, output_kind=CUMULATIVE: per-step yields
               with growing token lists. Header must carry BOTH groups
               (prefill + decode) with decode.ms > 0; decode.tokens = final
               cumulative count.
2. delta_sse — streaming, output_kind=DELTA: per-step 1-token yields.
               Header at first chunk = prefill group only; the log line must
               show the ACCUMULATED output token count (sum of deltas), not
               the last delta.
3. final_json — output_kind=FINAL_ONLY: single yield with the full output.
               Header omits the decode group (single timestamp);
               output_tokens still correct.
4. tuple_json — /v1/completions shape: yields are (index, RequestOutput)
               tuples from merge_async_iterators. perf_track must not raise
               and must record counts from the tuple's output element.
5. plain — non-generation endpoint: no header.

Run: /data/vllm-dev/bin/python sm70_tests/test_perf_log.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starlette.datastructures import State  # noqa: E402

from vllm.entrypoints.serve.perf_log import (  # noqa: E402
    RequestPerfMiddleware,
    perf_track,
    set_perf_output_kind,
    set_perf_request_id,
)

STEP = 0.01  # fake per-engine-step delay


class _O:
    def __init__(self, n):
        self.token_ids = list(range(n))


class FakeRes:
    def __init__(self, n_prompt, n_out, cached=0):
        self.prompt_token_ids = list(range(n_prompt))
        self.num_cached_tokens = cached
        self.outputs = [_O(n_out)]


async def engine(out_len_per_step, cached=8):
    """Yield one FakeRes per fake engine step."""
    for n in out_len_per_step:
        await asyncio.sleep(STEP)
        yield FakeRes(32, n, cached)


def cum_engine():
    # CUMULATIVE: each yield carries the growing full list (1,2,3,4 tokens)
    return engine([1, 2, 3, 4])


def delta_engine():
    # DELTA: each yield carries only the new tokens (1 each)
    return engine([1, 1, 1, 1])


def final_only_engine():
    # FINAL_ONLY: exactly one yield, carrying the full output
    return engine([4])


def tuple_engine():
    # /v1/completions: (index, RequestOutput) tuples
    async def gen():
        i = 0
        async for res in delta_engine():
            yield i, res
            i += 1

    return gen()


class FakeApp:
    def __init__(self, kind):
        self.kind = kind

    async def __call__(self, scope, receive, send):
        state = scope["state"]
        kind = self.kind
        if kind == "json_cum":
            set_perf_request_id(state, "chatcmpl-cum1")
            set_perf_output_kind(state, "CUMULATIVE")
            async for _res in perf_track(cum_engine()):
                pass
            start_ct = b"application/json"
        elif kind == "delta_sse":
            set_perf_request_id(state, "chatcmpl-sse1")
            set_perf_output_kind(state, "DELTA")
            start_ct = b"text/event-stream"
        elif kind == "final_json":
            set_perf_request_id(state, "chatcmpl-final1")
            set_perf_output_kind(state, "FINAL_ONLY")
            async for _res in perf_track(final_only_engine()):
                pass
            start_ct = b"application/json"
        elif kind == "tuple_json":
            set_perf_request_id(state, "cmpl-tuple1")
            set_perf_output_kind(state, "DELTA")
            async for _pair in perf_track(tuple_engine()):
                pass
            start_ct = b"application/json"
        else:
            start_ct = b"text/plain"

        if kind == "delta_sse":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", start_ct)],
                }
            )
            i = 0
            async for _res in perf_track(delta_engine()):
                i += 1
                await send(
                    {
                        "type": "http.response.body",
                        "body": f'data: {{"i": {i}}}\n\n'.encode(),
                        "more_body": i < 4,
                    }
                )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"data: [DONE]\n\n",
                    "more_body": False,
                }
            )
            return

        body = json.dumps({"id": f"r-{kind}"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", start_ct)],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )


async def call(app, path):
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "http_version": "1.1",
        "client": ("127.0.0.1", 54321),
        "scheme": "http",
        # Reproduce the live service: scope["state"] is a Starlette State
        # (attribute bag, no .get), not a plain dict.
        "state": State(),
    }
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(msg):
        sent.append(msg)

    await RequestPerfMiddleware(app)(scope, receive, send)
    return sent


def header_of(sent, name):
    start = next(m for m in sent if m["type"] == "http.response.start")
    for k, v in start["headers"]:
        if k.lower() == name:
            return v.decode()
    return None


def main():
    import vllm.entrypoints.serve.perf_log as pl

    lines = []
    h = logging.Handler()
    h.emit = lambda r: lines.append(r.getMessage())
    pl.logger.addHandler(h)
    pl.logger.setLevel(logging.INFO)

    # 1) CUMULATIVE non-streaming: both groups, real decode timing
    sent = asyncio.run(call(FakeApp("json_cum"), "/v1/chat/completions"))
    hdr = header_of(sent, b"x-request-perf")
    d = json.loads(hdr)
    assert set(d.keys()) == {"prefill", "decode"}, d
    assert d["prefill"]["tokens"] == 32 and d["prefill"]["cached"] == 8, d
    assert d["decode"]["tokens"] == 4, f"final cumulative count: {d}"
    assert d["decode"]["ms"] > 0, f"decode.ms must be > 0: {d}"
    assert d["decode"]["tps"] > 0, d
    print("1 json_cum header:", hdr)

    # 2) DELTA streaming: prefill-only header, accumulated count in log
    sent = asyncio.run(call(FakeApp("delta_sse"), "/v1/chat/completions"))
    hdr = header_of(sent, b"x-request-perf")
    d = json.loads(hdr)
    assert set(d.keys()) == {"prefill"}, f"decode group must be omitted: {d}"
    bodies = [m["body"] for m in sent if m["type"] == "http.response.body"]
    assert bodies[-1] == b"data: [DONE]\n\n"
    assert all(not b.startswith(b":") for b in bodies)
    assert len(bodies) == 5
    print("2 delta_sse header:", hdr)

    # 3) FINAL_ONLY: decode group omitted, count still full
    sent = asyncio.run(call(FakeApp("final_json"), "/v1/chat/completions"))
    hdr = header_of(sent, b"x-request-perf")
    d = json.loads(hdr)
    assert set(d.keys()) == {"prefill"}, d
    print("3 final_json header:", hdr)

    # 4) tuple yields (completions): no exception, counts recorded
    sent = asyncio.run(call(FakeApp("tuple_json"), "/v1/completions"))
    hdr = header_of(sent, b"x-request-perf")
    d = json.loads(hdr)
    assert d["prefill"]["tokens"] == 32, d
    assert d["decode"]["tokens"] == 4, f"tuple output count: {d}"
    print("4 tuple_json header:", hdr)

    # 5) plain: no header
    sent = asyncio.run(call(FakeApp("plain"), "/health"))
    assert header_of(sent, b"x-request-perf") is None
    print("5 plain: no header OK")

    assert len(lines) == 5, f"expected 5 log lines, got {len(lines)}"
    by_id = {}
    for l in lines:
        for rid in ("cum1", "sse1", "final1", "tuple1"):
            if f"chatcmpl-{rid}" in l or f"cmpl-{rid}" in l:
                by_id[rid] = l
    cum_line = by_id["cum1"]
    sse_line = by_id["sse1"]
    final_line = by_id["final1"]
    tuple_line = by_id["tuple1"]
    assert "output 4 tokens" in cum_line and "decode 0 ms" not in cum_line, cum_line
    assert "output 4 tokens" in sse_line, f"DELTA must accumulate: {sse_line}"
    assert "output 1 tokens" not in sse_line, sse_line
    assert "output" not in final_line, f"FINAL_ONLY: no decode clause: {final_line}"
    assert "output 4 tokens" in tuple_line, tuple_line
    print("LOG LINES:")
    for l in lines:
        print("  ", l)
    print("ALL OK")


if __name__ == "__main__":
    main()
