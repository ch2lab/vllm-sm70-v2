# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request performance log + ``X-Request-Perf`` response header.

Re-implementation of the 1Cat "X-Request-Perf" feature (the original source
was lost with the 1CatVllm backup; spec recovered from project memory).
Enabled by default in this tree; the SM70 serve script sets
``--disable-uvicorn-access-log`` so this middleware is the only access-log
source (one line per request, SM120 format):

    client - "POST /v1/chat/completions HTTP/1.1" 200 OK | <request_id>:
        input N tokens(prefill X ms, Y tok/s, cache hit: Z%),
        output N tokens(decode X ms, Y tok/s)

Where the perf info lands:
- non-streaming: X-Request-Perf header (full field set; generation is
  complete when the response starts) + end-of-response log line
- streaming (SSE): X-Request-Perf header at first-token time with the
  prefill fields only (decode fields not yet available -- omitted, never
  output as 0) + end-of-stream log line with the full field set. The
  stream body is byte-identical to upstream (no frames injected).

Header rule: fields that are not yet known when the header is sent are
OMITTED from the JSON (never output as 0/null).

Plumbing: the middleware opens the perf record on the ASGI scope at request
entry; endpoint handlers record the request id and wrap their engine result
generators with :func:`perf_track` (one line per loop); endpoints without a
wrapped loop call :func:`record_final_perf` with the final RequestOutput.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_perf_ctx: ContextVar[dict[str, Any] | None] = ContextVar(
    "vllm_sm70_perf", default=None
)

_REASON_PHRASES = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    409: "Conflict",
    413: "Payload Too Large",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


def begin_perf_state(scope: dict[str, Any]) -> dict[str, Any]:
    """Open the per-request perf record on the ASGI scope (middleware entry).

    The record lives in ``scope["state"]["perf"]`` so every layer (middleware,
    endpoint handler, response) sees the same dict for this request.
    """
    state = scope.setdefault("state", {})
    perf: dict[str, Any] = {
        "t0": time.perf_counter(),
        "request_id": None,
        "t_first": None,
        "t_last": None,
        "prompt_tokens": None,
        "cached_tokens": None,
        "output_tokens": None,
    }
    state["perf"] = perf
    _perf_ctx.set(perf)
    return perf


def _perf_from_state(state: Any) -> dict[str, Any] | None:
    """Perf record from an ASGI scope / Starlette request state.

    The container is a plain dict when this middleware opened it, but a
    Starlette State object (attribute bag, no ``.get``) when Starlette
    created one for the request. Both must work.
    """
    if state is None:
        return None
    if hasattr(state, "get"):  # plain dict
        return state.get("perf")
    return getattr(state, "perf", None)  # Starlette State


def set_perf_request_id(state: Any, request_id: str) -> None:
    """Endpoint handler records the (client-visible) request id.

    No-op when no record was opened. Prefers the context-var record (set by
    the middleware for this request) — the same dict the middleware closes
    over, regardless of the state container type.
    """
    perf = _perf_ctx.get()
    if perf is None:
        perf = _perf_from_state(state)
    if perf is not None:
        perf["request_id"] = request_id


def set_perf_output_kind(state: Any, output_kind: Any) -> None:
    """Endpoint records the engine output_kind for this request (DELTA vs
    CUMULATIVE vs FINAL_ONLY) so output-token counts accumulate correctly.

    No-op when no record was opened or no kind is available (e.g. beam
    search params).
    """
    perf = _perf_ctx.get()
    if perf is None:
        perf = _perf_from_state(state)
    if perf is not None and output_kind is not None:
        perf["output_kind"] = getattr(output_kind, "name", str(output_kind))


def _as_request_output(res: Any) -> Any:
    """Unwrap (index, RequestOutput) tuples from merge_async_iterators."""
    if isinstance(res, tuple) and len(res) == 2:
        return res[1]
    return res


def _update_from_output(
    res: Any, perf: dict[str, Any], with_timing: bool
) -> None:
    res = _as_request_output(res)
    if with_timing:
        now = time.perf_counter()
        if perf["t_first"] is None:
            perf["t_first"] = now
        perf["t_last"] = now
    if res.prompt_token_ids is not None:
        perf["prompt_tokens"] = len(res.prompt_token_ids)
    if res.num_cached_tokens is not None:
        perf["cached_tokens"] = res.num_cached_tokens
    if res.outputs:
        # DELTA yields carry only the new tokens -> accumulate; CUMULATIVE /
        # FINAL_ONLY carry the full (growing) token list -> replace.
        if perf.get("output_kind") == "DELTA":
            perf["output_tokens"] = (
                (perf.get("output_tokens") or 0)
                + sum(len(o.token_ids or []) for o in res.outputs)
            )
        else:
            perf["output_tokens"] = sum(len(o.token_ids or []) for o in res.outputs)


async def perf_track(gen: AsyncIterator) -> AsyncIterator:
    """Wrap an engine result generator, recording first/last output timing and
    token counts into the active perf record.

    Tolerates (index, RequestOutput) tuples from merge_async_iterators
    (/v1/completions) — the tuple is passed through untouched and timing/
    counts are recorded from the RequestOutput element.

    Pass-through when no record is active (non-OpenAI paths).
    """
    perf = _perf_ctx.get()
    if perf is None:
        async for res in gen:
            yield res
        return
    async for res in gen:
        _update_from_output(res, perf, with_timing=True)
        yield res


def record_final_perf(final_res: Any) -> None:
    """Endpoints without a perf_track-wrapped engine loop (e.g.
    /v1/responses) record the final output. Token counts only -- timing
    fields stay None and are omitted from the header/log line.
    """
    perf = _perf_ctx.get()
    if perf is not None and final_res is not None:
        _update_from_output(final_res, perf, with_timing=False)


def _perf_fields(perf: dict[str, Any]) -> dict[str, Any] | None:
    """Fields known so far; unknowns omitted. None when nothing usable.

    Decode fields only appear once at least one output has arrived AFTER
    the first (``t_last > t_first``) -- so the first-token SSE header
    carries the prefill fields only.
    """
    t_first = perf.get("t_first")
    if t_first is None:
        return None
    t0 = perf["t0"]
    t_last = perf.get("t_last")
    prompt_tokens = perf.get("prompt_tokens")
    output_tokens = perf.get("output_tokens")
    prefill_ms = (t_first - t0) * 1000.0
    decode_ms = (
        (t_last - t_first) * 1000.0 if t_last is not None and t_last > t_first
        else None
    )
    # Nested format (user ruling): prefill/decode groups. Within a group:
    # "ms" = wall time (1 decimal), "tokens"/"cached" = counts, "tps" =
    # throughput (tokens/s, int). Unknown fields omitted (never 0/null).
    fields: dict[str, Any] = {}
    if prompt_tokens is not None:
        prefill: dict[str, Any] = {
            "ms": round(prefill_ms, 1),
            "tokens": prompt_tokens,
            "cached": perf.get("cached_tokens") or 0,
        }
        if prefill_ms > 0:
            prefill["tps"] = round(prompt_tokens / (prefill_ms / 1000.0))
        fields["prefill"] = prefill
    else:
        fields["prefill"] = {"ms": round(prefill_ms, 1)}
    if output_tokens is not None and decode_ms is not None:
        fields["decode"] = {
            "ms": round(decode_ms, 1),
            "tokens": output_tokens,
            "tps": round(output_tokens / (decode_ms / 1000.0)),
        }
    return fields


def build_perf_header(perf: dict[str, Any] | None) -> str | None:
    """X-Request-Perf header value, or None when nothing is known yet."""
    if perf is None:
        return None
    fields = _perf_fields(perf)
    if not fields:
        return None
    return json.dumps(fields, separators=(",", ":"))


def format_perf_log_line(
    client: str,
    method: str,
    path: str,
    http_version: str,
    status: int,
    perf: dict[str, Any] | None,
) -> str:
    """SM120-format per-request line; bare line when no perf data (errors,
    non-generation endpoints)."""
    reason = _REASON_PHRASES.get(status, "")
    line = (
        f'{client} - "{method} {path} HTTP/{http_version}" {status} {reason}'
    ).rstrip()
    if perf is None or perf.get("t_first") is None:
        return line
    t0 = perf["t0"]
    t_first = perf["t_first"]
    t_last = perf.get("t_last")
    prompt_tokens = perf.get("prompt_tokens") or 0
    output_tokens = perf.get("output_tokens") or 0
    cached_tokens = perf.get("cached_tokens") or 0
    prefill_ms = (t_first - t0) * 1000.0
    decode_ms = (
        (t_last - t_first) * 1000.0
        if t_last is not None and t_last > t_first
        else None
    )
    prefill_tps = prompt_tokens / (prefill_ms / 1000.0) if prefill_ms > 0 else 0.0
    cache_hit = (
        cached_tokens / prompt_tokens * 100.0 if prompt_tokens > 0 else 0.0
    )
    detail = (
        f" | request_id={perf.get('request_id')}: "
        f"input {prompt_tokens} tokens(prefill {prefill_ms:.0f} ms, "
        f"{prefill_tps:.0f} tok/s, cache hit: {cache_hit:.0f}%)"
    )
    if output_tokens and decode_ms is not None:
        decode_tps = (
            output_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
        )
        detail += (
            f", output {output_tokens} tokens(decode {decode_ms:.0f} ms, "
            f"{decode_tps:.0f} tok/s)"
        )
    return line + detail


class RequestPerfMiddleware:
    """Outermost ASGI middleware: injects X-Request-Perf (fields known at
    header-send time only) and logs one perf line per request.

    Must be registered LAST in ``register.py`` so it is the outermost layer
    and sees every response. Non-streaming (JSON) responses get the full
    header at response start (generation is already complete); streaming
    (SSE) responses defer header emission to the first body chunk (first
    token). Response bodies are never modified.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        perf = begin_perf_state(scope)
        client = scope.get("client")
        client_str = f"{client[0]}:{client[1]}" if client else "-"
        method = scope.get("method", "-")
        path = scope.get("path", "-")
        http_version = scope.get("http_version", "1.1")
        status = 0
        # Start message held for responses whose header is deferred until
        # the first body chunk.
        pending_start: dict | None = None

        def attach_perf_header(message: dict, header: str | None) -> dict:
            if header is None:
                return message
            headers = [
                *message["headers"],
                (b"x-request-perf", header.encode("utf-8")),
            ]
            return {**message, "headers": headers}

        async def send_perf(message: dict) -> None:
            nonlocal pending_start, status
            mtype = message["type"]
            if mtype == "http.response.start":
                status = message["status"]
                content_type = b""
                for key, value in message["headers"]:
                    if key.lower() == b"content-type":
                        content_type = value
                if b"application/json" in content_type:
                    # Non-streaming: generation is complete at response start
                    # -> full header now.
                    header = build_perf_header(perf)
                    await send(attach_perf_header(message, header))
                else:
                    # Streaming (SSE) or unknown: defer to the first body
                    # chunk so the header carries first-token timing.
                    pending_start = message
                return
            if mtype != "http.response.body":
                if pending_start is not None:
                    # Unexpected message before any body: release the held
                    # start without a header.
                    await send(pending_start)
                    pending_start = None
                await send(message)
                return
            if pending_start is not None:
                header = build_perf_header(perf)
                await send(attach_perf_header(pending_start, header))
                pending_start = None
            await send(message)

        try:
            await self.app(scope, receive, send_perf)
        finally:
            # Bodyless response (e.g. 204): never saw a body message.
            if pending_start is not None:
                try:
                    await send(pending_start)
                except Exception:  # pragma: no cover - best effort
                    pass
                pending_start = None
            # One perf line per request, always (status 0 = no response at
            # all, e.g. handler crashed before sending anything).
            logger.info(
                "%s",
                format_perf_log_line(
                    client_str,
                    method,
                    path,
                    http_version,
                    status,
                    perf,
                ),
            )
