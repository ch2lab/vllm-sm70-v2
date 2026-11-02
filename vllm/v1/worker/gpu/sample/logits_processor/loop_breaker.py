# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampler-side loop breaker for verbatim self-repetition (V2 runner).

``Qwen3.6-35B-A3B`` can degenerate into repeating a token window verbatim when
a client samples without repetition penalties (the exported-session failure
mode: a ~1300-token section repeated, single sentences repeated 16-22x).
Client-side parameter fixes help only the clients that send them; this
processor fixes it in the engine for every client, and only while a loop is
actually present.

Detection is period-agnostic: the last ``window`` tokens are searched for
anywhere earlier in the tail. A loop of period ``P`` is visible when
``P <= tail - window``, so two scales are used:

* fast level: ``tail`` (default 2048) checked every ``every`` (16) steps;
* coarse level: ``coarse_tail`` (default 8192) checked every
  ``coarse_every`` (128) steps - catches long-period loops at ~3 us/step.

The search is O(tail) instead of O(tail * window): four anchor positions of
the trailing window must match at the same offset, then the best candidate is
verified exactly. Everything runs on device - there is no host sync and the
token history is never copied to the CPU.

Action when a loop is detected: the token that would extend it gets a one-step
near-ban (``penalty``, default 15 raw logits) and every token of the repeated
window gets a presence-style bias (``window_penalty``, default 0.5) that
accumulates while the loop persists. The condition is self-limiting - once the
text diverges the windows stop matching - so no per-request cooldown state is
needed.

Enable with ``--logits-processors
vllm.v1.worker.gpu.sample.logits_processor.loop_breaker:LoopBreakerLogitsProcessor``
(the serve script does this for ``VLLM_SM70_LOOP_BREAKER=1``). Off by default.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


class LoopBreakerLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState
    ):
        self.req_states = req_states
        # L: length of the repeated unit that must be found.
        self.window = int(os.environ.get("VLLM_LOOP_BREAKER_WINDOW", "64"))
        # M: how far back the fast level looks (visible period <= M - L).
        self.tail = int(os.environ.get("VLLM_LOOP_BREAKER_TAIL", "2048"))
        # Coarse level: long-period loops, checked less often.
        self.coarse_tail = int(
            os.environ.get("VLLM_LOOP_BREAKER_COARSE_TAIL", "8192")
        )
        self.coarse_every = max(
            1, int(os.environ.get("VLLM_LOOP_BREAKER_COARSE_EVERY", "128"))
        )
        # Raw-logit bias on the token that would extend the loop. A tight loop
        # puts ~0.999 on it, so -15 is a one-step near-ban.
        self.penalty = float(os.environ.get("VLLM_LOOP_BREAKER_PENALTY", "15.0"))
        self.window_penalty = float(
            os.environ.get("VLLM_LOOP_BREAKER_WINDOW_PENALTY", "0.5")
        )
        # Check every k steps: a loop survives many tokens, and skipping steps
        # keeps the per-step cost off the critical path.
        # 16, not 1: a loop persists for hundreds of tokens, so checking more
        # often only adds launch overhead (~300 us per check).
        self.every = max(1, int(os.environ.get("VLLM_LOOP_BREAKER_EVERY", "16")))
        self.log_fires = os.environ.get("VLLM_LOOP_BREAKER_LOG", "0") == "1"
        self.fires = 0
        self.calls = 0
        self._tick = 0
        self._offsets: dict[tuple[int, str], torch.Tensor] = {}
        self._table: dict[tuple[int, str], torch.Tensor] = {}

    def add_request(self, req_idx: int, sampling_params: "SamplingParams") -> bool:
        return True

    def _arange(self, n: int, device: torch.device) -> torch.Tensor:
        key = (n, str(device))
        t = self._table.get(key)
        if t is None:
            t = torch.arange(n, dtype=torch.int64, device=device)
            self._table[key] = t
        return t

    def _offsets_for(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, str(device))
        t = self._offsets.get(key)
        if t is None:
            t = torch.arange(-length, 0, dtype=torch.int64, device=device)
            self._offsets[key] = t
        return t

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        self._tick += 1
        if self._tick % self.every:
            return logits
        coarse = (self._tick % self.coarse_every) == 0
        m = self.coarse_tail if coarse else self.tail
        device = logits.device
        rows = ctx.expanded_idx_mapping.to(device)
        if rows.numel() == 0:
            return logits
        self.calls += 1
        w = self.window
        total_len = self.req_states.total_len.gpu[rows].to(torch.int64)
        prompt_len = self.req_states.prompt_len.gpu[rows].to(torch.int64)
        gen_len = total_len - prompt_len
        idx = (total_len.unsqueeze(1) + self._offsets_for(m, device)).clamp_(min=0)
        tail = self.req_states.all_token_ids.gpu[rows.unsqueeze(1), idx]

        last = tail[:, -w:]
        span = m - w + 1
        cand = tail[:, :span] == last[:, :1]
        step = max(1, w // 4)
        for off in range(step, w, step):
            cand &= tail[:, off : off + span] == last[:, off : off + 1]
        # The trailing window trivially matches itself: only *earlier*
        # occurrences count.
        cand[:, -w:] = False
        first = cand.to(torch.uint8).argmax(dim=1)
        recent = tail.gather(1, (first.unsqueeze(1) + self._arange(w, device)).clamp_(max=m - 1))
        exact = (recent == last).all(dim=1) & cand.any(dim=1)
        fire = exact & (gen_len >= 2 * w)
        follow = tail.gather(1, (first + w).clamp_(max=m - 1).unsqueeze(1))

        scale = fire.to(logits.dtype)
        idx_all = torch.cat([follow, recent], dim=1).to(torch.int64)
        vals = torch.cat(
            [
                (-self.penalty * scale).unsqueeze(1),
                (-self.window_penalty * scale).unsqueeze(1).expand(-1, w),
            ],
            dim=1,
        ).contiguous()
        logits.scatter_add_(1, idx_all, vals)

        if self.log_fires:
            n = int(fire.sum().item())
            if n:
                self.fires += 1
                if self.fires <= 6:
                    logger.info(
                        "loop breaker FIRED calls=%d fires=%d coarse=%s rows=%d",
                        self.calls, self.fires, coarse, n,
                    )
            elif self.calls % 400 == 0:
                logger.info(
                    "loop breaker: calls=%d fires=%d coarse=%s (no loop)",
                    self.calls, self.fires, coarse,
                )
        return logits
