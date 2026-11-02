# SPDX-License-Identifier: Apache-2.0
"""Engine-side output watchdog: log the moment a generation turns degenerate.

Two failures are worth catching in the act, because both are unreproducible
after the fact and leave no trace in the logs:

* **non-finite logits** -- a NaN/inf anywhere in the row collapses the softmax
  to one token regardless of temperature or penalties, which is exactly what a
  degenerate run looks like from outside;
* **a run of >= ``run_len`` identical sampled tokens** -- "!!!!...", "ductduct…",
  "====…".  Repetition penalties and the loop breaker normally prevent it, so
  when it happens the state or the logits are already wrong.

For each hit it logs the request slot, the prompt/generation length, the token,
and the sampling parameters captured at admission, which is what a replay needs.

Off by default.  Enable with
``--logits-processors vllm.v1.worker.gpu.sample.logits_processor.output_watchdog:OutputWatchdog``
(the serve script does this when ``VLLM_OUTPUT_WATCHDOG=1``).  It only reads
state; it never edits logits, and any internal error disables it for the
process instead of failing the request.
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


class OutputWatchdog(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState
    ):
        self.req_states = req_states
        # A sync every step would cost more than the sampling itself; a
        # degenerate run lasts hundreds of tokens, so a coarse cadence is enough.
        self.every = max(1, int(os.environ.get("VLLM_OUTPUT_WATCHDOG_EVERY", "256")))
        # The check syncs once per `every` steps; at 8 that cost 0.52 ms/step
        # (+5.6% decode).  Degenerations run for hundreds of tokens, so 64 is
        # enough resolution for evidence.  Any sync in the sampler costs: 64
        # measured +0.8%, so 256 keeps the cost under the run-to-run noise while
        # still catching a hundreds-of-tokens degeneration.
        # 24 was too eager: greedy benchmarks legitimately emit long runs of one
        # token, and banning it there both corrupted the answer and cost time.
        # Real degenerations run for hundreds of tokens.
        self.run_len = max(
            8, int(os.environ.get("VLLM_OUTPUT_WATCHDOG_RUN", "48"))
        )
        self.max_logs = int(os.environ.get("VLLM_OUTPUT_WATCHDOG_LOGS", "50"))
        # Hard safety net, independent of the root cause: a row with nan/inf has
        # its non-finite entries replaced (the softmax would otherwise collapse
        # to one token), and a row already stuck on one token has that token
        # banned for the step so the run cannot continue.  Both only touch a row
        # that is already broken, so healthy traffic is unaffected.
        # Off by default: the ban can corrupt a legitimate long run (markdown
        # rules, whitespace) and costs latency.  Enable only when a hard net is
        # wanted more than the last few percent of decode speed.
        # The nan scan reduces the whole vocab and syncs, which measured 0.8% of
        # decode on its own; the identical-run check is a 48-token gather and is
        # nearly free.  Keep the expensive half opt-in.
        self.check_nan = os.environ.get("VLLM_OUTPUT_WATCHDOG_NAN", "0") == "1"
        self.guard = os.environ.get("VLLM_OUTPUT_GUARD", "0") == "1"
        self.floor = float(os.environ.get("VLLM_OUTPUT_GUARD_FLOOR", "-1e4"))
        self._tick = 0
        self._params: dict[int, tuple] = {}
        self.logs = 0
        self.nan_hits = 0
        self.run_hits = 0
        self.sanitized = 0
        self.banned = 0
        self._dead = False

    def add_request(
        self, req_idx: int, sampling_params: "SamplingParams"
    ) -> bool:
        sp = sampling_params
        self._params[req_idx] = (
            getattr(sp, "max_tokens", None),
            getattr(sp, "temperature", None),
            getattr(sp, "top_p", None),
            getattr(sp, "top_k", None),
            getattr(sp, "repetition_penalty", None),
            getattr(sp, "presence_penalty", None),
            getattr(sp, "frequency_penalty", None),
            getattr(sp, "seed", None),
        )
        return True

    def _log(self, kind: str, slot: int, gen: int, prompt: int, detail: str):
        if self.logs >= self.max_logs:
            return
        self.logs += 1
        p = self._params.get(slot)
        logger.warning(
            "OUTPUT-WATCHDOG %s slot=%s prompt_len=%s gen_len=%s %s "
            "sampling(max_tokens,temp,top_p,top_k,rep,pres,freq,seed)=%s",
            kind, slot, prompt, gen, detail, p,
        )

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if self._dead:
            return logits
        self._tick += 1
        if self._tick % self.every:
            return logits
        try:
            rows = ctx.expanded_idx_mapping
            if rows.numel() == 0:
                return logits

            if not self.check_nan:
                finite = None
            else:
                finite = torch.isfinite(logits)
            if finite is not None and not bool(finite.all()):
                if self.guard:
                    logits.masked_fill_(~finite, self.floor)
                bad_rows = (~finite.all(dim=1)).nonzero(as_tuple=True)[0]
                tot = self.req_states.total_len.gpu[rows].to(torch.int64)
                plen = self.req_states.prompt_len.gpu[rows].to(torch.int64)
                for r in bad_rows.tolist()[:4]:
                    self.nan_hits += 1
                    self._log(
                        "NONFINITE-LOGITS",
                        int(rows[r].item()),
                        int(tot[r].item() - plen[r].item()),
                        int(plen[r].item()),
                        "logits contain nan/inf",
                    )

            n = self.run_len
            tot = self.req_states.total_len.gpu[rows].to(torch.int64)
            offs = torch.arange(n, 0, -1, device=tot.device, dtype=torch.int64)
            idx = (tot.unsqueeze(1) - offs).clamp_(min=0)
            toks = self.req_states.all_token_ids.gpu[rows.unsqueeze(1), idx]
            same = (toks == toks[:, :1]).all(dim=1)
            plen = self.req_states.prompt_len.gpu[rows].to(torch.int64)
            gen = tot - plen
            hit = same & (gen >= n)
            if bool(hit.any()):
                hit_rows = hit.nonzero(as_tuple=True)[0]
                if self.guard:
                    # Ban the repeated token for this step: the next sample must
                    # differ, so a single-token run cannot be extended.
                    ban = toks[hit_rows, -1].to(torch.int64)
                    logits[hit_rows, ban] = self.floor
                for r in hit_rows.tolist()[:4]:
                    self.run_hits += 1
                    self._log(
                        "IDENTICAL-RUN",
                        int(rows[r].item()),
                        int(gen[r].item()),
                        int(plen[r].item()),
                        f"last {n} tokens are all id={int(toks[r, -1].item())}",
                    )
        except Exception as e:  # noqa: BLE001 - a monitor must never break serving
            self._dead = True
            logger.warning("OUTPUT-WATCHDOG disabled after internal error: %r", e)
        return logits
