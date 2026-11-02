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
        self.every = max(1, int(os.environ.get("VLLM_OUTPUT_WATCHDOG_EVERY", "8")))
        self.run_len = max(
            4, int(os.environ.get("VLLM_OUTPUT_WATCHDOG_RUN", "24"))
        )
        self.max_logs = int(os.environ.get("VLLM_OUTPUT_WATCHDOG_LOGS", "50"))
        self._tick = 0
        self._params: dict[int, tuple] = {}
        self.logs = 0
        self.nan_hits = 0
        self.run_hits = 0
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

            finite = torch.isfinite(logits).all(dim=1)
            if not bool(finite.all()):
                bad_rows = (~finite).nonzero(as_tuple=True)[0]
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
                for r in hit.nonzero(as_tuple=True)[0].tolist()[:4]:
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
