# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-size-dispatched hybrid drafter: prompt lookup (ngram) + MTP.

Measured on the 27B SM70 build (4x V100, TP4, fp8 KV, same harness):

    MTP k=12    AL 13.00   8K 243.2 tok/s   64K 156.1   conc8 287.9
    ngram k=24  AL 25.00   8K 550.3 tok/s   64K 325.9   conc8 175.9

The prompt-lookup drafter is free (two Triton kernels, no model forward) and
its acceptance on lookup-friendly content is near perfect, but it collapses on
content whose next token does not appear in the history (AL 1.0 on random
tokens) and its history scan is O(num_reqs * max_model_len / block), so wide
batches pay for it twice.  MTP is the opposite: steady acceptance on any
content, but 1.77 ms per draft pass.

This speculator therefore keeps one child of each kind and dispatches per step:
small batches (latency-bound, where a free draft wins) go to prompt lookup,
wide batches go to MTP.  Steps that carry prefill tokens always go to MTP so
the draft model's own KV cache is populated for requests that may later be
served by MTP.
"""

import os

import torch

from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator

class NgramMTPHybridSpeculator(MTPSpeculator):
    """MTPSpeculator that hands small, decode-only batches to prompt lookup."""

    def __init__(self, vllm_config, device, req_states):
        super().__init__(vllm_config, device)
        # SpeculativeConfig zeroes prompt_lookup_min/max for every method other
        # than ngram (its final `else` branch), so with method="mtp" the ngram
        # child would assert on 1 <= 0 <= 0.  Restore the window here.
        spec = vllm_config.speculative_config
        if not spec.prompt_lookup_min:
            spec.prompt_lookup_min = int(
                os.environ.get("VLLM_SM70_HYBRID_LOOKUP_MIN", "2")
            )
        if not spec.prompt_lookup_max:
            spec.prompt_lookup_max = int(
                os.environ.get("VLLM_SM70_HYBRID_LOOKUP_MAX", "5")
            )
        if spec.prompt_lookup_min > spec.prompt_lookup_max:
            spec.prompt_lookup_min = spec.prompt_lookup_max
        self._ngram = NgramGPUSpeculator(vllm_config, device, req_states)
        self.ngram_max_reqs = int(
            os.environ.get("VLLM_SM70_HYBRID_NGRAM_MAX_REQS", "2")
        )

    def _use_ngram(self, args, kwargs) -> bool:
        if kwargs.get("dummy_run", False) or kwargs.get("is_profile", False):
            return False
        batch = kwargs.get("input_batch", args[0] if args else None)
        if batch is None or batch.num_reqs > self.ngram_max_reqs:
            return False
        k = kwargs.get("num_speculative_tokens") or self.num_speculative_steps
        # Decode steps schedule exactly k+1 tokens per request.  Anything else
        # (chunked prefill, mixed batches) must reach the MTP drafter so its KV
        # cache is filled for requests that may later be served by MTP.
        # num_scheduled_tokens is a host numpy array, so this costs no sync --
        # probing the GPU num_sampled tensor here cost ~8% of the step.
        sched = getattr(batch, "num_scheduled_tokens", None)
        if sched is None:
            return False
        width = int(k) + 1
        return bool((sched[: batch.num_reqs] == width).all())

    @torch.inference_mode()
    def propose(self, *args, **kwargs):
        if self._use_ngram(args, kwargs):
            drafts = self._ngram.propose(*args, **kwargs)
            k = kwargs.get("num_speculative_tokens")
            if k is not None and drafts.shape[1] > k:
                drafts = drafts[:, :k]
            return drafts
        return super().propose(*args, **kwargs)
