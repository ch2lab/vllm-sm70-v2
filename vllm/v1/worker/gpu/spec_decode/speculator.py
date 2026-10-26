# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.watermarking import create_watermarker
from vllm.v1.watermarking.spec_decode import (
    DraftWatermarker,
    create_speculative_draft_watermarker,
)
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
)
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

logger = init_logger(__name__)


def _target_feeds_hc_residual(vllm_config: VllmConfig) -> bool:
    """Whether the target replaces the drafter's input with its HC residual.

    Keyed on the same hook the model runner calls to perform that swap. It is
    resolved from the target model class because speculators are built before
    the target model is instantiated.
    """
    from vllm.model_executor.model_loader.utils import get_model_cls

    target_cls = get_model_cls(vllm_config.model_config)
    return hasattr(target_cls, "get_mtp_target_hidden_states")


class BaseSpeculator(ABC):
    num_query_per_req: int = 1
    # Extra query slots reserved per request outside the regular queries.
    num_extra_query_per_req: int = 0

    @abstractmethod
    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    @abstractmethod
    def capture(self) -> None:
        pass

    @abstractmethod
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [num_prefill_lookahead, max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        pass


class DraftModelSpeculator(BaseSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.draft_max_seq_len = self.max_model_len
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        # Widen for HC-multiplexed residuals: a target that implements
        # get_mtp_target_hidden_states() (e.g. DeepSeek V4) hands the drafter
        # its pre-hc_head (T, hc_mult * hidden_size) residual in place of the
        # collapsed hidden states, so the drafter's buffers must match. Key off
        # that hook rather than hc_mult alone -- HY V4 runs iHC in its backbone
        # (hc_mult=4) but its MTP head consumes the collapsed states, so
        # widening it feeds propose() a 4x-too-wide buffer.
        if _target_feeds_hc_residual(vllm_config):
            hc_mult = getattr(self.draft_model_config.hf_config, "hc_mult", 1)
            self.hidden_size = self.hidden_size * hc_mult
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype
        self.use_fp64_gumbel = vllm_config.model_config.use_fp64_gumbel
        self.use_local_argmax_reduction = (
            self.speculative_config.use_local_argmax_reduction
        )
        # SM70 testbed valve: decouple the online acceptance estimator from
        # adaptive verification (which needs varlen verify, unsupported by
        # UNIFORM_BATCH cudagraph backends). When on, the estimator runs to
        # feed runtime k selection; full draft logits are required.
        self.sm70_adaptive_k = (
            os.environ.get("VLLM_SM70_MTP_ADAPTIVE_K", "")
            in ("1", "true", "True")
        )
        if self.sm70_adaptive_k:
            # V3: self-calibrating uniform-K selection over {0, 1, k_full}.
            # score(k) = AL_hat(k) / step_ms[k], where step_ms[k] is an EMA of
            # the measured step wall time at each K on THIS engine's own
            # execution (cross-config cost constants are not trustworthy: the
            # k=1 step under a k=4-configured engine measured 20 ms vs 17 ms
            # for the native k=1 config at 250K ctx). k=0 (no speculation,
            # verify width 1) is the floor candidate — its score equals the
            # no-MTP rate by construction. Initial EMAs are conservative.
            self._ak_depth = self.num_speculative_steps
            self._ak_target: int | None = None
            self._ak_streak = 0
            self._ak_hyst = int(os.environ.get("VLLM_SM70_AK_HYST", "3"))
            self._ak_margin = float(os.environ.get("VLLM_SM70_AK_MARGIN", "1.10"))
            self._ak_step_ms = {
                0: float(os.environ.get("VLLM_SM70_AK_STEP0_MS", "12.0")),
                1: float(os.environ.get("VLLM_SM70_AK_STEP1_MS", "20.0")),
                self.num_speculative_steps: float(
                    os.environ.get("VLLM_SM70_AK_STEP4_MS", "47.0")
                ),
            }
            self._ak_last_observe_t: float | None = None
            self._ak_active_k: int | None = None
            self._ak_alpha = float(os.environ.get("VLLM_SM70_AK_EMA_ALPHA", "0.3"))
            self._ak_p_hat: list[float] | None = None
            # Steps observed per K (drives the extrapolation gate).
            self._ak_observed = {0: 0, 1: 0, self.num_speculative_steps: 0}
            # Model-free per-position acceptance EMA (the SpecDecoding
            # metric itself, accumulated on-GPU from the target verdicts).
            # Replaces the IRLS logistic as the k-selection input after it
            # failed to track long-ctx code content (p stuck at the 0.95
            # clamp across refits, measured 2026-10-23).
            self._ak_acc_ema = torch.full(
                (self.num_speculative_steps,),
                0.5,
                dtype=torch.float32,
                device=self.device,
            )
            self._ak_pos = torch.arange(
                self.num_speculative_steps, device=self.device
            )
            self._ak_acc_ready = False
            self._ak_sel_n = 0

        # DP configuration
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size

        self.eplb_state: EplbState | None = None

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.temperature = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.arange_np = np.arange(self.max_num_reqs + 1, dtype=np.int32)
        self.draft_is_prefilling = torch.zeros(self.max_num_reqs, dtype=torch.bool)

        self.draft_token_confidence_probs = torch.empty_like(
            self.draft_tokens, dtype=torch.float32
        )
        self.enable_adaptive_verification = (
            self.speculative_config.enable_adaptive_verification
        )
        self.use_acceptance_estimator = (
            self.enable_adaptive_verification or self.sm70_adaptive_k
        )
        self.acceptance_estimator: OnlineAcceptanceEstimator | None = None

        self.draft_logits: torch.Tensor | None = None
        if self.speculative_config.draft_sample_method == "probabilistic":
            # Pre-temperature logits, cached from the previous decode step.
            dtype, fill = self.draft_logits_spec(vllm_config)
            self.draft_logits = torch.full(
                (
                    self.max_num_reqs,
                    self.num_speculative_steps,
                    self.vocab_size,
                ),
                fill,
                dtype=dtype,
                device=device,
            )

        self.draft_watermarker: DraftWatermarker | None = None
        watermark_config = getattr(vllm_config, "watermark_config", None)
        if watermark_config is not None:
            watermarker = create_watermarker(watermark_config)
            self.draft_watermarker = create_speculative_draft_watermarker(
                watermarker,
                self.max_num_reqs,
                device,
                watermark_config.allow_target_only_watermarking,
                self.num_speculative_steps,
                watermark_config.deduplicate_contexts,
                watermark_config.deduplicate_contexts_max_history,
            )

        self.supports_mm_inputs = False
        self.pcp_manager: PCPManager | None = None
        # Whether every draft attention builder can refresh its decode metadata
        # in place, so the refresh can be recorded in the draft CUDA graph.
        self.supports_decode_metadata_update = False

    def _select_adaptive_draft_depth(self, num_reqs: int, max_seq_len: int) -> int:
        """Self-calibrating uniform K (V3): pick k' in {0, 1, k_full}
        maximizing AL_hat(k') / step_ms[k'].

        step_ms[k'] is the EMA of this engine's own measured step wall time
        at each K (updated in observe_verification), so the decision uses
        the cost this configuration actually pays — not cross-config
        constants. AL_hat(0) = 1 (the bonus token). A switch requires the
        challenger to beat the incumbent by _ak_margin (10%) for _ak_hyst
        consecutive steps: this keeps k=0 (the no-MTP floor) in place while
        the estimator's AL predictions are only marginally better, and
        prevents flapping. The choice is fed back to the scheduler via the
        adaptive-k file (1-step delay is by design)."""
        k_full = self.num_speculative_steps
        est = self.acceptance_estimator
        if est is None or self.dp_size > 1:
            return k_full
        if getattr(self, "_ak_step_has_prefill", True):
            return self._ak_depth
        # Per-position acceptance: measured EMA (model-free), refreshed on a
        # 16-step cadence — one GPU->CPU sync per 16 steps. The per-step
        # verdict accumulation itself stays on the GPU.
        self._ak_sel_n += 1
        if self._ak_p_hat is None or self._ak_sel_n % 16 == 0:
            self._ak_p_hat = [
                min(max(x, 0.02), 0.98) for x in self._ak_acc_ema.tolist()
            ]
        p_hat = self._ak_p_hat
        # AL_hat(k) = 1 (bonus) + sum of acceptance-prefix products.
        al = 1.0
        prod = 1.0
        al_hat = {0: 1.0}
        for k in range(1, k_full + 1):
            prod *= p_hat[k - 1]
            al += prod
            al_hat[k] = al
        candidates = sorted(set([0, 1, k_full]) & set(al_hat) & set(self._ak_step_ms))
        scores = {k: al_hat[k] / self._sm70_cost_ms(k, max_seq_len) for k in candidates}
        best_k = max(candidates, key=lambda k: scores[k])
        self._ak_diag_n = getattr(self, "_ak_diag_n", 0) + 1
        if self._ak_diag_n % 50 == 1:
            logger.info(
                "[SM70-ADAPTIVE-K] k=%d target=%s scores=%s p_hat=%s ctx=%d",
                self._ak_depth,
                best_k,
                {k: round(s, 4) for k, s in scores.items()},
                [round(p, 3) for p in p_hat],
                max_seq_len,
            )
        if best_k != self._ak_depth and scores[best_k] >= self._ak_margin * scores[self._ak_depth]:
            if best_k == self._ak_target:
                self._ak_streak += 1
            else:
                self._ak_target = best_k
                self._ak_streak = 1
            if self._ak_streak >= self._ak_hyst:
                self._ak_streak = 0
                self._ak_depth = best_k
                logger.info(
                    "[SM70-ADAPTIVE-K] K -> %d (scores %s, ctx=%d)",
                    self._ak_depth,
                    {k: round(s, 4) for k, s in scores.items()},
                    max_seq_len,
                )
        else:
            self._ak_target = None
            self._ak_streak = 0
        return self._ak_depth

    @abstractmethod
    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        pass

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self.load_draft_model(target_model, target_attn_layer_names)
        self._validate_local_argmax_reduction()

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self.draft_attn_layer_names = set(all_attn_layers) - target_attn_layer_names
        for layer_name in self.draft_attn_layer_names:
            all_attn_layers[layer_name].is_draft_layer = True

        target_supports_mm = self.vllm_config.model_config.supports_multimodal_inputs
        draft_supports_mm = supports_multimodal_embeddings(self.model)
        self.supports_mm_inputs = target_supports_mm and draft_supports_mm
        if target_supports_mm and not draft_supports_mm:
            logger.warning_once(
                "Draft model %s does not support external multimodal embeddings. "
                "Embeddings from the target model will not be passed to the "
                "drafter; using text-only draft inputs instead.",
                type(self.model).__name__,
            )

        if self.use_acceptance_estimator:
            if self.use_local_argmax_reduction and not self.sm70_adaptive_k:
                raise ValueError(
                    "Adaptive verification without a confidence head estimates "
                    "per-position acceptance from the draft logits, which "
                    "use_local_argmax_reduction never materializes. Disable one "
                    "of them."
                )
            self.acceptance_estimator = OnlineAcceptanceEstimator(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.device,
            )
        if self.sm70_adaptive_k:
            # Boot canary: proves the valve took effect (estimator created,
            # local-argmax off so draft logits materialize for predict).
            logger.info(
                "[SM70-ADAPTIVE-K] acceptance estimator active: "
                "local_argmax_reduction=%s adaptive_verification=%s "
                "k_set={1..%d}; runtime k selection pending",
                self.use_local_argmax_reduction,
                self.enable_adaptive_verification,
                self.num_speculative_steps,
            )

    def set_eplb_state(self, eplb_state: EplbState) -> None:
        """Inject EPLB state after construction."""
        self.eplb_state = eplb_state

    def _prepare_eplb_forward(self, num_unpadded_tokens: int) -> None:
        """Call EPLB prepare_forward if EPLB is active for the draft model."""
        if self.eplb_state is not None:
            self.eplb_state.prepare_forward(
                self.speculative_config.draft_model_config,
                num_unpadded_tokens,
            )

    @property
    def attn_vllm_config(self) -> VllmConfig:
        """Config for the draft's attention metadata builders. Overridden by
        speculators whose attention mode differs from the target's."""
        return self.vllm_config

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.attn_groups, self.attn_cg_support, _ = init_attn_backend(
            kv_cache_config,
            self.attn_vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )
        self.block_tables = block_tables
        # The target model runner's buffers and attention groups. Draft
        # prefill reuses the target model's attention metadata, so its
        # cudagraph capture must build dummy metadata through the same
        # builders and buffers.
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups
        self._configure_decode_metadata_update()

    def _configure_decode_metadata_update(self) -> None:
        unsupported_backends = sorted(
            {
                attn_group.backend.get_name()
                for attn_groups in self.attn_groups
                for attn_group in attn_groups
                if not attn_group.supports_draft_decode_metadata_update
            }
        )
        try:
            _diag = [
                f"{g.backend.get_name()}={g.supports_draft_decode_metadata_update}"
                f"({type(g.get_metadata_builder()).__name__})"
                for gs in self.attn_groups
                for g in gs
            ]
            print(
                f"[QSA UPD] unsupported={sorted(unsupported_backends)} "
                f"groups={len([g for gs in self.attn_groups for g in gs])} "
                + " | ".join(_diag),
                flush=True,
            )
        except Exception as _exc:  # pragma: no cover - diagnostic only
            print(f"[QSA UPD] diag failed: {_exc}", flush=True)
        self.supports_decode_metadata_update = not unsupported_backends
        if unsupported_backends:
            logger.info_once(
                "Attention backend(s) %s cannot update draft decode metadata in "
                "place; falling back to rebuilding attention metadata eagerly.",
                ", ".join(unsupported_backends),
            )

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
        slot_mappings: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        # A FULL graph replays a captured shape whose padded requests each hold
        # a full query width, so attention must see the padded token count.
        # PIECEWISE/eager needs the actual token count, because batch_desc may
        # carry graph or DP padding that no request owns, which would desync it
        # from query_start_loc.
        num_tokens = (
            batch_desc.num_tokens
            if batch_desc.cg_mode == CUDAGraphMode.FULL
            else int(query_start_loc_np[-1])
        )
        query_start_loc_cpu = torch.empty(num_reqs_padded + 1, dtype=torch.int32)
        query_start_loc_cpu[: num_reqs + 1] = torch.from_numpy(
            query_start_loc_np[: num_reqs + 1]
        )
        query_start_loc_cpu[num_reqs:] = query_start_loc_cpu[num_reqs]
        max_query_len = int((query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).max())
        block_tables = [
            x[:num_reqs_padded] for x in self.block_tables.input_block_tables
        ]
        if slot_mappings is None:
            slot_mappings = self.block_tables.slot_mappings[:, :num_tokens]
        draft_seq_lens_cpu_upper_bound = torch.zeros(
            num_reqs_padded, dtype=torch.int32, device="cpu"
        )
        torch.add(
            seq_lens_cpu_upper_bound[:num_reqs],
            step,
            out=draft_seq_lens_cpu_upper_bound[:num_reqs],
        )
        draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)
        if dcp_local_seq_lens is None and self.dcp_size > 1:
            # Draft steps advance and rewind their own global sequence lengths,
            # so the target model's DCP-local lengths may already be stale.
            dcp_local_seq_lens = prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
                self.block_tables.cp_rank,
                self.block_tables.cp_interleave,
            )
        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens,
            query_start_loc_gpu=self.input_buffers.query_start_loc[
                : num_reqs_padded + 1
            ],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            dcp_local_seq_lens=(
                None
                if dcp_local_seq_lens is None
                else dcp_local_seq_lens[:num_reqs_padded]
            ),
            max_seq_len=self.draft_max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            causal=causal,
            seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,
            positions=self.input_buffers.positions[:num_tokens],
            is_prefilling=self.draft_is_prefilling[:num_reqs_padded],
        )
        return attn_metadata

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        """Dtype and fill for the cached proposal distribution.

        Speculators that write only a subset of columns each step override this.
        """
        return vllm_config.model_config.head_dtype, 0.0

    def _validate_local_argmax_reduction(self) -> None:
        if not self.use_local_argmax_reduction:
            return
        if self.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "use_local_argmax_reduction is not compatible with "
                "draft_sample_method='probabilistic'."
            )
        if not hasattr(self.model, "get_top_tokens"):
            raise ValueError(
                "use_local_argmax_reduction is enabled but draft model "
                f"{self.model.__class__.__name__} does not implement "
                "get_top_tokens()."
            )
        logger.info(
            "Using local argmax reduction for draft token generation "
            "(communication: O(2*tp_size) vs O(vocab_size))."
        )

    def compute_draft_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int
    ) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def get_draft_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states)

    def get_draft_top_tokens_with_feature(
        self, hidden_states: torch.Tensor, spec_step_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        getter = getattr(model, "get_top_tokens_with_feature", None)
        if getter is None:
            raise ValueError(
                "VLLM_SM70_MTP_ADAPTIVE_K requires a draft model with "
                f"get_top_tokens_with_feature(); {type(model).__name__} lacks it."
            )
        return getter(hidden_states)

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
        spec_step_idx: int = 0,
        num_steps: int = 1,
    ) -> torch.Tensor:
        if draft_logits is None and self.use_local_argmax_reduction:
            if self.sm70_adaptive_k and self.acceptance_estimator is not None:
                # O(tp) feature path: sample via the local-argmax reduction and
                # feed the acceptance estimator the global max draft logit
                # (already gathered for the reduction) instead of full logits.
                sampled, feature_vals = self.get_draft_top_tokens_with_feature(
                    hidden_states, spec_step_idx
                )
                self.acceptance_estimator.predict_feature(
                    feature_vals,
                    idx_mapping,
                    draft_step,
                    self.draft_token_confidence_probs,
                )
                return sampled
            return self.get_draft_top_tokens(hidden_states, spec_step_idx)

        logits = self.compute_draft_logits(hidden_states, spec_step_idx)
        if draft_logits is not None:
            sampler = (
                gumbel_sample
                if self.draft_watermarker is None
                else partial(self.draft_watermarker.sample, num_steps=num_steps)
            )
            sampled = sampler(
                logits,
                idx_mapping,
                temperature,
                seeds,
                sample_src_positions,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=draft_logits,
                logits_cache_col=draft_step,
                use_fp64=self.use_fp64_gumbel,
            )
        else:
            sampled = logits.argmax(dim=-1)
        self._maybe_predict_acceptance(logits, idx_mapping, draft_step)
        return sampled

    def _maybe_predict_acceptance(
        self, logits: torch.Tensor, idx_mapping: torch.Tensor, draft_step: torch.Tensor
    ) -> None:
        if self.acceptance_estimator is not None:
            self.acceptance_estimator.predict(
                logits,
                idx_mapping,
                draft_step,
                self.draft_token_confidence_probs,
                self.temperature,
            )

    def observe_verification(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        sampled_token_ids: torch.Tensor | None = None,
    ) -> None:
        """Fold the target's verdict on the last drafts into the estimator.

        Must run before the next `propose`, which overwrites the per-slot
        features the verdict grades.
        """
        if self.acceptance_estimator is not None:
            if self.sm70_adaptive_k:
                self._sm70_update_acc_ema(
                    idx_mapping, num_sampled, sampled_token_ids
                )
            if (
                self.sm70_adaptive_k
                and self._ak_active_k == 0
                and sampled_token_ids is not None
            ):
                # k=0 self-grade: verdict = (this step's draft token == the
                # token the target actually sampled) — exactly the position-0
                # verdict a k=1 step would produce. Without this, k=0 carries
                # no learning signal and the estimator's predictions freeze,
                # locking the selector at the floor. LHS gathers drafts by
                # state slot into batch order; RHS rows are already in batch
                # order (do NOT index it with idx_mapping — its values are
                # state slots, out of range for a [num_reqs, 1] tensor).
                idx = idx_mapping[: num_sampled.shape[0]]
                acc = (
                    self._sm70_k0_draft[idx]
                    == sampled_token_ids[: num_sampled.shape[0], 0]
                ).to(num_sampled.dtype)
                self.acceptance_estimator.step(
                    idx_mapping, 1 + acc, 1 - acc
                )
            elif not (self.sm70_adaptive_k and self._ak_active_k == 0):
                # k=0 with no sampled ids available: leave the estimator frozen.
                self.acceptance_estimator.step(idx_mapping, num_sampled, num_rejected)
            if self.sm70_adaptive_k:
                self._sm70_update_step_ema()
                self._sm70_write_adaptive_k()
            if getattr(self, "_sm70_adaptive_k_first_observe", True):
                self._sm70_adaptive_k_first_observe = False
                # Runtime canary: the online estimator loop is live. The
                # mean of this step's per-position predictions is the raw
                # k-selection input; logged once (eager path, graph-safe).
                preds = self.acceptance_estimator.predictions
                logger.info(
                    "[SM70-ADAPTIVE-K] first observe: per-position "
                    "acceptance pred mean/pos=%s",
                    [
                        round(float(preds[: idx_mapping.shape[0], pos].mean()), 4)
                        for pos in range(preds.shape[1])
                    ],
                )

    def _sm70_update_acc_ema(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        sampled_token_ids: torch.Tensor | None,
    ) -> None:
        """Model-free per-position acceptance EMA from the target verdicts.

        labels[i] = 1 iff position i was accepted (i < num_sampled - 1) — the
        exact quantity the SpecDecoding metric reports, accumulated here on
        the GPU with no host sync. At k=0 there are no verdicts; fall back to
        the self-grade (draft token vs the target's sampled token) at
        position 0 so the EMA keeps learning instead of freezing the floor.
        """
        try:
            k = self.num_speculative_steps
            n = num_sampled.shape[0]
            # Only pure decode steps carry meaningful verdicts. Prefill chunks
            # (and mixed batches) sample their own prompt tokens; letting
            # their num_sampled=1 labels in drags the EMA to zero (measured:
            # a 24-chunk prefill collapsed it from 0.5 to ~0.05).
            if getattr(self, "_ak_step_has_prefill", True):
                return
            if self._ak_active_k == 0:
                draft = getattr(self, "_sm70_k0_draft", None)
                if draft is None or sampled_token_ids is None:
                    return
                acc0 = (
                    draft[idx_mapping[:n]] == sampled_token_ids[:n, 0]
                ).float()
                self._ak_acc_ema[0].mul_(0.9).add_(0.1 * acc0.mean())
            else:
                labels = (
                    self._ak_pos.unsqueeze(0) < (num_sampled - 1).unsqueeze(1)
                ).float()
                self._ak_acc_ema.mul_(0.9).add_(0.1 * labels.mean(dim=0))
            self._ak_acc_ready = True
        except Exception:  # noqa: BLE001 — diagnostics only, never break a step
            pass

    def _sm70_update_step_ema(self) -> None:
        """EMA of the step wall time at the K this step ran with.

        Consecutive observes are one engine step apart, so the interval is
        the step cost (CPU + GPU + sync, exactly what the next step competes
        for). Sanity-bounded so prefill chunks and stalls are rejected.
        """
        now = time.perf_counter()
        last = self._ak_last_observe_t
        k_now = self._ak_active_k
        self._ak_last_observe_t = now
        if last is None or k_now is None:
            return
        step_ms = (now - last) * 1000.0
        if not 0.5 < step_ms < 1000.0:
            return
        base = self._ak_step_ms.get(k_now)
        if base is not None:
            # Jump guard: reject outliers (cold starts, stalls) that are
            # more than 3x the running estimate; the true step cost moves
            # slowly.
            if step_ms < 3.0 * base:
                self._ak_step_ms[k_now] = (
                    1.0 - self._ak_alpha
                ) * base + self._ak_alpha * step_ms
                self._ak_observed[k_now] = self._ak_observed.get(k_now, 0) + 1

    def _sm70_cost_ms(self, k: int, max_seq_len: int) -> float:
        """Step cost at K: measured EMA when that K has been observed
        enough; otherwise linear extrapolation from the anchors. cost(k) =
        EMA[1] + (k-1) * G with G from the measured k=1..k_full span when it
        is trustworthy, else the ctx-scaled prior (1 ms @10K .. 11 ms @250K).
        k=0 uses its own EMA (the no-spec step)."""
        if k == 0:
            return self._ak_step_ms[0]
        k_full = self.num_speculative_steps
        # k_full == 1 makes the two observation keys identical, so the check
        # below would pass after 8 steps and divide by (1 - 1). Guard on the
        # depth itself: with a single draft step there is no slope to fit.
        if (
            k_full >= 2
            and self._ak_observed.get(1, 0) >= 8
            and self._ak_observed.get(k_full, 0) >= 8
        ):
            g = (self._ak_step_ms[k_full] - self._ak_step_ms[1]) / (k_full - 1)
            if 0.5 < g < 30.0:
                return self._ak_step_ms[1] + (k - 1) * g
        t = (max_seq_len - 10000) / (250000 - 10000)
        g_prior = 1.0 + 10.0 * min(max(t, 0.0), 1.0)
        return self._ak_step_ms[1] + (k - 1) * g_prior

    def _sm70_write_adaptive_k(self) -> None:
        """Feed the recommended K back to the scheduler (testbed valve).

        Atomic tmp+replace file write, ~us per decode step; the scheduler
        clamps its scheduled K to this value (1-step delay by design).
        """
        try:
            path = "/tmp/sm70_adaptive_k.json"
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "k": int(self._ak_depth),
                        "steps": {str(k): round(v, 2) for k, v in self._ak_step_ms.items()},
                    },
                    f,
                )
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 — feedback loss = scheduler keeps max K
            pass

    def prepare_watermarking(
        self,
        sampler: "GPUWatermarkSampler",
        idx_mapping: torch.Tensor,
    ) -> None:
        if self.draft_watermarker is None:
            return
        self.draft_watermarker.prepare(
            sampler._get_contexts(idx_mapping),
            sampler.watermarking.gpu[idx_mapping],
            sampler.req_states.all_token_ids.gpu,
            sampler.req_states.prompt_len.gpu,
            sampler.req_states.total_len.gpu,
        )

    def _copy_request_inputs(
        self,
        num_reqs: int,
        # [num_reqs]
        idx_mapping: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dummy_run: bool = False,
    ) -> None:
        # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.
        # NOTE(woosuk): For draft sampling, we only consider the temperature
        # and ignore the other sampling parameters such as top_k and top_p,
        # for simplicity and performance.
        # While this may slightly degrade the acceptance rate, it does not
        # affect the output distribution after rejection sampling.
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        # idx_mapping == -1 marks a row the drafter must not act on: sampling
        # skips it and compute_slot_mappings emits PAD. CUDA-graph padded rows
        # always get it; a dummy batch gets it for every row, since its arange
        # idx_mapping names request-state slots with stale block tables.
        if dummy_run:
            self.idx_mapping.fill_(-1)
        else:
            self.idx_mapping[:num_reqs].copy_(idx_mapping)
            self.idx_mapping[num_reqs:].fill_(-1)

    def _build_uniform_batch_dp_sync(
        self,
        target_dp_sync_state: DPSyncState,
        num_reqs: int,
        num_query_per_req: int = 1,
    ) -> tuple[DPSyncState, int]:
        num_batch_tokens = target_dp_sync_state.num_reqs * num_query_per_req
        assert num_reqs * num_query_per_req <= num_batch_tokens, (
            "reusing a DP sync that does not cover this batch's requests"
        )
        return replace(
            target_dp_sync_state,
            num_tokens_across_dp=torch.full_like(
                target_dp_sync_state.num_tokens_across_dp, num_batch_tokens
            ),
            uniform_token_count=num_query_per_req,
            eager=False,
        ), num_batch_tokens

    def _build_uniform_attn_metadata(
        self,
        batch_desc: BatchExecutionDescriptor,
        num_reqs: int,
        num_query_per_req: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        query_start_loc_np = self.arange_np[: num_reqs + 1] * num_query_per_req
        return self._build_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            query_start_loc_np=query_start_loc_np,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            causal=causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )

    def _update_draft_decode_metadata(
        self, attn_metadata: dict[str, Any], num_reqs: int
    ) -> None:
        if self.block_tables.cp_size > 1:
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.block_tables.cp_size,
                self.block_tables.cp_rank,
                self.block_tables.cp_interleave,
            )
        for groups in self.attn_groups:
            for group in groups:
                group.update_draft_decode_metadata(attn_metadata)
