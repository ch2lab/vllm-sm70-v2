# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import os

from collections.abc import Callable, Iterable
from typing import ClassVar, cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm import envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.config.cache import CacheConfig, CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from .opattr_hc import hc_flush as _qflush, hc_mark as _qmark
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.qwen3_vl import mrope_positions_factor
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import (
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
    nvfp4_kv_cache_full_dim,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
    replace,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer

_DIAG_SKIP_QSA_BODY = os.environ.get("VLLM_SM70_DIAG_SKIP_QSA_BODY", "0") == "1"
_DIAG_SKIP_QSA_ATTN = (
    os.environ.get("VLLM_SM70_DIAG_SKIP_QSA_ATTN", "0") == "1"
)

# One-shot state for the VLLM_SM70_QSA_DUMP reader-side diagnostic probe.
from vllm.logger import init_logger

logger = init_logger(__name__)

_QSA_READER_DUMP: dict[str, bool] = {}
_qsa_slice_probe_count = 0


class Qwen4ExpQSAQKVIndexerLinear(MergedColumnParallelLinear):
    """Pack sharded Q/gate, K, V and replicated indexer Q/K in one GEMM."""

    def __init__(
        self,
        qkv_proj: QKVParallelLinear,
        index_size: int,
        quant_config: QuantizationConfig | None,
    ) -> None:
        self.num_kv_head_replicas = qkv_proj.num_kv_head_replicas
        super().__init__(
            input_size=qkv_proj.input_size,
            output_sizes=[*qkv_proj.output_sizes, index_size * qkv_proj.tp_size],
            bias=False,
            params_dtype=qkv_proj.params_dtype,
            quant_config=quant_config,
            prefix=qkv_proj.prefix,
        )

    def _load_shard(
        self,
        loader: Callable[..., None],
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == 3:
            shard_rank = 0
        elif loaded_shard_id in (1, 2):
            shard_rank = tp_rank // self.num_kv_head_replicas
        else:
            shard_rank = tp_rank
        self.tp_rank = shard_rank
        if param_tp_rank is not None:
            param.tp_rank = shard_rank
        try:
            loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(super().weight_loader, param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(
            super().weight_loader_v2, param, loaded_weight, loaded_shard_id
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        def remap_shards():
            for name, weight in weights:
                shard_id = getattr(weight, "shard_id", None)
                if isinstance(shard_id, str):
                    weight = weight.detach()
                    weight.shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
                yield name, weight

        return super().load_weights(remap_shards())


def qsa_kv_cache_dtype(cache_config: CacheConfig, prefix: str) -> CacheDType | str:
    """The layer's KV cache dtype.

    Priority: ``--kv-cache-dtype-override-layers`` (per-layer '<idx>=<dtype>')
    > ``--kv-cache-dtype-skip-layers`` ('auto') > ``--kv-cache-dtype``.
    """
    idx = str(extract_layer_index(prefix))
    for entry in cache_config.kv_cache_dtype_override_layers:
        override_idx, sep, dtype = entry.partition("=")
        if sep and override_idx == idx:
            return dtype
    if idx in cache_config.kv_cache_dtype_skip_layers:
        return "auto"
    return cache_config.cache_dtype


_QSA_STORE_DUMP: dict[str, bool] = {}


def _qsa_store_dump(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
) -> None:
    """One-shot diagnostic: the nvfp4 pool store's inputs + view geometry."""
    dump = os.getenv("VLLM_SM70_QSA_STORE_DUMP")
    if dump and key.shape[0] in (57, 65) and not _QSA_STORE_DUMP.get(str(key.shape[0])):
        _QSA_STORE_DUMP[str(key.shape[0])] = True
        torch.save(
            {
                "key": key.contiguous().cpu(),
                "value": value.contiguous().cpu(),
                "slot_mapping": slot_mapping.detach().cpu(),
                "kv_cache_strides": kv_cache.stride(),
                "kv_cache_shape": tuple(kv_cache.shape),
                "k_cache_strides": k_cache.stride(),
                "k_cache_shape": tuple(k_cache.shape),
                "key_data_ptr": key.data_ptr(),
                "kv_cache_data_ptr": kv_cache.data_ptr(),
                "k_cache_data_ptr": k_cache.data_ptr(),
                "stream_capturing": torch.cuda.is_current_stream_capturing(),
            },
            dump.rsplit(".pt", 1)[0] + f"_{key.shape[0]}.pt",
        )


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        # FlashAttentionMetadata carries no real request count: num_reqs is
        # only a local inside the base build() and num_prefill_reqs stays 0
        # without DCP, while block_table/seq_lens are padded to max-num-seqs.
        # The QSA page4 dispatch needs the scheduler's count to slice those to
        # the live rows, so attach it here. Prefill runs eager (graphs are
        # UNIFORM_BATCH decode only), so this is the current step's count.
        # The scheduler's num_reqs is normally already unpadded, but some
        # batched-prefill paths hand over a metadata object padded to
        # max-num-seqs (observed at TP4: 32 rows for a one-request prefill).
        # Counting positive query-length increments on the CPU copy recovers
        # the live request count in that case without a device sync, since
        # padding repeats the last offset (zero increments).
        num_reqs = int(common_attn_metadata.num_reqs)
        qsl_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
        if qsl_cpu is not None and qsl_cpu.numel() > 1:
            live = int((qsl_cpu[1:] > qsl_cpu[:-1]).sum())
            if live:
                num_reqs = live
        metadata.num_reqs = num_reqs
        return metadata


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.bfloat16,
        torch.float16,
    ]
    # fp8/fp8_e4m3: e4m3 bytes in a uint8 cache, written by reshape_and_cache
    # with the layer's per-tensor scales and dequantized on load inside the QSA
    # Triton kernel. flash-attn never runs over this cache, so its fp8 probe
    # does not apply (see supports_kv_cache_dtype and the impl constructor).
    # fp16: the same Triton kernels run over FP16 caches on SM70.
    # nvfp4: packed fp4 pool (store kernel + in-kernel-dequant page4 readers);
    # QSA dequantizes via the page4-TP8 route, never the fp8/e4m3 Triton arm.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    ]

    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        """NVFP4 stores K and V as separate per-head slots of packed fp4 data
        plus e4m3 block scales (same convention as the FlashInfer backend:
        page = [K data | K scale | V data | V scale])."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
        full_dim = nvfp4_kv_cache_full_dim(spec.head_size)
        assert full_dim == nvfp4_kv_cache_full_dim(spec.head_size_v)
        return replace(
            spec,
            num_head_slots=2 * spec.num_kv_heads,
            state_content_bytes=full_dim * get_dtype_size(spec.dtype),
        )

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # QSA dequantizes the fp8 KV in its own Triton kernel and never runs
        # flash-attn over the quantized cache, so the parent's fp8-KV rejection
        # does not apply and every combination it is handed is accepted here.
        return None

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        # The parent constructor probes flash-attn for quantized-KV support and
        # raises where it is unavailable (sm120), but QSA dequantizes fp8 inside
        # its own Triton kernel and never runs flash-attn over the cache. Hand
        # the parent "auto" for that probe and restore the real dtype afterwards:
        # the parent only uses it there, and do_kv_cache_update reads the
        # attribute at call time.
        real_kv_cache_dtype = kv_cache_dtype
        if kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2", "nvfp4"):
            kv_cache_dtype = "auto"
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        self.kv_cache_dtype = real_kv_cache_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in (
            "auto",
            "bfloat16",
            "fp16",
            "fp8",
            "fp8_e4m3",
            "fp8_e5m2",
            "nvfp4",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP16 or FP8-e4m3/e5m2 main KV "
                "cache"
            )
        self.supports_quant_query_input = False

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """KV write. NVFP4 pool pages pack [K data | K scale | V data |
        V scale] with K and V as separate head slots (not interleaved in
        the last dim), so the parent's head_size split does not apply; the
        store kernel is the stable-ABI reshape_and_cache_flash (same op the
        FlashInfer backend uses for nvfp4)."""
        if self.kv_cache_dtype == "nvfp4":
            k_cache, v_cache = kv_cache.transpose(1, 2).split(
                self.num_kv_heads, dim=-2
            )
            _qsa_store_dump(key, value, kv_cache, slot_mapping, k_cache)
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                "nvfp4",
                layer._k_scale,
                layer._v_scale,
            )
            return
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_gate: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        # D4a shadow (spec d4a-spec.md §2.2 S3): when
        # VLLM_SM70_QSA_DECODE_FUSE == 1 (matching the fused dispatch in
        # indexer_qsa.py — every other value, incl. reserved 2, is today)
        # and the batch is non-empty, the sparse attention chain
        # (splitk + merge) fully overwrites `output` (empty rows written 0
        # by both kernels — spec §1.2 proof), so the fill is skipped.
        # env != 1 (default 0) keeps the fill on every path — byte-identical
        # off path (rule 11670); an empty batch also keeps it (the chain
        # below early-returns without writing `output`).
        if not (envs.VLLM_SM70_QSA_DECODE_FUSE == 1 and num_tokens > 0):
            output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        k_scale = v_scale = None
        if self.kv_cache_dtype == "nvfp4":
            # Pool cells are per-token [K data | K scale | V data | V scale]
            # (288 B/token) with K and V as separate head slots; ops.qsa
            # validates these side views against the in-kernel-dequant
            # readers' cell geometry.
            key_cache = kv_cache[:, 0].unsqueeze(2)
            value_cache = kv_cache[:, 1].unsqueeze(2)
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            # The cache is allocated as uint8; reinterpret the packed fp8
            # bytes (same itemsize, so shape and strides are preserved).
            fp8_dtype = (
                torch.float8_e5m2
                if self.kv_cache_dtype == "fp8_e5m2"
                else torch.float8_e4m3fn
            )
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)
            # Host-side per-tensor dequant scales (Python floats), as used by
            # other host-scale backends; folded into the kernel's scales.
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float
        elif self.kv_cache_dtype == "nvfp4":
            # Host-side per-tensor scales of the nvfp4 store kernel (1.0 by
            # default); the reader decodes the per-token e4m3 scale codes in
            # kernel and folds these in as floats.
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float
        if query.dtype not in (torch.bfloat16, torch.float16) or (
            key_cache.dtype
            not in (
                torch.bfloat16,
                torch.float16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
                torch.uint8,
            )
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16/FP16 Q and BF16, FP16 or FP8 K/V"
            )

        query_positions = None
        sequence_lengths = None
        from .ops.qsa import _qsa_page4_tp8_enabled
        if (
            os.getenv("VLLM_SM70_QSA_FLASHV100", "0") == "1"
            or _qsa_page4_tp8_enabled()
        ):
            # Absolute per-row positions for the Flash-V100 page4 planner:
            # row r of request i sits at seq_len_i - qlen_i + (r - start_i).
            qsl = attn_metadata.query_start_loc
            seq_lens = attn_metadata.seq_lens
            if envs.VLLM_SM70_QSA_POS_FUSE == 1:
                # Candidate H: one launch instead of the 8-9 index kernels
                # below, which are recomputed for every QSA layer (12/step).
                # Same integer formula => bit-identical (gated, default 0).
                from .ops.qsa_pos_fuse import qsa_positions_fused

                query_positions = qsa_positions_fused(
                    token_to_req, qsl, seq_lens, num_tokens
                )
            else:
                local = (
                    torch.arange(num_tokens, device=qsl.device) - qsl[token_to_req]
                )
                query_positions = (
                    seq_lens[token_to_req]
                    - (qsl[1:] - qsl[:-1])[token_to_req]
                    + local
                ).to(torch.int64)
            sequence_lengths = seq_lens.to(torch.int32)

        from .ops.qsa import qsa_sparse_paged_attention

        dump = os.getenv("VLLM_SM70_QSA_DUMP")
        if dump and num_tokens == 57 and not _QSA_READER_DUMP.get(str(num_tokens)):
            # One-shot diagnostic: reader-side view/metadata plus the cache
            # as stored, to cross-check the prepare write offline. The flag
            # is module-level so all 12 layers dump the same first call,
            # matching the prepare-side (function-level) one-shot.
            _QSA_READER_DUMP[str(num_tokens)] = True
            torch.save(
                {
                    "layer_name": self.layer_name,
                    "kv_dtype": self.kv_cache_dtype,
                    "kv_cache_dtype": str(kv_cache.dtype),
                    "kv_cache_shape": tuple(kv_cache.shape),
                    "kv_cache_strides": kv_cache.stride(),
                    "key_cache_shape": tuple(key_cache.shape),
                    "key_cache_strides": key_cache.stride(),
                    "key_first_blocks": key_cache[:32].detach()
                        .contiguous().cpu(),
                    "value_first_blocks": value_cache[:32].detach()
                        .contiguous().cpu(),
                    "logical_indices": logical_indices.cpu(),
                    "block_table": attn_metadata.block_table.cpu(),
                    "token_to_req": token_to_req[:num_tokens].cpu(),
                    "seq_lens": attn_metadata.seq_lens.cpu(),
                    "query_start_loc": attn_metadata.query_start_loc.cpu(),
                    "query_positions": (
                        query_positions.cpu()
                        if query_positions is not None
                        else None
                    ),
                    "k_scale": k_scale,
                    "v_scale": v_scale,
                    "num_tokens": num_tokens,
                    "use_prefill_config": use_prefill_config,
                    "q_dtype": str(query.dtype),
                },
                dump.rsplit(".pt", 1)[0] + f"_{num_tokens}.pt",
            )

        # 2026-10-24: block_table and seq_lens are both padded to max-num-seqs
        # (32 rows at TP4) even for a single-request prefill, so the page4
        # gate's `block_table.shape[0] != 1` check rejected every call and
        # pushed prefill onto the decode kernel (313632 CTAs, 71 ms/call).
        # num_reqs is the scheduler's real request count: slicing by it leaves
        # a single-request prefill with one row (page4 eligible) while any
        # multi-request or decode+prefill batch keeps N > 1 rows and stays on
        # the Triton route exactly as the gate's comment intends. seq_lens must
        # be sliced with it because the page4 planner requires
        # sequence_lengths.size(0) == block_table.size(0).
        _num_reqs = int(getattr(attn_metadata, "num_reqs", 0)) or int(
            attn_metadata.block_table.shape[0]
        )
        _num_reqs = min(_num_reqs, int(attn_metadata.block_table.shape[0]))
        global _qsa_slice_probe_count
        # Log the first few prefill calls instead of one: the warmup batch is
        # also a prefill, so a one-shot probe reports warmup and hides the
        # shape the dispatch decision actually sees during the measurement.
        if (
            _qsa_slice_probe_count < 8
            and use_prefill_config
            and num_tokens >= 512
        ):
            _qsa_slice_probe_count += 1
            logger.info(
                "QSA slice probe: md=%s num_reqs=%d block=%s seq_lens=%s "
                "actual_tokens=%d rows=%d rows%%8=%d",
                type(attn_metadata).__name__, _num_reqs,
                tuple(attn_metadata.block_table.shape),
                tuple(sequence_lengths.shape)
                if sequence_lengths is not None
                else None,
                attn_metadata.num_actual_tokens, num_tokens, num_tokens % 8,
            )
        _block_table = attn_metadata.block_table[:_num_reqs]
        _sequence_lengths = (
            sequence_lengths[:_num_reqs] if sequence_lengths is not None else None
        )
        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            _block_table,
            token_to_req,
            use_prefill_config,
            output[:num_tokens],
            k_scale=k_scale,
            v_scale=v_scale,
            output_gate=output_gate[:num_tokens],
            query_positions=query_positions,
            sequence_lengths=_sequence_lengths,
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        # Per-layer dtype: QSA layers can sit in --kv-cache-dtype-skip-layers
        # (e.g. global nvfp4 KV) and keep their own storage dtype.
        if qsa_kv_cache_dtype(cache_config, prefix) not in (
            "auto",
            "bfloat16",
            "fp16",
            "fp8",
            "fp8_e4m3",
            "fp8_e5m2",
            "nvfp4",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP16 or FP8-e4m3/e5m2 main "
                "KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # Serve the full configured context even when the checkpoint's
        # max_position_embeddings is shorter: with rope_type=default the cache
        # is the same frequency formula evaluated at more positions, so the
        # extension is exact rather than an approximation. The drafter keeps its
        # own config limit while consuming the target's positions, so it has to
        # cover the target's context too.
        spec = vllm_config.speculative_config
        target_model_config = spec.target_model_config if spec is not None else None
        self._rope_cache_positions = max(
            int(config.max_position_embeddings),
            int(model_config.max_model_len),
            int(target_model_config.max_model_len)
            if target_model_config is not None
            else 0,
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=self._rope_cache_positions,
            rope_parameters=config.rope_parameters,
            mrope_positions_factor=mrope_positions_factor(vllm_config),
        )
        # Nothing bounds the cos/sin row index at use time: the fused QSA
        # prepare kernel and the rotary kernels index the cache with raw token
        # positions. A cache shorter than the served context reads past its end,
        # which faults only when that memory happens to be unmapped (and
        # otherwise silently returns garbage).
        if self.rotary_emb.cos_sin_cache.shape[0] < self._rope_cache_positions:
            raise ValueError(
                "RoPE cos/sin cache covers "
                f"{self.rotary_emb.cos_sin_cache.shape[0]} positions but the "
                f"served context needs {self._rope_cache_positions}"
            )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = qsa_kv_cache_dtype(cache_config, prefix)
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.uint8,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16, FP16 or FP8-e4m3 (uint8) cache "
                "storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.impl.layer_name = self.layer_name
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        # One launch does the indexer prepare, the main QK-norm/RoPE/gate and
        # the main K/V cache write (see QSAIndexer.forward); otherwise all of
        # them take the separate kernels. For nvfp4 the fused launch instead
        # emits bf16 K/V (the pool store cannot run inside the Triton
        # kernel); _run_qsa writes them through do_kv_cache_update.
        self.use_fused_qsa_prepare = (
            self.use_fused_qk_norm_rope_gate
            and self.indexer.use_fused_pre_indexer
        )
        self.fuse_indexer_projection = vllm_config.lora_config is None
        if self.fuse_indexer_projection:
            self.index_qk_size = self.indexer.index_qk_proj.output_size
            self.qkv_proj = Qwen4ExpQSAQKVIndexerLinear(
                self.qkv_proj,
                self.index_qk_size,
                model.without_modelopt_fp4(quant_config),
            )
            del self.indexer.index_qk_proj

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # PACKED selection buffer: the trailing column holds each row's
        # valid-entry count (written by the expand kernel) — never a token
        # index; the sparse attention kernel reads it as its loop bound.
        # MTP skip_topk steps reuse rows frozen from step 0; the count is
        # a row column, so compaction/reuse keep it paired with the content.
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.packed_output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = None
        if self.fuse_indexer_projection:
            mapper = WeightsMapper(
                orig_to_new_stacked={
                    "indexer.index_qk_proj.": ("qkv_proj.", 3),
                }
            )
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        output_gate: torch.Tensor | None,
        qkv: torch.Tensor,
    ) -> None:
        # query/key/value/output_gate are None when the fused prepare runs
        # inside the indexer launch.
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        _qmark("qsa:pre")
        # The selection buffer is sized from the scheduler config at build
        # time, but chunked prefill can hand this layer a larger step
        # (page-aligned chunks, e.g. 15680 with max_num_batched_tokens=16K).
        # Grow lazily here; _run_qsa always runs eagerly (breakable-graph
        # boundary), so reallocating is capture-safe.
        if num_tokens > self.topk_indices_buffer.shape[0]:
            self.topk_indices_buffer = torch.empty(
                num_tokens,
                self.indexer.packed_output_width,
                dtype=torch.int32,
                device=self.topk_indices_buffer.device,
            )
        selected, main_outputs = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
            attn=self,
            qkv=qkv,
            slot_mapping=main_metadata.slot_mapping,
        )
        _qmark("qsa:indexer")
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        _qmark("qsa:kvu_start")
        if main_outputs is None:
            assert key is not None and value is not None
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        else:
            query, output_gate, main_key, main_value = main_outputs
            if main_key is not None:
                # nvfp4: the fused prepare emitted bf16 K/V instead of
                # writing the pool; store it through the nvfp4 pool update.
                impl.do_kv_cache_update(
                    self,
                    main_key,
                    main_value,
                    self.kv_cache,
                    main_metadata.slot_mapping,
                )
        assert query is not None and output_gate is not None
        _qmark("qsa:attn_start")
        if _DIAG_SKIP_QSA_ATTN:
            # Diagnostic only (numerically wrong): keep the indexer selection and
            # the KV write, skip the page4 attention body.
            return
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
            output_gate=output_gate,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _qmark("qsa:begin")
        qkv, _ = self.qkv_proj(hidden_states)
        if self.fuse_indexer_projection:
            qkv, projected_qk = qkv.split(
                [2 * self.q_size + 2 * self.kv_size, self.index_qk_size], dim=-1
            )
        else:
            projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        _qmark("qsa:qkv")
        num_tokens = hidden_states.shape[0]
        if not self.use_fused_qsa_prepare:
            q, k, v, gate = self._project_qkv_gate(qkv, positions)
            assert gate is not None
            query = q.view(num_tokens, self.num_heads, self.head_dim)
            key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            # Norm/RoPE/gate and the K/V cache write happen inside _run_qsa.
            query = key = value = gate = None
        if _DIAG_SKIP_QSA_BODY:
            # Diagnostic only (numerically wrong): keep qkv/indexer/o_proj and
            # skip the attention body (norm/rope/KV write + page4 + select).
            attn_output = qkv.new_zeros(num_tokens, self.num_heads, self.head_dim)
        else:
            attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
            self._run_qsa(
                projected_qk,
                positions,
                query,
                key,
                value,
                attn_output,
                gate,
                qkv,
            )
        _qmark("qsa:body")
        flat_output = attn_output.view(num_tokens, -1)
        output, _ = self.o_proj(flat_output)
        _qmark("qsa:o")
        _qflush("qsa:layer")
        return output


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
]
