# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Qwen4Exp n-gram embedding storage with device and pinned-host backends.

Both the NVIDIA and AMD Qwen4Exp implementations use these classes so the (large)
n-gram embedding table can be kept in pinned host memory and looked up through
Unified Virtual Addressing on any CUDA-alike platform.
"""

import json
import os
import re
import threading
from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm import envs
from vllm.distributed import get_dp_group, get_etp_group, get_tp_group
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .ple import PLEVocabParallelEmbedding
from ..nvidia.nvfp4_ple import (
    nvfp4_e2m1_lut,
    nvfp4_e4m3_lut,
    nvfp4_from_fp8,
)

logger = init_logger(__name__)


def _ple_sync_requested() -> bool:
    """VLLM_SM70_PLE_SYNC=1: run the PLE UVA lookup on the current stream.

    Diagnostic switch for the TP4xPP2 illegal-memory-access hunt (see
    PinnedPleEmbedding.start_prefetch). Default off, so the overlapped side-stream
    path stays the production behaviour.
    """
    return os.environ.get("VLLM_SM70_PLE_SYNC", "0") == "1"


class Qwen4ExpPLEEmbedding(PLEVocabParallelEmbedding, ABC):
    """ETP-sharded PLE table shared by device and pinned-host backends."""

    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: "Qwen4ExpPLEEmbeddingMethod",
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
    ) -> None:
        del num_ngram_heads, max_total_tokens
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
            parallel_group=get_etp_group(),
        )
        self.embedding_method = embedding_method
        self.data_parallel_rank = data_parallel_rank
        tp_size = get_tp_group().world_size
        if self.tp_size % tp_size:
            raise ValueError(
                "ETP size must be divisible by TP size, but got "
                f"ETP={self.tp_size} and TP={tp_size}"
            )
        self.etp_data_parallel_size = self.tp_size // tp_size
        # Callers under an outer TP reduce-scatter keep the per-rank partial and
        # set this False; then ETP must equal TP so the outer reduce is complete.
        self.reduce_results = reduce_results

    @abstractmethod
    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate storage for the complete embedding weight."""
        raise NotImplementedError

    def dequantize(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Delegate storage-format conversion to the embedding method."""
        return self.embedding_method.dequantize(self, embeddings, output_dtype)

    def _get_dp_gather_slot(self, local_num_tokens: int) -> tuple[int, int]:
        """Return the per-DP slot size and this rank's slot offset."""
        if self.etp_data_parallel_size == 1:
            return local_num_tokens, 0
        dp_metadata: DPMetadata | None = get_forward_context().dp_metadata
        if dp_metadata is None:
            raise RuntimeError("ETP spanning DP requires DP token metadata")
        group_start = (self.data_parallel_rank // self.etp_data_parallel_size) * (
            self.etp_data_parallel_size
        )
        group_end = group_start + self.etp_data_parallel_size
        token_counts = dp_metadata.num_tokens_across_dp_cpu.tolist()
        group_counts = token_counts[group_start:group_end]
        slot_size = max(group_counts)
        dp_rank = get_dp_group().rank_in_group
        return slot_size, dp_rank * slot_size

    def _gather_dp_ids(
        self,
        ngram_ids: torch.Tensor,
        slot_size: int,
    ) -> torch.Tensor:
        """Gather DP-local IDs that share one ETP-sharded PLE table."""
        if self.etp_data_parallel_size == 1:
            return ngram_ids
        if ngram_ids.shape[0] < slot_size:
            padding = ngram_ids.new_zeros(
                slot_size - ngram_ids.shape[0], ngram_ids.shape[1]
            )
            ngram_ids = torch.cat((ngram_ids, padding), dim=0)
        return get_dp_group().all_gather(ngram_ids, dim=0)

    def _select_embeddings(
        self,
        embeddings: torch.Tensor,
        local_num_tokens: int,
        slot_offset: int,
    ) -> torch.Tensor:
        """Select this DP rank's rows from the ETP-reduced embeddings."""
        if self.etp_data_parallel_size == 1:
            return embeddings
        return embeddings.narrow(0, slot_offset, local_num_tokens)

    @abstractmethod
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Start an asynchronous lookup when supported."""
        raise NotImplementedError

    @abstractmethod
    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Return prefetched embeddings for the full input token set."""
        raise NotImplementedError


class Qwen4ExpPLEEmbeddingMethod(QuantizeMethodBase):
    """Quantization interface shared by resident and pinned PLE tables."""

    # PLE post-load processing only validates scales in their current storage.
    requires_device_loading: bool = False

    @staticmethod
    def from_quant_config(
        quant_config: QuantizationConfig | None,
        prefix: str,
        embedding_dtype: str | None = None,
    ) -> "Qwen4ExpPLEEmbeddingMethod":
        """Select the concrete PLE embedding format for a layer."""
        if embedding_dtype == "nvfp4":
            return Qwen4ExpPLENvFp4EmbeddingMethod()
        if embedding_dtype == "float8_e4m3fn":
            return Qwen4ExpPLEFp8EmbeddingMethod()
        if embedding_dtype is not None:
            raise ValueError(
                "ple_embedding_dtype must be 'nvfp4' or 'float8_e4m3fn', "
                f"got {embedding_dtype!r}"
            )
        if quant_config is None:
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(quant_config, ModelOptQuantConfigBase):
            if quant_config.is_layer_excluded(prefix):
                return Qwen4ExpPLEUnquantizedEmbeddingMethod()
            if isinstance(quant_config, ModelOptMixedPrecisionConfig):
                quant_algo = quant_config._resolve_quant_algo(prefix)
                if quant_algo == "FP8":
                    return Qwen4ExpPLEFp8EmbeddingMethod()
                if quant_algo == "NVFP4":
                    return Qwen4ExpPLENvFp4EmbeddingMethod()
                if quant_algo is not None:
                    raise ValueError(
                        f"Unsupported quant_algo {quant_algo!r} for the PLE embedding "
                        f"{prefix!r}; expected FP8 or NVFP4"
                    )
                return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if (
            isinstance(quant_config, CompressedTensorsConfig)
            and quant_config.get_scheme_dict(None, layer_name=prefix) is None
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if (
            isinstance(quant_config, INCConfig)
            and not quant_config.config_parser.resolve(None, prefix).quantized
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # Quark quantizes only Linear and MoE layers; PLE tables stay BF16.
        if isinstance(quant_config, QuarkConfig):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(quant_config, Fp8Config):
            ignored_layers = quant_config.ignored_layers
            if is_layer_skipped(
                prefix,
                ignored_layers,
                quant_config.packed_modules_mapping,
                match_mode=quant_config.ignored_layers_match_mode,
            ):
                return Qwen4ExpPLEUnquantizedEmbeddingMethod()
            # PLE checkpoint shards form one runtime embedding parameter.
            shard_prefix = f"{prefix}.shard_"
            if any(name.startswith(shard_prefix) for name in ignored_layers):
                return Qwen4ExpPLEUnquantizedEmbeddingMethod()
            if quant_config.is_checkpoint_fp8_serialized:
                return Qwen4ExpPLEFp8EmbeddingMethod()
            raise NotImplementedError(
                "Qwen4Exp PLE embedding only supports serialized FP8 checkpoints"
            )

        logger.warning(
            "Falling back to unquantized PLE embedding for %s "
            "with quantization config %s.",
            prefix,
            type(quant_config).__name__,
        )
        return Qwen4ExpPLEUnquantizedEmbeddingMethod()

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)

    @abstractmethod
    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert looked-up PLE rows to the activation dtype."""
        raise NotImplementedError


class Qwen4ExpPLEUnquantizedEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Unquantized PLE embedding storage and lookup semantics."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        weight = nn.Parameter(
            layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer, output_dtype
        return embeddings


class Qwen4ExpPLEFp8EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """FP8 PLE embedding with one global checkpoint scale."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject FP8 PLE checkpoints without a global scale."""
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale == sentinel):
            raise ValueError("FP8 PLE checkpoint is missing its global scale")
        # The weight cache IPC loader replaces the storage objects out of band
        # (by-value CPU copies); restore pinning and the UVA views. No-op when
        # the storage objects are the ones the views were built from.
        refresh = getattr(layer, "refresh_lookup_state", None)
        if refresh is not None:
            refresh()

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)


# Checkpoint key pattern of the PLE ngram-embedding CODE shards — the large
# packed PLE table (``...ple.ple_embedding.ngram_embedding.shard_<i>.weight``).
# When the NVFP4 sidecar is active this checkpoint data is redundant (the
# sidecar supplies the packed codes), so it can be read metadata-only to skip
# the ~51 GiB materialization.
_PLE_NGRAM_CODE_WEIGHT_RE = re.compile(
    r".*\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$"
)


def is_ple_ngram_code_weight(name: str) -> bool:
    """True if ``name`` is a PLE ngram-embedding code shard checkpoint weight."""
    return _PLE_NGRAM_CODE_WEIGHT_RE.match(name) is not None


def is_ple_ngram_code_weight_metadata_only(name: str) -> bool:
    """True if ``name``'s checkpoint data may be read metadata-only.

    Only true when the PLE NVFP4 sidecar is active (``VLLM_PLE_QUANT_DIR``
    set): the sidecar is the canonical PLE table, so the checkpoint's FP8 PLE
    codes are never consumed and need not be read from disk.
    """
    return bool(envs.VLLM_PLE_QUANT_DIR) and is_ple_ngram_code_weight(name)


class _PleNvfp4Sidecar:
    """Pre-quantized NVFP4 PLE sidecar reader (MagneticLab-compatible layout).

    One safetensors file per shard: ``weight_e2m1`` U8 [rows, dim//2] (packed
    low-nibble-first E2M1 codes), ``weight_scale`` F8_E4M3 [rows, dim//16]
    (per-16 group scales), and the ``weight_scale_2`` F32 scalar global scale,
    plus a ``META.json`` provenance record. Logical row g lives at row
    ``g % rows_per_shard`` of ``shard_{g // rows_per_shard}.safetensors``.
    Reads are CPU-only safe_open slices, so loading never touches the
    accelerator (unlike the boot-time FP8 re-quantization).
    """

    def __init__(self, root: str, *, total_rows: int, dim: int) -> None:
        self.root = root
        meta_path = os.path.join(root, "META.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"PLE NVFP4 sidecar {root}: missing META.json")
        with open(meta_path) as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError(
                f"PLE NVFP4 sidecar {root}: META.json must be a JSON object "
                f"(got {type(meta).__name__})"
            )
        raw_shards = meta.get("shards")
        raw_rows = meta.get("rows")
        for name, raw in (("shards", raw_shards), ("rows", raw_rows)):
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(
                    f"PLE NVFP4 sidecar {root}: META.json '{name}' must be "
                    f"an integer (got {raw!r})"
                )
        shards, rows = raw_shards, raw_rows
        # `rows` must divide evenly: the sidecar layout is fixed at build time
        # (the build tool guarantees an even division), so an uneven row count
        # is a corrupt or foreign META — fail fast rather than guess the last
        # shard's length.
        if shards <= 0 or rows <= 0 or rows % shards:
            raise ValueError(f"PLE NVFP4 sidecar {root}: invalid META.json {meta}")
        if rows != total_rows:
            raise ValueError(
                f"PLE NVFP4 sidecar row mismatch: sidecar has {rows} rows "
                f"but the checkpoint table has {total_rows}"
            )
        raw_width = meta.get("width", dim)
        if isinstance(raw_width, bool) or not isinstance(raw_width, int):
            raise ValueError(
                f"PLE NVFP4 sidecar {root}: META.json 'width' must be an "
                f"integer (got {raw_width!r})"
            )
        if raw_width != dim:
            raise ValueError(
                f"PLE NVFP4 sidecar width mismatch: sidecar width="
                f"{raw_width} but the table dim is {dim}"
            )
        self.shards = shards
        self.rows_per_shard = rows // shards
        for shard_idx in range(shards):
            path = os.path.join(root, f"shard_{shard_idx}.safetensors")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"PLE NVFP4 sidecar {root}: missing {path}")
        # Validate the layout against the actual first shard file.
        self._handles: dict[int, object] = {}
        first = self.open_shard(0)
        try:
            for key, want_dtype, want_shape in (
                ("weight_e2m1", "U8", (self.rows_per_shard, dim // 2)),
                ("weight_scale", "F8_E4M3", (self.rows_per_shard, dim // 16)),
            ):
                sl = first.get_slice(key)
                if sl.get_dtype() != want_dtype or tuple(sl.get_shape()) != want_shape:
                    raise ValueError(
                        f"PLE NVFP4 sidecar {key}: got dtype={sl.get_dtype()} "
                        f"shape={sl.get_shape()}, expected {want_dtype} {list(want_shape)}"
                    )
            ws2 = first.get_slice("weight_scale_2")
            if ws2.get_dtype() != "F32" or tuple(ws2.get_shape()) != ():
                raise ValueError(
                    f"PLE NVFP4 sidecar weight_scale_2: expected F32 scalar, "
                    f"got {ws2.get_dtype()} {ws2.get_shape()}"
                )
        except KeyError as e:
            raise ValueError(
                f"PLE NVFP4 sidecar shard 0 is missing required key {e} "
                f"(expected 'weight_e2m1', 'weight_scale', 'weight_scale_2')"
            ) from e
        self.global_scale = float(first.get_tensor("weight_scale_2").item())
        self._handles[0] = first

    def open_shard(self, shard_idx: int):
        # Handles are kept open for the process life on purpose: RL reloads
        # re-invoke the loader, which reads through them (closing them here
        # would break the reload path). Open-fd count per rank is bounded by
        # the sidecar shards its TP row span covers (~16-18 in the production
        # layout, x N PLE layers).
        handle = self._handles.get(shard_idx)
        if handle is None:
            path = os.path.join(self.root, f"shard_{shard_idx}.safetensors")
            handle = safe_open(path, framework="pt")
            self._handles[shard_idx] = handle
        return handle

    def copy_global_rows(
        self,
        global_row0: int,
        row_count: int,
        dest_codes: torch.Tensor,
        dest_scales: torch.Tensor,
    ) -> int:
        """Copy logical rows ``[global_row0, global_row0 + row_count)`` into the
        first ``row_count`` rows of the destinations (CPU -> pinned)."""
        row = 0
        while row < row_count:
            shard_idx, off = divmod(global_row0 + row, self.rows_per_shard)
            take = min(row_count - row, self.rows_per_shard - off)
            handle = self.open_shard(shard_idx)
            dest_codes[row : row + take].copy_(
                handle.get_slice("weight_e2m1")[off : off + take, :]
            )
            dest_scales[row : row + take].copy_(
                handle.get_slice("weight_scale")[off : off + take, :].view(torch.uint8)
            )
            row += take
        return row_count


class Qwen4ExpPLENvFp4EmbeddingMethod(Qwen4ExpPLEFp8EmbeddingMethod):
    """NVFP4 PLE storage: the FP8 checkpoint table is requantized at load into
    4-bit E2M1 codes (dim//2 bytes/row) + per-16 E4M3 group scales (dim//16
    bytes/row), halving the pinned footprint vs the FP8 table. The per-tensor
    ``weight_scale`` is carried as the NVFP4 global scale, so ``dequantize`` is
    unchanged (looked-up fp16 rows * weight_scale). Device-side lookups decode
    the packed storage in :meth:`embedding`; the pinned backend decodes with
    its own Triton kernel and never calls :meth:`embedding`.

    The per-16 group scales are computed at load (they are not in the FP8
    checkpoint) and are filled by the codes weight_loader as a side effect.

    With ``VLLM_PLE_QUANT_DIR`` set, the codes and scales instead come from a
    pre-quantized sidecar (``_PleNvfp4Sidecar``): the GPU re-quantization is
    skipped. The sidecar's F32 global scale is the decode scale of the
    sidecar's table: a process that loads the table from the sidecar adopts
    it (replacing the checkpoint FP8 scale, which does not pair with the
    sidecar's quantization); a process holding a non-sidecar table (e.g. one
    shipped by a provider running without the env var) fails loudly at PWL
    time instead of mispairing scales. With the env set the sidecar is the
    canonical frozen table: a layerwise (RL) reload re-reads it (idempotent)
    and deliberately does not apply an incoming PLE weight stream — the table
    is not trained through this path.
    """

    # IPC (weights_already_processed) context: the loader never runs in the
    # engine, so the adoption branch in process_weights_after_loading (the
    # only tensor transform) is unreachable there — the engine keeps the
    # provider's scale via the ADOPT-OR-RAISE verify/early-return instead. If
    # the adoption branch ever becomes reachable under pre-processed weights,
    # guard it on is_weights_pre_processed().
    supports_pre_processed_weights = True

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        rows = sum(output_partition_sizes)
        dim = input_size_per_partition
        # IPC weight-cache engine: the daemon owns the pinned PLE and ships
        # it by value in the cache state. A local pinned allocation here would
        # sit (transiently) next to the daemon's copy and the incoming copy,
        # tripling PLE host memory during the out-of-band swap and exhausting
        # the host: allocate Meta storage instead. The incoming copy replaces
        # it, and refresh_lookup_state (PWAL) pins the copy and builds the
        # UVA views.
        from vllm.config import get_current_vllm_config

        if get_current_vllm_config().load_config.load_format == "ipc_cache":
            codes_data = torch.empty(rows, dim // 2, dtype=torch.uint8, device="meta")
            scales = torch.empty(rows, dim // 16, dtype=torch.uint8, device="meta")
        elif os.environ.get("VLLM_WCM_PLE_SHM") == "1":
            # Weight-cache daemon with shared PLE: the PLE lives in named
            # /dev/shm files that engines map directly (zero copy, zero
            # re-pin, no by-value serialization). The daemon keeps the files
            # for its lifetime (ple_shm.close_ple_shm(unlink=True) on exit).
            from vllm.model_executor.model_loader.weight_cache.ple_shm import (
                allocate_ple_shm,
            )

            label = _ple_shm_label()
            codes_data = allocate_ple_shm(
                f"{label}_codes", (rows, dim // 2), torch.uint8
            )
            scales = allocate_ple_shm(
                f"{label}_scales", (rows, dim // 16), torch.uint8
            )
        else:
            codes_data = layer.allocate_embedding_weight(rows, dim // 2, torch.uint8)
            scales = layer.allocate_embedding_weight(rows, dim // 16, torch.uint8)
        # Registered (non-persistent) so the weight cache IPC daemon can export
        # it via named_buffers and the engine re-registers it: a plain
        # attribute would be invisible to the export and the engine would
        # decode with uninitialized scales. Non-persistent keeps it out of
        # state_dict and out of checkpoint loading.
        layer.register_buffer("nvfp4_scales", scales, persistent=False)

        # The group-scale destination as a slot so the loader keeps writing
        # into the buffer the UVA views point at: after the weight cache IPC
        # swap the original allocation is gone, and
        # refresh_lookup_state re-points the slot at the live buffer.
        scales_slot = [scales]
        self._ple_scale_slot = scales_slot
        # Set once the NVFP4 passthrough branch below lands its first shard:
        # tells process_weights_after_loading to keep the per-tensor reload
        # accounting instead of re-stamping the FP8 side-effect contract.
        passthrough = [False]
        self._ple_passthrough = passthrough
        # The group scales are never loaded through their own weight loader
        # (the codes loader fills them as a side effect); declare zero
        # loadable elements so the layerwise reload trigger counts only the
        # streamed parameters and releases their payload when they arrive.
        scales.weight_loader_numel = 0

        # VLLM_PLE_QUANT_DIR: optional directory of pre-quantized sidecar
        # shards. When set, rows are copied straight from the files (CPU-only,
        # no boot-time conversion) and the sidecar's global scale replaces the
        # checkpoint FP8 scale in process_weights_after_loading.
        sidecar_root = envs.VLLM_PLE_QUANT_DIR
        if sidecar_root:
            # One sidecar serves one PLE table. A model with several PLE
            # layers would load this same sidecar into every layer and
            # silently share one table (a shared ngram vocab passes the
            # per-layer row check), so refuse until per-layer directories
            # exist.
            from vllm.config import get_current_vllm_config
            from vllm.config.engram import engram_layer_ids

            layer_ids = engram_layer_ids(get_current_vllm_config().model_config)
            if layer_ids and len(layer_ids) > 1:
                raise ValueError(
                    "VLLM_PLE_QUANT_DIR serves a single PLE table, but this "
                    f"model has {len(layer_ids)} PLE layers ({layer_ids}). "
                    "Provide one sidecar directory per layer before enabling "
                    "the sidecar."
                )
        self._ple_sidecar: _PleNvfp4Sidecar | None = None
        if sidecar_root:
            self._ple_sidecar = _PleNvfp4Sidecar(
                sidecar_root, total_rows=layer.org_vocab_size, dim=dim
            )
            logger.info(
                "NVFP4 PLE: sidecar pre-quantization enabled (%s): %d shards x "
                "%d rows, global scale %.6g; boot-time FP8 re-quantization skipped",
                sidecar_root,
                self._ple_sidecar.shards,
                self._ple_sidecar.rows_per_shard,
                self._ple_sidecar.global_scale,
            )

        def nvfp4_loader(param, loaded_weight, checkpoint_start=None):
            if checkpoint_start is None:
                raise ValueError("NVFP4 PLE requires row-sharded checkpoint weights")
            # Resolve the live group-scale buffer at call time (see
            # _ple_scale_slot): a post-swap reload must land in the storage
            # the UVA views read, not the superseded original allocation.
            scales = scales_slot[0]
            overlap = layer._compute_shard_overlap(
                checkpoint_start=checkpoint_start,
                checkpoint_rows=loaded_weight.shape[0],
            )
            if overlap is None:
                return 0
            dst = overlap.destination_start
            cnt = overlap.row_count
            # NVFP4 passthrough (sm70/08, 2026-10-18): this checkpoint shard is
            # ALREADY the runtime storage format — U8 E2M1 codes at dim//2
            # bytes/row, with the per-16 E4M3 group scales in a separate
            # `shard_i.weight_scale` tensor (loaded by nvfp4_scale_loader
            # below). Copy the rows and return; nothing is requantized and the
            # group-scale side-effect copy of the FP8 path does not apply.
            #
            # Discriminator: pre-quantized codes are uint8 at HALF the
            # embedding dim. The FP8 table this method was written for is one
            # byte per element at the FULL dim, so it can never match and the
            # FP8 / sidecar paths below stay byte-identical.
            if (
                loaded_weight.dtype == torch.uint8
                and loaded_weight.ndim == 2
                and loaded_weight.shape[1] == dim // 2
            ):
                src = loaded_weight.narrow(0, overlap.source_start, cnt)
                with torch.no_grad():
                    param.narrow(0, dst, cnt).copy_(src.to(device=param.device))
                if not passthrough[0]:
                    # Per-tensor ownership under passthrough: each loader
                    # counts the tensor it lands (codes here, group scales in
                    # nvfp4_scale_loader). The SUM is unchanged — every row is
                    # still dim/2 + dim/16 elements — which is the quantity the
                    # layerwise trigger compares, so both contracts agree.
                    param.weight_loader_numel = param.numel()
                    scales_slot[0].weight_loader_numel = scales_slot[0].numel()
                    passthrough[0] = True
                    logger.info(
                        "NVFP4 PLE: PASSTHROUGH load enabled — the checkpoint "
                        "shards are already runtime NVFP4 (row-for-row copy, no "
                        "re-quantization, no sidecar)"
                    )
                return cnt
            sidecar = self._ple_sidecar
            if sidecar is not None:
                # Pre-quantized path: pure CPU copy of the already-packed
                # rows. The sidecar is indexed by global row exactly like the
                # checkpoint, so the mapping is row-identical.
                #
                # The sidecar is the CANONICAL frozen table for this model:
                # the incoming `loaded_weight` is deliberately not applied,
                # including during a layerwise (RL) reload (idempotent
                # re-read). The PLE table is not trained through this path —
                # applying a requantized stream would swap the runtime table
                # for a different quantization (cos ~0.995) and silently
                # shift decode quality. If PLE ever becomes RL-trainable,
                # gate this branch to boot time only.
                copied = sidecar.copy_global_rows(
                    checkpoint_start + overlap.source_start,
                    cnt,
                    param.narrow(0, dst, cnt),
                    scales.narrow(0, dst, cnt),
                )
                assert copied == cnt, (
                    "sidecar returned fewer rows than requested — a corrupt "
                    "sidecar must not load partially"
                )
                # This process's table came from the sidecar: process_weights
                # after_loading adopts the sidecar's global scale (the
                # checkpoint's FP8 scale does not pair with this table).
                self._ple_sidecar_loaded = True
                return cnt
            # Shared PLE storage (weight-cache daemon shm) is FROZEN: re-
            # quantized bytes must not be written into the daemon's shared
            # table, which is visible to every attached engine. The sidecar
            # branch above is the only sanctioned writer (idempotent,
            # identical bytes).
            if getattr(param, "_wcm_ple_shm", None):
                raise RuntimeError(
                    "refusing to write re-quantized PLE into shared daemon "
                    "shm storage; PLE reloads of shared storage require the "
                    "sidecar path (VLLM_PLE_QUANT_DIR)"
                )
            # Re-quantization is a BOOT-TIME-only path: a layerwise (RL)
            # reload probes this loader on META tensors first (the arrival
            # count), and the re-quantization device handling cannot run
            # there (outputs route through the param's meta device, and a
            # copy out of meta is not allowed). The reload-safe path is the
            # sidecar branch above — fail loudly with the remedy:
            if param.is_meta:
                raise ValueError(
                    "NVFP4 PLE re-quantization ran inside a layerwise "
                    "(RL) reload, where the loader is probed on meta "
                    "tensors and cannot requantize. RL reloads of the PLE "
                    "table require the sidecar path: set VLLM_PLE_QUANT_DIR "
                    "to the pre-quantized sidecar directory."
                )
            src = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
            # Requant in row-chunks on the accelerator: the argmin intermediates
            # scale ~10 KB/row, so a full ~1.2M-row PLE shard needs ~12 GiB and
            # OOMs a 32GB V100. A 128K-row chunk bounds the peak to ~1.3 GiB.
            # Pinned storage is a runtime concern, so load may use the GPU.
            chunk = 131072
            qdev = torch.device("cuda") if torch.cuda.is_available() else param.device
            for i in range(0, cnt, chunk):
                c, sc = nvfp4_from_fp8(
                    src[i : i + chunk].to(device=qdev, dtype=torch.float16)
                )
                param.narrow(0, dst + i, c.shape[0]).copy_(c.to(device=param.device))
                scales.narrow(0, dst + i, sc.shape[0]).copy_(
                    sc.to(device=param.device)
                )
            return cnt

        def nvfp4_scale_loader(loaded_weight, checkpoint_start=None):
            """Land a pre-quantized per-16 E4M3 group-scale shard (passthrough).

            The checkpoint carries ``shard_i.weight_scale`` as float8_e4m3fn
            [rows, dim//16]; the runtime buffer holds exactly those bytes as
            uint8, so the copy reinterprets rather than converts. The row
            mapping is the codes loader's algorithm applied to this tensor, so
            codes and scales may arrive in any order (they never wait on each
            other).
            """
            if checkpoint_start is None:
                raise ValueError(
                    "NVFP4 PLE group scales require row-sharded weights"
                )
            scales = scales_slot[0]
            overlap = layer._compute_shard_overlap(
                checkpoint_start=checkpoint_start,
                checkpoint_rows=loaded_weight.shape[0],
            )
            if overlap is None:
                return 0
            src = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
            dst = scales.narrow(
                0, overlap.destination_start, overlap.row_count
            )
            with torch.no_grad():
                dst.copy_(src.view(torch.uint8).to(device=dst.device))
            return overlap.row_count

        # Kept on the method (it survives the out-of-band Parameter swap of
        # the weight cache IPC loader, which the tensor attributes do not) so
        # process_weights_after_loading can re-attach them to the swapped
        # parameters: a later layerwise reload must replay through these
        # loaders, not the generic default_weight_loader.
        self._ple_codes_loader = nvfp4_loader
        self._ple_scale_loader = weight_loader
        self._ple_nvfp4_scale_loader = nvfp4_scale_loader
        # Also on the layer module: NVGramEmbedding.load_weights routes
        # `shard_i.weight_scale` to this loader and can only rely on attributes
        # that survive the out-of-band IPC Parameter swap (module attributes do;
        # Parameter weight_loader attributes do not).
        layer._ple_nvfp4_scale_loader = nvfp4_scale_loader

        codes = ModelWeightParameter(
            data=codes_data,
            input_dim=1,
            output_dim=0,
            weight_loader=nvfp4_loader,
        )
        layer.register_parameter("weight", codes)
        # Parameter wrapping drops tensor-level attributes (the _wcm_ple_shm
        # path set by allocate_ple_shm): re-stamp it on the registered
        # parameter so the cache export (TensorEntry.from_tensor) and the
        # reload guard can see the shared storage. The handle must move with
        # it too — exit cleanup walks the registered parameters, and a path
        # without its handle silently skips the file's unlink.
        codes_shm = getattr(codes_data, "_wcm_ple_shm", None)
        if codes_shm:
            from vllm.model_executor.model_loader.weight_cache.ple_shm import (
                mark_ple_shm,
            )

            mark_ple_shm(codes, codes_shm)
            codes._wcm_ple_shm_handle = codes_data._wcm_ple_shm_handle
            codes_data._wcm_ple_shm_handle = None
        # CopyCounter counts EVERY copy_ inside the loader — the group-scale
        # side-effect copy included — so each streamed shard loads
        # cnt*(dim/2 + dim/16) elements, not cnt*dim/2 (verified for BOTH the
        # sidecar and the re-quantization branches: test_ple_ipc_path.py
        # 6c/6g). Declare the true loadable size (codes + group scales) so
        # the layerwise inline trigger's target matches the actual count:
        # with an under-declared target the trigger fires before the last of
        # K >= 9 shards streams, and the tail shards are then dropped as
        # "Excessive loading" (silent table-tail loss).
        codes.weight_loader_numel = codes.numel() + scales.numel()
        # NOTE: the sharded-RDT bake's fake-tensor recorder only records
        # copies whose SOURCE is a fake — the sidecar branch's sources are
        # real, so under RDT the PLE module never records a complete load
        # and stays on the slow path (its ~3.6 GB/rank payload is streamed
        # and discarded on every update — consistent with the frozen-table
        # contract; excluding it from the RDT stream is a trainer-side
        # change, deferred).
        weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            dim,
            None,
            weight_loader,
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale", weight_scale)
        logger.info(
            "NVFP4 PLE: per-rank pinned codes=%s (%.2f GB) + per-16 scales=%s "
            "(%.2f GB) = %.2f GB pinned vs %.2f GB FP8 (ratio 0.5625)",
            tuple(codes.data.shape), codes.data.numel() * 1e-9,
            tuple(scales.shape), scales.numel() * 1e-9,
            (codes.data.numel() + scales.numel()) * 1e-9,
            rows * dim * 1e-9,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # The IPC loader replaces the storage objects out of band with plain
        # Parameters, dropping the weight_loader attributes; re-attach the
        # captured loaders so a later layerwise reload (RL) replays through the
        # real PLE loaders (requantization / sidecar copy + group-scale side
        # effect), not the generic default_weight_loader. No-op on a plain
        # disk boot (same function re-set on the same Parameter).
        codes_loader = getattr(self, "_ple_codes_loader", None)
        if codes_loader is not None:
            weight = getattr(layer, "weight", None)
            if weight is not None:
                weight.weight_loader = codes_loader
        scale_loader = getattr(self, "_ple_scale_loader", None)
        if scale_loader is not None:
            weight_scale = getattr(layer, "weight_scale", None)
            if weight_scale is not None:
                weight_scale.weight_loader = scale_loader
        weight = getattr(layer, "weight", None)
        scales_buf = getattr(layer, "nvfp4_scales", None)
        if weight is not None and scales_buf is not None:
            # The IPC swap strips the reload-accounting declarations set in
            # create_weights (weight_loader_numel); re-stamp them — without
            # them the engine's per-layer total no longer matches what the
            # loaders actually count, so the inline trigger never fires and
            # the PLE layer is only processed by the delayed
            # finalize_layerwise_reload at the end of the update. Idempotent
            # on a plain disk boot (same values create_weights already set).
            # (During a layerwise reload this PWL runs on the MATERIALIZED
            # transients, about to be discarded — the stamps there are
            # harmless; the kernel tensors keep their own identical stamps,
            # re-captured at reload init. Do not "optimize" this away.)
            weight.weight_loader_numel = weight.numel() + scales_buf.numel()
            scales_buf.weight_loader_numel = 0
            if getattr(self, "_ple_passthrough", [False])[0]:
                # Passthrough (sm70/08): each loader owns its own tensor — the
                # codes loader no longer fills the group scales as a side
                # effect (nvfp4_scale_loader lands them). The SUM is identical
                # to the FP8 contract above, and the sum is what the layerwise
                # trigger compares, so the target is unchanged either way; only
                # the per-tensor attribution differs.
                weight.weight_loader_numel = weight.numel()
                scales_buf.weight_loader_numel = scales_buf.numel()
        sidecar = getattr(self, "_ple_sidecar", None)
        if sidecar is None:
            return
        if getattr(self, "_ple_sidecar_loaded", False):
            # This process's table came from the sidecar (a disk boot, or an
            # RL reload replayed through the sidecar loader). The checkpoint's
            # FP8 global scale was loaded with it but does not pair with the
            # sidecar's quantization — adopt the sidecar's (decode contract:
            # E2M1 x E4M3-group x this global scale).
            with torch.no_grad():
                layer.weight_scale.data.fill_(sidecar.global_scale)
            logger.info(
                "NVFP4 PLE: runtime weight_scale = %.6g (adopted from the "
                "sidecar)",
                sidecar.global_scale,
            )
            return
        # The table did NOT come from the sidecar in this process (e.g. it was
        # shipped by a provider running without VLLM_PLE_QUANT_DIR): its scale
        # belongs to its own quantization, so adopting the sidecar's scale
        # would silently mispair. The provider must agree with us — verify
        # loudly instead.
        shard = getattr(layer, "shard_indices", None)
        owns_rows = (
            shard is None
            or shard.org_vocab_start_index < shard.org_vocab_end_index
        )
        if (
            getattr(layer, "weight", None) is None
            or layer.weight.numel() == 0
            or not owns_rows
        ):
            # A rank that owns no org-rows (a degenerate TP range — its
            # padded allocation can still have numel > 0) cannot be
            # mispaired with anything; do not fail such a rank.
            return
        current = float(layer.weight_scale)
        if current != sidecar.global_scale:
            raise ValueError(
                "NVFP4 PLE sidecar mismatch: runtime weight_scale is "
                f"{current!r} but the sidecar's global scale is "
                f"{sidecar.global_scale!r}, and this process's table was not "
                "loaded from the sidecar. The weight provider and this "
                "process must use the same VLLM_PLE_QUANT_DIR."
            )

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Decode NVFP4 rows for device-side lookup.

        ``F.embedding`` on the packed ``weight`` would return raw E2M1 code
        bytes (dim//2 per row), so unpack the referenced rows here: each
        nibble pair decodes through the E2M1 LUT and the per-16 E4M3 group
        scale is multiplied back — the same fp16 contract as the pinned NVFP4
        lookup kernel. ``dequantize`` still applies the global ``weight_scale``.
        """
        codes = F.embedding(input_, layer.weight)        # [..., dim//2] uint8
        scales = F.embedding(input_, layer.nvfp4_scales)  # [..., dim//16] uint8
        e2 = nvfp4_e2m1_lut(codes.device)
        e4 = nvfp4_e4m3_lut(codes.device)
        half = codes.shape[-1]
        dim = half * 2
        vals = torch.empty(
            *codes.shape[:-1], dim, dtype=torch.float16, device=codes.device
        )
        vals[..., 0::2] = e2[(codes & 0x0F).long()]  # even d = low nibble
        vals[..., 1::2] = e2[(codes >> 4).long()]   # odd d = high nibble
        sf = e4[scales.long()]                      # [..., dim//16] fp16
        vals.view(*codes.shape[:-1], dim // 16, 16).mul_(sf.unsqueeze(-1))
        return vals


def _ple_shm_label() -> str:
    """Label for this worker's PLE shm files (readable, unique per process)."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
    except Exception:
        rank = os.getpid()
    return f"r{rank}_{os.getpid()}"


class Qwen4ExpPLEDeviceEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table allocated on the active model device."""

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight on the active device."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype)

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Resident embedding prefetch is a no-op."""
        return None

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Reject prefetch finalization for resident embeddings."""
        raise RuntimeError("Device PLE embedding does not support prefetch")

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP inputs, look up embeddings, and select local rows."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = super().forward(gathered_ids)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )


_FP8_LUT_CACHE: dict[torch.dtype, torch.Tensor] = {}


def _get_fp8_decode_lut(dtype: torch.dtype, device) -> torch.Tensor:
    """256-entry exact code->value table for FP8 storage on GPUs whose Triton
    cannot compile fp8e4nv (SM70/Volta). Built from PyTorch's own cast."""
    cached = _FP8_LUT_CACHE.get(dtype)
    if cached is None:
        codes = torch.arange(256, dtype=torch.uint8)
        cached = codes.view(dtype).to(torch.float16).contiguous()
        _FP8_LUT_CACHE[dtype] = cached
    return cached.to(device)


@triton.jit
def _lookup_ple_embedding_from_pinned_kernel(
    weight_ptr: tl.pointer_type(tl.uint8),  # type: ignore[valid-type]
    ids_ptr,
    output_ptr: tl.pointer_type(tl.uint8),  # type: ignore[valid-type]
    row_bytes,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Copy TP-owned PLE rows as raw bytes from a CUDA view of pinned host memory.

    Byte pointers keep the storage dtype out of the kernel signature, so
    dtypes Triton cannot lower on the GPU (FP8 E4M3FN before SM89) still work.
    """
    row_id = tl.program_id(0).to(tl.int64)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < row_bytes
    load_mask = store_mask & in_range
    values = tl.load(
        weight_ptr + local_idx * row_bytes + offsets,
        mask=load_mask,
        other=0,
    )
    tl.store(
        output_ptr + row_id * row_bytes + offsets,
        values,
        mask=store_mask,
    )


@triton.jit
def _lookup_ple_embedding_nvfp4_kernel(
    codes_ptr,
    scales_ptr,
    ids_ptr,
    output_ptr,
    e2_lut_ptr,
    e4_lut_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Look up TP-owned PLE rows stored as NVFP4 (packed E2M1 codes + per-16 E4M3
    group scales) through a CUDA view of pinned host memory. Decodes each
    referenced row to fp16 = E2M1(code) * E4M3(group scale); the per-tensor global
    scale is applied by the caller's dequantize, as with the FP8 PLE path."""
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < embedding_dim
    load_mask = store_mask & in_range
    code_byte = tl.load(
        codes_ptr + local_idx * (embedding_dim // 2) + offsets // 2,
        mask=load_mask,
        other=0,
    ).to(tl.int32)
    code = tl.where(offsets % 2 == 0, code_byte & 0xF, code_byte >> 4)
    e2v = tl.load(e2_lut_ptr + code, mask=load_mask, other=0.0)
    sf_byte = tl.load(
        scales_ptr + local_idx * (embedding_dim // 16) + offsets // 16,
        mask=load_mask,
        other=0,
    ).to(tl.int32)
    sfv = tl.load(e4_lut_ptr + sf_byte, mask=load_mask, other=0.0)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        e2v * sfv,
        mask=store_mask,
    )


@triton.jit
def _lookup_ple_embedding_lut_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    lut_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Decode TP-owned FP8 PLE rows through a 256-entry code->value LUT so Volta
    (SM70), whose Triton has no fp8e4nv type, can read an FP8 PLE table without
    upcasting the whole table in memory. weight_ptr is a uint8 view of the
    storage bytes; the output is fp16."""
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < embedding_dim
    load_mask = store_mask & in_range
    codes = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=load_mask,
        other=0,
    )
    values = tl.load(lut_ptr + codes.to(tl.int32), mask=load_mask, other=0.0)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        values,
        mask=store_mask,
    )


class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table loaded into pinned CPU memory and looked up through UVA."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
    ) -> None:
        if not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            reduce_results=reduce_results,
        )
        self._build_lookup_state()
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetch_buffer: torch.Tensor | None = None
        # ③ PLE prefetch double-buffer (VLLM_SM70_PLE_DBUF=1, default 0 = today).
        # The single persistent _prefetch_buffer is what made the overlapped
        # side stream unsafe: step k+1's lookup writes it while step k's
        # _finalize_prefetch may still read it. Two buffers alternating per
        # prefetch remove the hazard by construction, so the overlap can be
        # re-enabled without the PLE_SYNC serialization.
        self._prefetch_dbuf = int(os.environ.get("VLLM_SM70_PLE_DBUF", "0")) == 1
        self._prefetch_buffers: list[torch.Tensor] | None = None
        self._prefetch_slot = 0
        self._prefetch_active: torch.Tensor | None = None
        self._prefetch_alloc_lock = threading.Lock()
        self._prefetch_rows = max_total_tokens * self.etp_data_parallel_size
        self._num_ngram_heads = num_ngram_heads
        self._output_dim = num_ngram_heads * self.embedding_dim

    def _build_lookup_state(self) -> None:
        """Build (or rebuild) the UVA lookup views over the current storage."""
        # IPC weight-cache mode starts with Meta storage (see create_weights):
        # UVA registration needs a real CPU tensor, so the views are built by
        # refresh_lookup_state (PWAL) once the daemon's copy is in place.
        if self.weight.device.type == "meta":
            return
        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)
        self._row_bytes = self.embedding_dim * self.weight.element_size()
        self._block_d = triton.next_power_of_2(self._row_bytes)
        self._nvfp4_scales = self._ple_nvfp4_scales()
        if self._nvfp4_scales is not None:
            # NVFP4: weight is packed E2M1 codes; decode alongside per-16 E4M3
            # scales via the LUT kernel (no fp8 LUT; the kernel emits fp16).
            self._is_nvfp4 = True
            self._decode_lut = None
            self._lookup_dtype = torch.float16
            self._lookup_weight = self._uva_weight
            self._lookup_scales = get_accelerator_view_from_cpu_tensor(
                self._nvfp4_scales
            )
            self._e2_lut = nvfp4_e2m1_lut(self._uva_weight.device)
            self._e4_lut = nvfp4_e4m3_lut(self._uva_weight.device)
            # The NVFP4 kernel iterates over embedding_dim elements, so its
            # BLOCK_D must cover that, not the dim//2-byte packed row.
            self._block_d = triton.next_power_of_2(self.embedding_dim)
        else:
            self._is_nvfp4 = False
            self._decode_lut = None
            if self.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and (
                not current_platform.has_device_capability(89)
            ):
                self._decode_lut = _get_fp8_decode_lut(
                    self.weight.dtype, self._uva_weight.device
                )
            self._lookup_dtype = (
                torch.float16 if self._decode_lut is not None else self.weight.dtype
            )
            self._lookup_weight = (
                self._uva_weight.view(torch.uint8)
                if self._decode_lut is not None
                else self._uva_weight
            )
        self._lookup_state_key = self._current_lookup_state_key()

    def _ple_nvfp4_scales(self) -> torch.Tensor | None:
        """The NVFP4 group-scale storage (a non-persistent buffer), if any."""
        return getattr(self, "nvfp4_scales", None)

    def _current_lookup_state_key(self) -> tuple:
        """Identity of the storage the UVA views should point at. Includes the
        storage address (not just the object id): an object id can be reused
        after GC, but a live storage address cannot (the UVA views keep the
        allocation alive)."""
        scales = self._ple_nvfp4_scales()
        return (
            id(self.weight),
            self._storage_id(self.weight),
            (id(scales), self._storage_id(scales)) if scales is not None else (0, 0),
        )

    @staticmethod
    def _storage_id(tensor: torch.Tensor) -> int:
        try:
            return tensor.untyped_storage().data_ptr()
        except (RuntimeError, ValueError):
            return 0

    def refresh_lookup_state(self) -> None:
        """Restore pinned storage and UVA views after the weight objects were
        replaced out of band by the weight cache IPC loader.

        The IPC transport ships CPU tensors by value, so the engine receives
        plain (non-pinned) copies that replace the original pinned
        allocations, and the views built in ``__init__`` dangle. Re-pin in
        place and rebuild the views.

        No-ops: when the current storage objects are the ones the views were
        built from, or when the storage is not on CPU — layerwise reload
        materializes transient copies on the layer during processing (on the
        process-default device, usually CPU) and then copies the values back
        in place into the original pinned storage, where the original views
        remain valid (a CPU view of a CUDA tensor would also hard-fail).
        """
        scales = self._ple_nvfp4_scales()
        if self.weight.device.type != "cpu" or (
            scales is not None and scales.device.type != "cpu"
        ):
            return
        # Mid layerwise reload: the layer temporarily holds the materialized
        # transient objects. Refreshing now would re-pin the transients and
        # rebuild the views on them, but the reload machinery then copies the
        # values back into the ORIGINAL kernel tensors and re-registers them —
        # and the group scales never go through the reload at all (the codes
        # loader writes them through _ple_scale_slot) — so the views would be
        # stranded on transients whose scales buffer was never loaded. The
        # views built on the originals stay valid; skip. (get_layerwise_info
        # creates the layer's weak entry as a side effect when absent —
        # harmless: kernel_tensors is None for models that never reload.)
        from vllm.model_executor.model_loader.reload.layerwise import (
            get_layerwise_info,
        )

        if get_layerwise_info(self).kernel_tensors is not None:
            return
        if self._current_lookup_state_key() == getattr(self, "_lookup_state_key", None):
            return
        if not self.weight.is_pinned():
            pinned = torch.empty_like(self.weight, pin_memory=True)
            pinned.copy_(self.weight)
            self.weight.data = pinned
        if scales is not None:
            if not scales.is_pinned():
                pinned_scales = torch.empty_like(scales, pin_memory=True)
                pinned_scales.copy_(scales)
                scales.data = pinned_scales
            # The IPC loader re-registers buffers as persistent; restore the
            # daemon-side non-persistent flag so state_dict and layerwise
            # reload (which snapshots _non_persistent_buffers_set) treat the
            # scales identically on both sides.
            self.register_buffer("nvfp4_scales", scales, persistent=False)
            # The nvfp4_loader closure writes group scales through a slot;
            # re-point it at the live buffer so post-swap loads (e.g. a
            # layerwise RL reload) land in the storage the UVA views read.
            method = getattr(self, "quant_method", None)
            slot = getattr(method, "_ple_scale_slot", None)
            if slot is not None:
                slot[0] = scales
        self._build_lookup_state()

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight directly in pinned CPU memory."""
        return torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up local ETP rows, decoding FP8/NVFP4 storage to fp16 where needed."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=self._lookup_dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != self._lookup_dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, weight dtype, "
                "and input device"
            )

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            if self._is_nvfp4:
                _lookup_ple_embedding_nvfp4_kernel[(flat_ids.numel(),)](
                    self._lookup_weight,
                    self._lookup_scales,
                    flat_ids,
                    output,
                    self._e2_lut,
                    self._e4_lut,
                    self.embedding_dim,
                    self.shard_indices.org_vocab_start_index,
                    self.shard_indices.org_vocab_end_index,
                    BLOCK_D=self._block_d,
                )
            elif self._decode_lut is not None:
                _lookup_ple_embedding_lut_kernel[(flat_ids.numel(),)](
                    self._lookup_weight,
                    flat_ids,
                    output,
                    self._decode_lut,
                    self.embedding_dim,
                    self.shard_indices.org_vocab_start_index,
                    self.shard_indices.org_vocab_end_index,
                    BLOCK_D=self._block_d,
                )
            else:
                _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                    self._uva_weight,
                    flat_ids,
                    output,
                    self._row_bytes,
                    self.shard_indices.org_vocab_start_index,
                    self.shard_indices.org_vocab_end_index,
                    BLOCK_D=self._block_d,
                )
        return output

    def sync_lookup(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Synchronous UVA lookup for platforms without prefetch wiring."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = self._lookup(gathered_ids)
        if self.reduce_results:
            embeddings = self._reduce_etp_embeddings(embeddings)
        else:
            # The outer TP reduce-scatter requires ETP=TP.
            assert self.etp_data_parallel_size == 1
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )

    def _reduce_etp_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Combine pinned lookup results owned by different ETP ranks."""
        if self.tp_size == 1:
            return embeddings
        assert self.parallel_group is not None
        if embeddings.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Each vocabulary row has one owner, so reduce the raw FP8 bytes.
            reduced = self.parallel_group.all_reduce(embeddings.view(torch.int8))
            return reduced.view(embeddings.dtype)
        return self.parallel_group.all_reduce(embeddings)

    @eager_break_during_capture
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Gather ETP IDs and launch their UVA lookup on the side stream.

        VLLM_SM70_PLE_SYNC=1 runs the lookup on the CURRENT stream instead. This is
        the controlled switch for the TP4xPP2 illegal-memory-access hunt: that shape
        boots READY under CUDA_LAUNCH_BLOCKING=1 and dies in the warmup decode step
        without it, which is the signature of a cross-stream ordering bug rather than
        a bounds bug -- and this side stream is the only place the model launches work
        ahead of the stream that consumes it (`_prefetch_buffer` is a single persistent
        buffer, and step k+1's lookup writes it while step k's `_finalize_prefetch` may
        still be reading it). If serializing makes PP2 boot clean without launch
        blocking, the fault is inside this region.
        """
        if getattr(self, "_uva_weight", None) is None:
            # The disk-load path never runs the weight-cache refresh hook, so the
            # UVA views are missing and the allocation below would raise
            # AttributeError. Build them from the current storage here.
            self.refresh_lookup_state()
        buffer = self._prefetch_buffer
        if buffer is None:
            # First use allocates. The eager profile run always precedes
            # cudagraph capture, so allocation never happens mid-capture;
            # the lock keeps concurrent first callers from tearing the
            # stream/buffer pair.
            with self._prefetch_alloc_lock:
                buffer = self._prefetch_buffer
                if buffer is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "pinned PLE prefetch buffer must be allocated "
                            "eagerly, before cudagraph capture"
                        )
                    self._prefetch_stream = torch.cuda.Stream(
                        device=self._uva_weight.device
                    )
                    buffer = torch.empty(
                        self._prefetch_rows,
                        self._num_ngram_heads,
                        self.embedding_dim,
                        dtype=self._lookup_dtype,
                        device=self._uva_weight.device,
                    )
                    self._prefetch_buffer = buffer
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE prefetch stream was not allocated")
        if self._prefetch_dbuf:
            bufs = self._prefetch_buffers
            if bufs is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "PLE double buffer must be allocated eagerly, "
                        "before cudagraph capture"
                    )
                with self._prefetch_alloc_lock:
                    bufs = self._prefetch_buffers
                    if bufs is None:
                        bufs = [
                            self._prefetch_buffer,
                            torch.empty_like(self._prefetch_buffer),
                        ]
                        self._prefetch_buffers = bufs
            buffer = bufs[self._prefetch_slot & 1]
            self._prefetch_slot += 1
        self._prefetch_active = buffer
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        if gathered_ids.shape[0] > buffer.shape[0]:
            raise ValueError(
                f"pinned PLE prefetch buffer holds {buffer.shape[0]} rows, "
                f"but the batch needs {gathered_ids.shape[0]}"
            )
        active_output = buffer[: gathered_ids.shape[0]]
        if _ple_sync_requested() and not self._prefetch_dbuf:
            self._lookup(gathered_ids, output=active_output)
            return
        prefetch_stream.wait_stream(torch.cuda.current_stream())
        gathered_ids.record_stream(prefetch_stream)
        with torch.cuda.stream(prefetch_stream):
            self._lookup(gathered_ids, output=active_output)

    @eager_break_during_capture
    def _finalize_prefetch(
        self,
        prefetch_output: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Join the side stream, reduce ETP shards, and select local rows."""

        def _marker_break() -> None:
            # Temporary diagnostic (SM70 PW-graph IMA bisect, default off).
            if os.getenv("VLLM_SM70_DEBUG_PLE_BREAKS", "0") != "1":
                return
            cap = BreakableCUDAGraphCapture.current()
            if cap is not None and cap._capturing:
                cap.add_eager(lambda: None)

        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE finalize requires a prior start_prefetch")
        # VLLM_SM70_PLE_SYNC: with the lookup serialized onto the current
        # stream there is no side-stream work to join (and joining an idle
        # uncaptured stream would invalidate breakable-cudagraph capture).
        # With VLLM_SM70_PLE_DBUF=1 the lookup always runs on the side stream
        # (that is the point), so the join is required even under PLE_SYNC=1.
        if not _ple_sync_requested() or self._prefetch_dbuf:
            torch.cuda.current_stream().wait_stream(prefetch_stream)
        _marker_break()
        slot_size, slot_offset = self._get_dp_gather_slot(output.shape[0])
        active_output = prefetch_output[: slot_size * self.etp_data_parallel_size]
        if self.reduce_results:
            embeddings = self._reduce_etp_embeddings(active_output)
        else:
            # The outer TP reduce-scatter requires ETP=TP.
            assert self.etp_data_parallel_size == 1
            embeddings = active_output
        _marker_break()
        embeddings = self._select_embeddings(
            embeddings,
            output.shape[0],
            slot_offset,
        )
        _marker_break()
        output.copy_(embeddings.flatten(-2))

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Finish the pinned lookup for ``total_tokens`` rows."""
        active = getattr(self, "_prefetch_active", None)
        buffer = active if active is not None else self._prefetch_buffer
        if buffer is None:
            raise RuntimeError("pinned PLE lookup requires a prior start_prefetch")
        output = buffer.new_empty((total_tokens, self._output_dim))
        self._finalize_prefetch(buffer, output)
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Finish the pinned lookup into graph-owned output storage."""
        if os.getenv("VLLM_SM70_DEBUG_PLE_EAGER_FWD", "0") == "1":
            _cap = BreakableCUDAGraphCapture.current()
            if _cap is not None:
                logger.warning(
                    "PLE_EAGER_FWD probe: capturing=%s",
                    _cap._capturing,
                )
        return self.finalize_prefetch(hidden_states.shape[0])
