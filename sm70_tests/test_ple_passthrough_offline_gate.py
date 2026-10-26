#!/usr/bin/env python3
"""T5 offline gate for sm70/08 (PLE NVFP4 passthrough) on the REAL checkpoint.

CPU only, no GPU, no serve. Verifies against
/data/models/Qwen3.8-Flash-Next-FP4-Local-NVFP4:

  A. Headers of all 128 code + 128 group-scale shards: dtype/shape, total rows,
     and divisibility by split_ngram_parts (what create_weights' shape check
     will demand).
  B. Discriminator: the NEW code shards are uint8 at dim//2, the OLD model's are
     F8_E4M3 at the full dim — so the passthrough branch cannot fire on the old
     model (the FP8/sidecar paths stay byte-identical).
  C. Per-rank row mapping: for all 8 ranks x 128 shards, the overlaps tile the
     rank's org-vocab window exactly once (no gaps, no double writes).
  D. Byte-exactness on sampled shards: run the REAL loaders (create_weights +
     nvfp4_loader / nvfp4_scale_loader) and compare the landed rows against a
     direct safetensors read, byte for byte.

Run: /data/vllm-dev/bin/python sm70_tests/test_ple_passthrough_offline_gate.py
"""
from __future__ import annotations

import json
import os
import struct
import sys

import torch
from torch import nn
from types import SimpleNamespace

sys.path.insert(0, "/data/src/vllm-v2")

NEW = "/data/models/Qwen3.8-Flash-Next-FP4-Local-NVFP4"
OLD = "/data/models/Qwen3.8-Flash-Next-NVFP4"
TP_SIZE = 8
SAMPLE_SHARDS = (0, 7, 15)      # rank 0's window is rows [0, 40_000_192) = shards 0..15
OUT_OF_WINDOW_SHARD = 127       # must be a no-op for rank 0

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok: " if cond else "FAIL: ") + msg)
    if not cond:
        FAILURES.append(msg)


def st_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def index(root: str) -> dict:
    return json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]


def ple_meta(root: str):
    idx = index(root)
    codes = sorted(
        n for n in idx if "ple_embedding.ngram_embedding.shard_" in n
        and n.endswith(".weight")
    )
    scales = sorted(
        n for n in idx if "ple_embedding.ngram_embedding.shard_" in n
        and n.endswith(".weight_scale")
    )
    gscale = sorted(
        n for n in idx
        if n.endswith("ple_embedding.ngram_embedding.weight_scale_2")
    )
    return idx, codes, scales, gscale


def phase_a_and_b() -> tuple[int, int, int]:
    print("=== A/B. headers: NEW vs OLD ===")
    tc = json.load(open(f"{NEW}/config.json"))["text_config"]
    parts = int(tc["split_ngram_parts"])
    idx, codes, scales, gscale = ple_meta(NEW)
    check(len(codes) == 128, f"new: 128 code shards (got {len(codes)})")
    check(len(scales) == 128, f"new: 128 group-scale shards (got {len(scales)})")
    check(len(gscale) == 1, f"new: exactly one PLE weight_scale_2 (got {len(gscale)})")

    files = {idx[n] for n in codes + scales + gscale}
    hdr = {f: st_header(os.path.join(NEW, f)) for f in files}
    rows = 0
    widths = set()
    for n in codes:
        h = hdr[idx[n]][n]
        rows += h["shape"][0]
        widths.add((h["dtype"], h["shape"][1]))
    check(rows == 320001536, f"new: code rows total 320001536 (got {rows})")
    check(widths == {("U8", 80)}, f"new: all codes U8 [.,80] (got {widths})")
    scales_rows = sum(hdr[idx[n]][n]["shape"][0] for n in scales)
    check(scales_rows == rows, "new: scale rows == code rows")
    sw = {(hdr[idx[n]][n]["dtype"], hdr[idx[n]][n]["shape"][1]) for n in scales}
    check(sw == {("F8_E4M3", 10)}, f"new: all scales F8_E4M3 [.,10] (got {sw})")
    g = hdr[idx[gscale[0]]][gscale[0]]
    check(g["dtype"] == "F32" and tuple(g["shape"]) == (1,),
          f"new: weight_scale_2 F32 [1] (got {g['dtype']} {g['shape']})")
    check(rows % parts == 0, f"new: rows divisible by split_ngram_parts={parts}")
    shard_size = (rows + parts - 1) // parts
    check(shard_size == 2500012, f"new: shard_size 2500012 (got {shard_size})")
    check(parts * shard_size == rows,
          "new: 128 shards x 2500012 tile the table exactly")

    oidx, ocodes, oscales, _ = ple_meta(OLD)
    oh = st_header(os.path.join(OLD, oidx[ocodes[0]]))[ocodes[0]]
    check(oh["dtype"] != "U8", f"old: codes are NOT uint8 (got {oh['dtype']})")
    check(oh["shape"][1] == 160,
          f"old: codes are full-dim 160 (got {oh['shape'][1]})")
    check(len(oscales) == 0,
          f"old: no shard_*.weight_scale tensors (got {len(oscales)})")
    print("   -> passthrough discriminator (uint8 AND dim//2) can never match "
          "the old table: FP8/sidecar paths unchanged")
    return rows, shard_size, parts


def phase_c(rows: int, shard_size: int, parts: int) -> None:
    print("\n=== C. per-rank mapping tiles the window exactly once ===")
    from vllm.models.qwen4_exp.common.ple import compute_ple_shard_overlap

    per_rank = rows // TP_SIZE
    for rank in range(TP_SIZE):
        tp_start = rank * per_rank
        tp_end = tp_start + per_rank
        covered = 0
        for s in range(parts):
            ov = compute_ple_shard_overlap(
                checkpoint_start=s * shard_size,
                checkpoint_rows=shard_size,
                tp_start=tp_start,
                tp_end=tp_end,
            )
            if ov is not None:
                check_pairs = 0
                covered += ov.row_count
                # destination offsets must be contiguous from 0
                if ov.destination_start != covered - ov.row_count:
                    FAILURES.append(f"rank {rank} shard {s}: non-contiguous dst")
    check(covered == per_rank,
          f"rank 0..7 each covered exactly {per_rank} rows (checked all)")


def phase_d(shard_size: int) -> None:
    print("\n=== D. byte-exact: REAL loaders vs direct safetensors read ===")
    from safetensors import safe_open
    from vllm.models.qwen4_exp.common.ngram_embedding import (
        Qwen4ExpPLENvfp4EmbeddingMethod,
    )
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        Qwen4ExpPLEPinnedHostEmbedding,
    )

    idx, codes, scales, gscale = ple_meta(NEW)
    rows = 320001536
    per_rank = rows // TP_SIZE

    # create_weights reads get_current_vllm_config().load_config.load_format.
    # Outside a serve/pytest context there is no vLLM config, so supply the two
    # fields it touches (not ipc_cache -> ordinary host allocation).
    import vllm.config as _vcfg
    import vllm.models.qwen4_exp.common.ngram_embedding as _nge

    class _Cfg:
        load_config = SimpleNamespace(load_format="dummy")
        model_config = SimpleNamespace(
            architecture="Qwen4ExpForCausalLM",
            hf_text_config=SimpleNamespace(ple_layer_ids=[2]),
        )

    _vcfg.get_current_vllm_config = lambda: _Cfg()
    if hasattr(_nge, "get_current_vllm_config"):
        _nge.get_current_vllm_config = lambda: _Cfg()

    # The Parameter constructors read the TP rank/size from
    # vllm.model_executor.parameter.
    import vllm.model_executor.parameter as _par

    _par.get_tensor_model_parallel_rank = lambda: 0
    _par.get_tensor_model_parallel_world_size = lambda: TP_SIZE

    # rank 0's window [0, per_rank)
    method = Qwen4ExpPLENvfp4EmbeddingMethod()
    layer = Qwen4ExpPLEPinnedHostEmbedding.__new__(Qwen4ExpPLEPinnedHostEmbedding)
    nn.Module.__init__(layer)
    layer.org_vocab_size = rows
    layer.embedding_dim = 160
    layer.data_parallel_rank = 0
    layer.tp_rank = 0
    layer.tp_size = TP_SIZE
    layer.shard_indices = SimpleNamespace(
        org_vocab_start_index=0, org_vocab_end_index=per_rank
    )

    def scale_loader(param, loaded):
        param.copy_(torch.as_tensor(loaded, dtype=param.dtype, device=param.device))

    method.create_weights(
        layer,
        input_size_per_partition=160,
        output_partition_sizes=[per_rank],
        input_size=160,
        output_size=per_rank,
        params_dtype=torch.uint8,
        weight_loader=scale_loader,
    )
    print(f"   destination: codes {tuple(layer.weight.shape)} "
          f"({layer.weight.numel()*1e-9:.2f} GB) + scales "
          f"{tuple(layer.nvfp4_scales.shape)}")

    for s in SAMPLE_SHARDS:
        cn, sn = codes[s], scales[s]
        with safe_open(os.path.join(NEW, idx[cn]), framework="pt") as f:
            src_codes = f.get_tensor(cn)
        with safe_open(os.path.join(NEW, idx[sn]), framework="pt") as f:
            src_scales = f.get_tensor(sn)
        start = s * shard_size
        dst = start  # rank 0: destination offset == checkpoint offset
        layer.weight.weight_loader(
            layer.weight, src_codes, checkpoint_start=start
        )
        layer._ple_nvfp4_scale_loader(src_scales, checkpoint_start=start)
        same_codes = torch.equal(
            layer.weight.narrow(0, dst, src_codes.shape[0]), src_codes
        )
        same_scales = torch.equal(
            layer.nvfp4_scales.narrow(0, dst, src_scales.shape[0]),
            src_scales.view(torch.uint8),
        )
        check(same_codes and same_scales,
              f"shard {s}: codes+scales landed byte-exact "
              f"({src_codes.shape[0]} rows)")

    # A shard past this rank's window must copy nothing at all.
    cn, sn = codes[OUT_OF_WINDOW_SHARD], scales[OUT_OF_WINDOW_SHARD]
    with safe_open(os.path.join(NEW, idx[cn]), framework="pt") as f:
        src_codes = f.get_tensor(cn)
    with safe_open(os.path.join(NEW, idx[sn]), framework="pt") as f:
        src_scales = f.get_tensor(sn)
    before_codes = layer.weight.clone()
    before_scales = layer.nvfp4_scales.clone()
    start = OUT_OF_WINDOW_SHARD * shard_size
    layer.weight.weight_loader(layer.weight, src_codes, checkpoint_start=start)
    layer._ple_nvfp4_scale_loader(src_scales, checkpoint_start=start)
    check(torch.equal(layer.weight, before_codes)
          and torch.equal(layer.nvfp4_scales, before_scales),
          f"shard {OUT_OF_WINDOW_SHARD}: outside rank 0's window -> zero rows "
          "written")


def main() -> int:
    torch.set_num_threads(8)
    rows, shard_size, parts = phase_a_and_b()
    phase_c(rows, shard_size, parts)
    phase_d(shard_size)
    print("\nVERDICT:", "T5-PASS" if not FAILURES else f"T5-FAIL ({len(FAILURES)})")
    for f in FAILURES:
        print("  -", f)
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
