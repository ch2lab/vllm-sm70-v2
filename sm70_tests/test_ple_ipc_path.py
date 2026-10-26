#!/usr/bin/env python3
"""Unit tests for PLE NVFP4 over the weight cache IPC path (load_format=ipc_cache),
the layerwise (RL) reload interaction, and the IPC + reload composition.

Covers:
(1) nvfp4_scales must be exported (non-persistent buffer: named_buffers yes,
    state_dict no) so the daemon ships it and the engine re-registers it;
(2) refresh_lookup_state must re-pin + rebuild the UVA views after the IPC
    loader replaces the storage objects out of band (mirroring _apply_entries),
    re-point the group-scale slot, restore the non-persistent flag, and be a
    no-op otherwise (including during a layerwise reload window and on
    non-CPU storage);
(3) the sidecar weight_scale consistency check in PWL;
(4) transient/non-CPU storage is left alone by refresh;
(5) FP8 (non-NVFP4) pinned layer: refresh no-op path;
(6a) disk-boot engine + layerwise reload in the PRODUCTION configuration
    (record_metadata_for_reloading before anything; restore_device left at the
    process default — CPU; no set_default_device exists in the tree): the
    views must stay on the original pinned storage;
(6b) IPC-booted engine + layerwise reload in the REAL engine ordering (record
    before the IPC swap; swapped Parameters carry no weight_loader attribute;
    PWL re-attaches loaders and re-points the slot).

Also: get_layer_size excludes the never-loaded group-scale buffer (the
weight_loader_numel = 0 contract), so the layerwise inline trigger fires on
the last streamed parameter and the payload is released.
"""

import json
import os
import pickle
import shutil
import tempfile

import torch
from safetensors.torch import save_file
from torch import nn

from vllm.model_executor.model_loader.reload.layerwise import (
    _layerwise_process,
    finalize_layerwise_processing,
    get_layerwise_info,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.utils import get_layer_size
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLENvFp4EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    _PleNvfp4Sidecar,
)

# Single-rank process: report rank/size directly to ModelWeightParameter
# (no real process group is spawned; only (6d)'s create_weights reads these).
import vllm.model_executor.parameter as _ple_param_mod

_ple_param_mod.get_tensor_model_parallel_rank = lambda: 0
_ple_param_mod.get_tensor_model_parallel_world_size = lambda: 1


# ---- self-contained tiny sidecar (for the PWL consistency tests) ------------
TINY_ROOT = tempfile.mkdtemp(prefix="ple_sidecar_test_")
TINY = os.path.join(TINY_ROOT, "tiny")
os.makedirs(TINY)
_t_rows, _t_dim, _t_shards = 16, 16, 2
for _k in range(_t_shards):
    save_file(
        {
            "weight_e2m1": torch.randint(0, 255, (8, _t_dim // 2), dtype=torch.uint8),
            "weight_scale": torch.randint(0, 255, (8, _t_dim // 16), dtype=torch.uint8).view(torch.float8_e4m3fn),
            "weight_scale_2": torch.tensor(1.5e-5, dtype=torch.float32),
        },
        os.path.join(TINY, f"shard_{_k}.safetensors"),
    )
with open(os.path.join(TINY, "META.json"), "w") as _f:
    json.dump(
        {"layout": "group16_e2m1_e4m3scale_lownibblefirst", "shards": _t_shards,
         "rows": _t_rows, "width": _t_dim},
        _f,
    )


# (1) registration semantics -------------------------------------------------


class _BufLayer(nn.Module):
    pass


bl = _BufLayer()
scales = torch.empty(4, 10, dtype=torch.uint8, pin_memory=True)
bl.register_buffer("nvfp4_scales", scales, persistent=False)
assert bl.nvfp4_scales is scales, "attribute access must resolve to the buffer"
assert dict(bl.named_buffers())["nvfp4_scales"] is scales, "daemon export sees it"
assert "nvfp4_scales" not in bl.state_dict(), "must not enter state_dict"
print("ok: nvfp4_scales exported via named_buffers, absent from state_dict")

# (1b) get_layer_size excludes the never-loaded group-scale buffer -----------
sz = nn.Module()
sz.register_parameter(
    "weight", nn.Parameter(torch.empty(4, 80, dtype=torch.uint8), requires_grad=False)
)
_sz_scales = torch.empty(4, 10, dtype=torch.uint8)
_sz_scales.weight_loader_numel = 0  # what create_weights now sets
sz.register_buffer("nvfp4_scales", _sz_scales, persistent=False)
assert get_layer_size(sz) == 320, "side-effect buffer excluded from the loadable size"
print("ok: get_layer_size excludes the group-scale buffer (inline trigger intact)")


# (2) refresh_lookup_state ----------------------------------------------------
def make_min_layer():
    """Minimal real nn.Module with PLE pinned storage, mirroring create_weights
    (weight/weight_scale as Parameters, nvfp4_scales as non-persistent buffer)."""
    l = object.__new__(Qwen4ExpPLEPinnedHostEmbedding)
    nn.Module.__init__(l)
    l.embedding_dim = 160
    w = torch.zeros(4, 80, dtype=torch.uint8, pin_memory=True)
    w[0, 0] = 0xAB
    s = torch.zeros(4, 10, dtype=torch.uint8, pin_memory=True)
    s[0, 0] = 0xCD
    l.register_parameter("weight", nn.Parameter(w, requires_grad=False))
    l.register_parameter(
        "weight_scale", nn.Parameter(torch.tensor([0.5]), requires_grad=False)
    )
    l.register_buffer("nvfp4_scales", s, persistent=False)
    l._build_lookup_state()
    return l, w, s


l, w0, s0 = make_min_layer()
assert l._lookup_state_key == l._current_lookup_state_key()

# Simulate the IPC replacement exactly as ipc_loader._register does: pop the
# old object, register a fresh non-pinned by-value copy (buffer: default
# persistence, i.e. persistent — the daemon's flag is lost).
w1 = torch.empty(4, 80, dtype=torch.uint8)
w1.copy_(w0)
s1 = torch.empty(4, 10, dtype=torch.uint8)
s1.copy_(s0)
old_uva_w = l._uva_weight
old_uva_s = l._lookup_scales
l._parameters.pop("weight", None)
l.register_parameter("weight", nn.Parameter(w1, requires_grad=False))
l._buffers.pop("nvfp4_scales", None)
l.register_buffer("nvfp4_scales", s1)
assert "nvfp4_scales" not in l._non_persistent_buffers_set, "IPC lost the flag"
assert l._current_lookup_state_key() != l._lookup_state_key, "swap must be detected"

l.refresh_lookup_state()
assert l.weight.is_pinned(), "weight must be re-pinned"
assert l.nvfp4_scales.is_pinned(), "scales must be re-pinned"
assert l._uva_weight is not old_uva_w, "weight view must be rebuilt"
assert l._lookup_scales is not old_uva_s, "scales view must be rebuilt"
assert l.weight.numpy().tobytes() == w0.numpy().tobytes(), "contents preserved"
assert l.nvfp4_scales.numpy().tobytes() == s0.numpy().tobytes(), "contents preserved"
assert "nvfp4_scales" in l._non_persistent_buffers_set, (
    "non-persistent flag must be restored for reload/layerwise"
)
assert "nvfp4_scales" not in l.state_dict(), "must not enter state_dict"
print("ok: refresh re-pins, rebuilds UVA views, preserves contents, "
      "restores non-persistent flag")

# Idempotent: second call is a no-op.
uva_after = l._uva_weight
l.refresh_lookup_state()
assert l._uva_weight is uva_after, "second refresh must not rebuild"
print("ok: refresh is a no-op when storage objects are unchanged")

# No-op from the start (daemon side: nothing was replaced).
l2, _, _ = make_min_layer()
uva = l2._uva_weight
l2.refresh_lookup_state()
assert l2._uva_weight is uva
print("ok: refresh no-op for untouched storage")

# (2b) precondition: the IPC by-value transport really does arrive non-pinned
p = torch.arange(8, dtype=torch.uint8).reshape(2, 4).pin_memory()
p_rt = pickle.loads(pickle.dumps(p))
assert not p_rt.is_pinned() and p_rt.numpy().tobytes() == p.numpy().tobytes()
print("ok: by-value (pickle) transport arrives non-pinned, contents intact "
      "(justifies the re-pin in refresh)")

# (3) sidecar scale consistency in PWL ----------------------------------------
sidecar = _PleNvfp4Sidecar(TINY, total_rows=16, dim=16)
method = Qwen4ExpPLENvFp4EmbeddingMethod()
method._ple_sidecar = sidecar


class _PleLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 80, dtype=torch.uint8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.tensor([scale]), requires_grad=False)

    def refresh_lookup_state(self):
        pass


bad = _PleLayer(1.99e-4)  # e.g. daemon ran without the sidecar
try:
    method.process_weights_after_loading(bad)
    raise AssertionError("expected sidecar mismatch error")
except ValueError as e:
    assert "sidecar mismatch" in str(e), str(e)
print("ok: PWL raises on daemon/engine sidecar mismatch")

good = _PleLayer(sidecar.global_scale)
method.process_weights_after_loading(good)
assert float(good.weight_scale) == sidecar.global_scale
method.process_weights_after_loading(good)  # idempotent
assert float(good.weight_scale) == sidecar.global_scale
print("ok: PWL passes on matching sidecar scale (idempotent)")

# Plain sidecar boot (route A): this process's table came from the sidecar,
# so the checkpoint's FP8 scale (loaded first, different value) must be
# REPLACED by the sidecar's — not verified equal to it.
method._ple_sidecar_loaded = True
adopted = _PleLayer(1.99e-4)  # the checkpoint FP8 scale
method.process_weights_after_loading(adopted)
assert float(adopted.weight_scale) == sidecar.global_scale, "sidecar scale adopted"
method._ple_sidecar_loaded = False
print("ok: PWL adopts the sidecar scale when the table came from the sidecar")

# No sidecar here (daemon-on/engine-off): keep the provider's scale, no error.
method._ple_sidecar = None
free = _PleLayer(2.0345052689663135e-05)
method.process_weights_after_loading(free)
assert float(free.weight_scale) == 2.0345052689663135e-05
print("ok: no-sidecar process keeps the provider's scale (daemon-on/engine-off)")


# (4) transient / non-CPU storage must be left alone by refresh --------------
def swap_to(layer, w, s):
    layer._parameters.pop("weight", None)
    layer.register_parameter("weight", nn.Parameter(w, requires_grad=False))
    if s is not None:
        layer._buffers.pop("nvfp4_scales", None)
        layer.register_buffer("nvfp4_scales", s)


l3, _, _ = make_min_layer()
uva_before = l3._uva_weight
w_cuda = torch.zeros(4, 80, dtype=torch.uint8, device="cuda")
w_cuda[0, 0] = 0xAB
s_cuda = torch.zeros(4, 10, dtype=torch.uint8, device="cuda")
s_cuda[0, 0] = 0xCD
swap_to(l3, w_cuda, s_cuda)
l3.refresh_lookup_state()  # must not raise
assert l3._uva_weight is uva_before, "transient CUDA storage must not rebuild"
method_nosidecar = Qwen4ExpPLENvFp4EmbeddingMethod()
method_nosidecar.process_weights_after_loading(l3)  # full PWL path, no raise
assert l3._uva_weight is uva_before
print("ok: layerwise transient CUDA storage ignored (no hard-fail, no rebuild)")

# Mixed skew (version-skew daemon): CPU weight + non-CPU scales -> also left
# alone (the scales guard, not just the weight guard, must cover it).
l3b, _, _ = make_min_layer()
s2 = torch.zeros(4, 10, dtype=torch.uint8, device="cuda")
l3b._buffers.pop("nvfp4_scales", None)
l3b.register_buffer("nvfp4_scales", s2)
uva_b = l3b._uva_weight
l3b.refresh_lookup_state()  # must not raise
assert l3b._uva_weight is uva_b, "mixed device skew must not rebuild"
print("ok: mixed CPU-weight/non-CPU-scales skew ignored")

# (5) FP8 pinned layer (no nvfp4_scales): refresh no-op + LUT lookup state
l4 = object.__new__(Qwen4ExpPLEPinnedHostEmbedding)
nn.Module.__init__(l4)
l4.embedding_dim = 160
l4.register_parameter(
    "weight_scale", nn.Parameter(torch.tensor([0.5]), requires_grad=False)
)
w8 = torch.zeros(4, 160, dtype=torch.float8_e4m3fn, pin_memory=True)
l4.register_parameter("weight", nn.Parameter(w8, requires_grad=False))
l4._build_lookup_state()
assert l4._nvfp4_scales is None
assert l4._lookup_weight.dtype == torch.uint8, "sm70 FP8 path uses the LUT view"
uva4 = l4._uva_weight
l4.refresh_lookup_state()
assert l4._uva_weight is uva4, "FP8 no-op refresh must not rebuild"
print("ok: FP8 pinned refresh is a no-op (LUT lookup state intact)")


def _make_loader(layer, slot, codes, scale):
    """Mirrors create_weights: method-retained loaders + the group-scale slot."""
    def ple_loader(param, loaded):
        param.copy_(loaded["codes"])
        slot[0].copy_(loaded["scales"])  # live destination via the slot

    def scale_loader(param, loaded):
        param.copy_(torch.as_tensor(loaded, dtype=param.dtype, device=param.device))

    m = Qwen4ExpPLENvFp4EmbeddingMethod()
    m._ple_codes_loader = ple_loader
    m._ple_scale_loader = scale_loader
    m._ple_scale_slot = slot
    layer.quant_method = m
    return m, ple_loader, scale_loader


# (6a) disk-boot engine + layerwise reload — the PRODUCTION configuration:
# record_metadata_for_reloading runs in initialize_model, which the disk
# loader calls INSIDE its `with target_device:` context (base_loader.py).
# On torch >= 2.8 get_default_device() reflects that context, so the RECORDED
# restore device is the GPU — the materialize device must be re-derived from
# the layer's own storage (pinned CPU), not the recorded context device.
ld, _, _ = make_min_layer()
# Production disk boot: the 1-element per-tensor scale lives on the
# accelerator (allocated under the loader's target-device context) while the
# table is pinned CPU — a MIXED-device layer, so the materialize device must
# come from the BULK tensor (the uniform-device rule would keep the recorded
# cuda here and re-break production).
ld.register_parameter(
    "weight_scale",
    nn.Parameter(
        torch.zeros(1, dtype=torch.float32, device="cuda"),
        requires_grad=False,
    ),
)
uva0 = ld._uva_weight
weight_obj0 = ld.weight
scales_obj0 = ld.nvfp4_scales
codes_a = torch.randint(0, 255, (4, 80), dtype=torch.uint8)
scales_a = torch.randint(0, 255, (4, 10), dtype=torch.uint8)
slot_a = [scales_obj0]
ld_method, ple_loader_a, scale_loader_a = _make_loader(ld, slot_a, codes_a, scales_a)
ld.weight.weight_loader = ple_loader_a  # what create_weights attached
ld.weight_scale.weight_loader = scale_loader_a
with torch.device("cuda"):  # the disk loader's target-device context
    record_metadata_for_reloading(ld)
info_a = get_layerwise_info(ld)
assert info_a.restore_device.type == "cuda", (
    "torch >= 2.8: the recorded device follows the ambient context"
)
initialize_layerwise_reload(ld)
assert info_a.restore_device == torch.device("cpu"), (
    "initialize re-derives the materialize device from the BULK kernel "
    "tensor (the pinned CPU table beats the 1-element cuda scale — the "
    "recorded cuda would bounce the whole table through HBM)"
)
assert ld.weight.is_meta and ld.nvfp4_scales.is_meta
ld.weight.weight_loader(ld.weight, {"codes": codes_a, "scales": scales_a})
ld.weight_scale.weight_loader(ld.weight_scale, 0.75)
_layerwise_process(ld, get_layerwise_info(ld))
assert ld.weight is weight_obj0, "original Parameter restored"
assert ld.nvfp4_scales is scales_obj0, "original buffer restored"
assert weight_obj0.numpy().tobytes() == codes_a.numpy().tobytes(), (
    "new codes round-tripped in place into the pinned storage"
)
assert scales_obj0.numpy().tobytes() == scales_a.numpy().tobytes(), (
    "group scales landed in the live buffer via the slot"
)
assert float(ld.weight_scale) == 0.75
assert ld._uva_weight is uva0, (
    "views untouched during reload (no transient rebuild — the production "
    "CPU-materialize regression)"
)
ld.refresh_lookup_state()  # key realigned -> no-op
assert ld._uva_weight is uva0
print("ok: disk-boot engine survives a layerwise RL reload "
      "(production CPU materialize, views valid, values in place)")

# (6b) IPC-booted engine + layerwise reload — the REAL engine ordering: record
# metadata BEFORE the IPC swap. The IPC engine's initialize_model runs under
# `with torch.device("meta")` (ipc_loader.py), so on torch >= 2.8 the RECORDED
# restore device is meta (materializing on meta is a no-op that breaks the
# post-load checks) — the device must be re-derived at reload init time. The
# swapped Parameters carry no weight_loader attribute (ipc_loader._register
# re-wraps by-value tensors); PWL re-attaches the method-retained loaders and
# re-points the slot.
lc, _, _ = make_min_layer()
# On an IPC engine the per-tensor scale arrives as a CUDA (IPC) tensor —
# the layer is mixed-device (pinned CPU table + cuda scale); the materialize
# device must come from the BULK tensor.
lc.register_parameter(
    "weight_scale",
    nn.Parameter(
        torch.zeros(1, dtype=torch.float32, device="cuda"),
        requires_grad=False,
    ),
)
codes_b = torch.randint(0, 255, (4, 80), dtype=torch.uint8)
scales_b = torch.randint(0, 255, (4, 10), dtype=torch.uint8)
slot_b = [lc.nvfp4_scales]  # what create_weights captured, pre-swap
lc_method, ple_loader_b, scale_loader_b = _make_loader(lc, slot_b, codes_b, scales_b)
lc.weight.weight_loader = ple_loader_b  # create_weights attached it pre-swap
lc.weight_scale.weight_loader = scale_loader_b
# create_weights also declares the reload accounting on the (meta) tensors —
# the record below must capture it, exactly as on a real IPC engine:
lc.weight.weight_loader_numel = lc.weight.numel() + lc.nvfp4_scales.numel()
lc.nvfp4_scales.weight_loader_numel = 0
# Record metadata AFTER the loaders are attached (initialize_model order:
# create_weights runs first) — the meta copies must retain them. The IPC
# engine records under its meta-device context:
with torch.device("meta"):
    record_metadata_for_reloading(lc)
assert get_layerwise_info(lc).restore_device.type == "meta", (
    "torch >= 2.8: the recorded device follows the ambient meta context"
)

# IPC swap: fresh by-value objects with NO loader attributes (real _register
# state), then the engine PWL after the apply.
w1b = torch.empty(4, 80, dtype=torch.uint8)
w1b.copy_(lc.weight)
s1b = torch.empty(4, 10, dtype=torch.uint8)
s1b.copy_(lc.nvfp4_scales)
swap_to(lc, w1b, s1b)
assert not hasattr(lc.weight, "weight_loader"), "IPC state: loader attribute gone"
assert slot_b[0] is not lc.nvfp4_scales, "slot still points at the dead original"
lc_method.process_weights_after_loading(lc)
assert lc.weight.weight_loader is ple_loader_b, "PWL re-attached the codes loader"
assert lc.weight_scale.weight_loader is scale_loader_b, "PWL re-attached the scale loader"
assert slot_b[0] is lc.nvfp4_scales, "slot re-pointed at the live buffer"
weight_obj_b = lc.weight
scales_obj_b = lc.nvfp4_scales
uva_b2 = lc._uva_weight

# The PWL re-attached the loaders AND re-stamped the reload accounting on
# the swapped tensors (create_weights' declarations were stripped by the IPC
# swap) — without this the engine's per-layer total no longer matches what
# the loaders actually count and the inline trigger never fires (no RL flow
# calls finalize_layerwise_processing, so the PLE layer would never process):
assert (
    lc.weight.weight_loader_numel
    == lc.weight.numel() + lc.nvfp4_scales.numel()
), "PWL re-stamped the weight's loadable size (codes + group scales)"
assert lc.nvfp4_scales.weight_loader_numel == 0, (
    "PWL re-stamped the group-scale buffer (never loaded via its own loader)"
)

# RL round: the materialize device is re-derived from the layer's storage
# (the recorded meta must NOT win — meta materialization is a no-op).
initialize_layerwise_reload(lc)
assert get_layerwise_info(lc).restore_device == torch.device("cpu"), (
    "initialize re-derives the materialize device from the BULK kernel "
    "tensor (the pinned CPU table beats the 1-element cuda scale — the "
    "recorded meta is discarded)"
)
assert lc.weight.is_meta and lc.nvfp4_scales.is_meta
lc.weight.weight_loader(lc.weight, {"codes": codes_b, "scales": scales_b})
assert lc.weight.is_meta, "trigger must not fire before the final load"
lc.weight_scale.weight_loader(lc.weight_scale, 0.75)
assert lc.weight is weight_obj_b, (
    "inline trigger on the final load restored the original (post-swap) "
    "Parameter"
)
assert lc.nvfp4_scales is scales_obj_b, "original (post-swap) buffer restored"
assert weight_obj_b.numpy().tobytes() == codes_b.numpy().tobytes(), (
    "new codes round-tripped in place into the pinned storage"
)
assert scales_obj_b.numpy().tobytes() == scales_b.numpy().tobytes(), (
    "group scales landed in the LIVE buffer (re-pointed slot), not a transient"
)
assert float(lc.weight_scale) == 0.75
assert lc._uva_weight is uva_b2, "views untouched during reload"
assert "nvfp4_scales" in lc._non_persistent_buffers_set, "persistence preserved"
lc.refresh_lookup_state()  # key realigned -> no-op
assert lc._uva_weight is uva_b2
print("ok: IPC-booted engine survives a layerwise RL reload "
      "(real ordering, loaders re-attached, live scales, views valid)")

# (6c) multi-shard reload accounting: the inline trigger must fire only after
# the LAST streamed shard. The loader's CopyCounter count includes the
# group-scale side-effect copy, so create_weights declares
# weight_loader_numel = codes + scales on the weight param; with K >= 9
# shards an under-declared target fires the trigger early and the tail
# shards are dropped as "Excessive loading" (silent table-tail loss).
rows, dim, packed = 10, 160, 80
lz = object.__new__(Qwen4ExpPLEPinnedHostEmbedding)
nn.Module.__init__(lz)
lz.embedding_dim = dim
wz = torch.zeros(rows, packed, dtype=torch.uint8, pin_memory=True)
sz = torch.zeros(rows, dim // 16, dtype=torch.uint8, pin_memory=True)
lz.register_parameter("weight", nn.Parameter(wz, requires_grad=False))
lz.register_parameter(
    "weight_scale", nn.Parameter(torch.tensor([0.5]), requires_grad=False)
)
lz.register_buffer("nvfp4_scales", sz, persistent=False)
sz.weight_loader_numel = 0  # what create_weights sets (never loaded via its own loader)
lz._build_lookup_state()
slot_z = [sz]
z_method = Qwen4ExpPLENvFp4EmbeddingMethod()


def ple_loader_z(param, loaded, dst):
    # like the real nvfp4_loader: copy into the row slice, not the whole param
    param.narrow(0, dst, 1).copy_(loaded["codes"])
    slot_z[0].narrow(0, dst, 1).copy_(loaded["scales"])


def scale_loader_z(param, loaded):
    param.copy_(torch.as_tensor(loaded, dtype=param.dtype, device=param.device))


lz.weight.weight_loader = ple_loader_z
lz.weight_scale.weight_loader = scale_loader_z
# what create_weights now declares (codes + group scales):
lz.weight.weight_loader_numel = wz.numel() + sz.numel()
z_method._ple_codes_loader = ple_loader_z
z_method._ple_scale_loader = scale_loader_z
z_method._ple_scale_slot = slot_z
lz.quant_method = z_method
uva_z = lz._uva_weight
record_metadata_for_reloading(lz)
initialize_layerwise_reload(lz)
# Mid-reload refresh: with the reload window open (kernel_tensors snapshot
# live, storage meta) the refresh must be a no-op — no re-pin, no view
# rebuild, no hard-fail on the meta transients.
uva_z_mid = lz._uva_weight
lz.refresh_lookup_state()
assert lz._uva_weight is uva_z_mid, (
    "refresh inside the reload window must not rebuild the views"
)
for r in range(1, rows + 1):
    lz.weight.weight_loader(
        lz.weight,
        {
            "codes": torch.full((1, packed), r, dtype=torch.uint8),
            "scales": torch.full((1, dim // 16), r, dtype=torch.uint8),
        },
        r - 1,
    )
    if r < rows:
        assert lz.weight.is_meta, "trigger must not fire before the last shard"
lz.weight_scale.weight_loader(lz.weight_scale, 0.75)
assert wz.numpy().tolist() == [[r] * packed for r in range(1, rows + 1)], (
    "all 10 shards landed in the pinned storage (incl. the last one)"
)
assert sz.numpy().tolist() == [[r] * (dim // 16) for r in range(1, rows + 1)]
assert float(lz.weight_scale) == 0.75
assert lz._uva_weight is uva_z, "views untouched during reload"
print("ok: 10-shard stream triggers only on the final load (no tail loss)")

# (6d) env-on RL reload through the REAL create_weights + nvfp4_loader
# (sidecar branch): the sidecar is the canonical frozen table — the incoming
# RL weight stream (checkpoint-style) must be ignored, the sidecar re-read
# (idempotent), and the sidecar scale re-adopted (an incoming FP8 scale is
# NOT applied).
import vllm.config as _vllm_config_mod
from types import SimpleNamespace

_orig_gcv = _vllm_config_mod.get_current_vllm_config
try:
    class _FakeVllmConfig:
        # unknown architecture -> engram_layer_ids returns None -> guard passes
        model_config = SimpleNamespace(architecture="FakePLEArch")
        # create_weights reads load_config.load_format to pick the meta/shared
        # PLE storage; without it the harness died at the first create_weights
        # (pre-existing on HEAD, independent of sm70/08). A non-ipc format keeps
        # this case on the re-quantization path the test intends.
        load_config = SimpleNamespace(load_format="dummy")

    _vllm_config_mod.get_current_vllm_config = lambda: _FakeVllmConfig()

    def scale_loader_d(param, loaded):
        param.copy_(torch.as_tensor(loaded, dtype=param.dtype, device=param.device))

    # env off (default): create_weights stays on the re-quantization path
    # and never constructs a sidecar (inertness through the REAL code path).
    lr0 = object.__new__(Qwen4ExpPLEPinnedHostEmbedding)
    nn.Module.__init__(lr0)
    lr0.org_vocab_size = 16
    lr0.embedding_dim = 16
    lr0.shard_indices = SimpleNamespace(
        org_vocab_start_index=0, org_vocab_end_index=16
    )
    r_method0 = Qwen4ExpPLENvFp4EmbeddingMethod()
    r_method0.create_weights(
        lr0,
        input_size_per_partition=16,
        output_partition_sizes=[16],
        input_size=16,
        output_size=16,
        params_dtype=torch.uint8,
        weight_loader=scale_loader_d,
    )
    assert r_method0._ple_sidecar is None, "no sidecar when the env is unset"
    print("ok: create_weights stays on the requant path with the env unset")

    # (6d) env-on:
    os.environ["VLLM_PLE_QUANT_DIR"] = TINY

    lr = object.__new__(Qwen4ExpPLEPinnedHostEmbedding)
    nn.Module.__init__(lr)
    lr.org_vocab_size = 16
    lr.embedding_dim = 16
    lr.data_parallel_rank = 0
    # The VPE init (bypassed) stamps these for update_param_tp_status:
    lr.tp_rank = 0
    lr.tp_size = 1
    lr.shard_indices = SimpleNamespace(
        org_vocab_start_index=0, org_vocab_end_index=16
    )
    r_method = Qwen4ExpPLENvFp4EmbeddingMethod()
    r_method.create_weights(
        lr,
        input_size_per_partition=16,
        output_partition_sizes=[16],
        input_size=16,
        output_size=16,
        params_dtype=torch.uint8,
        weight_loader=scale_loader_d,
    )
    assert r_method._ple_sidecar is not None, "sidecar must be built from the env"
    assert lr.nvfp4_scales is not None and "nvfp4_scales" not in lr.state_dict()
    lr.quant_method = r_method  # what the VPE init does

    # Expected canonical rows straight from the sidecar:
    exp_codes = torch.empty(16, 8, dtype=torch.uint8)
    exp_scales = torch.empty(16, 1, dtype=torch.uint8)
    r_method._ple_sidecar.copy_global_rows(0, 16, exp_codes, exp_scales)

    # A checkpoint-style incoming payload (all-0xFF FP8-ish rows), distinct
    # from the sidecar's contents:
    fake_payload = torch.full((16, 16), 200, dtype=torch.float8_e4m3fn)

    # Boot-time load through the REAL loader (route-A sequence): the
    # sidecar branch must ignore the incoming payload and copy the sidecar's
    # rows; the checkpoint streams its own (FP8) global scale separately.
    lr.weight.weight_loader(lr.weight, fake_payload, 0)
    assert r_method._ple_sidecar_loaded is True
    assert lr.weight.numpy().tobytes() == exp_codes.numpy().tobytes(), (
        "boot load must come from the sidecar, not the incoming payload"
    )
    assert lr.nvfp4_scales.numpy().tobytes() == exp_scales.numpy().tobytes()
    lr.weight_scale.weight_loader(lr.weight_scale, 1.99e-4)  # checkpoint FP8 scale
    r_method.process_weights_after_loading(lr)
    assert float(lr.weight_scale) == r_method._ple_sidecar.global_scale, (
        "sidecar scale adopted over the checkpoint FP8 scale"
    )
    uva_r = lr._uva_weight
    assert uva_r is not None, "views built"
    weight_obj_r = lr.weight

    # RL reload through the REAL machinery: the incoming stream (and its
    # scale) must be ignored; the sidecar is re-read and the scale re-adopted.
    record_metadata_for_reloading(lr)
    initialize_layerwise_reload(lr)
    assert lr.weight.is_meta and lr.nvfp4_scales.is_meta
    lr.weight.weight_loader(lr.weight, fake_payload, 0)
    lr.weight_scale.weight_loader(lr.weight_scale, 0.99)  # incoming FP8 scale
    # inline trigger fired on the final load (see 6c's accounting)
    assert lr.weight is weight_obj_r, "original Parameter restored"
    assert lr.weight.numpy().tobytes() == exp_codes.numpy().tobytes(), (
        "table still the canonical sidecar (incoming stream ignored)"
    )
    assert lr.nvfp4_scales.numpy().tobytes() == exp_scales.numpy().tobytes()
    assert float(lr.weight_scale) == r_method._ple_sidecar.global_scale, (
        "sidecar scale re-adopted (incoming 0.99 not applied)"
    )
    assert lr._uva_weight is uva_r, "views untouched during reload"
    print("ok: env-on RL reload re-reads the sidecar (frozen-table contract, "
          "scale re-adopted, incoming stream ignored)")

    # (6e) the scale contract skips an EMPTY table: a rank that owns no rows
    # (degenerate TP range) cannot mispair a scale, so the verify must not
    # reject such a rank.
    empty = _PleLayer(1.99e-4)
    empty.weight = nn.Parameter(
        torch.empty(0, dtype=torch.uint8), requires_grad=False
    )
    r_method._ple_sidecar_loaded = False
    r_method.process_weights_after_loading(empty)
    print("ok: PWL scale contract skips an empty table (no false rejection)")

    # (6f) a model with MORE THAN ONE PLE layer is refused while the sidecar
    # env is set (one sidecar serves one table).
    class _FakeMultiConfig:
        model_config = SimpleNamespace(
            architecture="Qwen4ExpForCausalLM",
            hf_text_config=SimpleNamespace(ple_layer_ids=[3, 7]),
        )
        # see _FakeVllmConfig: create_weights needs load_config.load_format
        # (pre-existing harness gap fixed here, unrelated to sm70/08).
        load_config = SimpleNamespace(load_format="dummy")

    _vllm_config_mod.get_current_vllm_config = lambda: _FakeMultiConfig()
    lr_multi = object.__new__(Qwen4ExpPLEPinnedHostEmbedding)
    nn.Module.__init__(lr_multi)
    lr_multi.org_vocab_size = 16
    lr_multi.embedding_dim = 16
    lr_multi.shard_indices = SimpleNamespace(
        org_vocab_start_index=0, org_vocab_end_index=16
    )
    try:
        r_method.create_weights(
            lr_multi,
            input_size_per_partition=16,
            output_partition_sizes=[16],
            input_size=16,
            output_size=16,
            params_dtype=torch.uint8,
            weight_loader=scale_loader_d,
        )
        raise AssertionError("expected the multi-PLE-layer rejection")
    except ValueError as e:
        assert "PLE layers" in str(e), str(e)
    print("ok: multi-PLE-layer model refused with a single sidecar")

    # (6j) DELTA stream: the weight arrives but the 1-element weight_scale
    # is omitted (frozen-table sync semantics). The inline trigger's target
    # includes the scale's +1, so it never fires; the PLE layer is processed
    # only by the delayed finalize at end of update — the R7 safety net (the
    # never-arrived scale is fixed by the PWL sidecar adoption).
    _vllm_config_mod.get_current_vllm_config = lambda: _FakeVllmConfig()
    lr2 = object.__new__(Qwen4ExpPLEPinnedHostEmbedding)
    nn.Module.__init__(lr2)
    lr2.org_vocab_size = 16
    lr2.embedding_dim = 16
    lr2.data_parallel_rank = 0
    lr2.tp_rank = 0
    lr2.tp_size = 1
    lr2.shard_indices = SimpleNamespace(
        org_vocab_start_index=0, org_vocab_end_index=16
    )
    j_method = Qwen4ExpPLENvFp4EmbeddingMethod()
    j_method.create_weights(
        lr2,
        input_size_per_partition=16,
        output_partition_sizes=[16],
        input_size=16,
        output_size=16,
        params_dtype=torch.uint8,
        weight_loader=scale_loader_d,
    )
    weight_obj_j = lr2.weight
    # Boot (route-A sequence): sidecar load, checkpoint scale, PWL (adopts the
    # sidecar scale and builds the UVA views).
    lr2.quant_method = j_method
    boot_payload_j = torch.full((16, 16), 200, dtype=torch.float8_e4m3fn)
    lr2.weight.weight_loader(lr2.weight, boot_payload_j, 0)
    lr2.weight_scale.weight_loader(lr2.weight_scale, 1.99e-4)  # checkpoint FP8 scale
    j_method.process_weights_after_loading(lr2)
    assert float(lr2.weight_scale) == j_method._ple_sidecar.global_scale
    uva_j = lr2._uva_weight
    assert uva_j is not None, "views built"
    record_metadata_for_reloading(lr2)
    initialize_layerwise_reload(lr2)
    payload_j = torch.randint(
        0, 200, (16, 16), dtype=torch.uint8
    ).view(torch.float8_e4m3fn)
    lr2.weight.weight_loader(lr2.weight, payload_j, 0)
    assert lr2.weight.is_meta, (
        "delta stream (scale pending) must not trigger the inline load"
    )
    finalize_layerwise_processing(lr2, None)
    assert lr2.weight is weight_obj_j, "delayed finalize restored the original"
    assert lr2.weight.numpy().tobytes() == exp_codes.numpy().tobytes(), (
        "delayed path landed the canonical sidecar rows"
    )
    assert float(lr2.weight_scale) == j_method._ple_sidecar.global_scale, (
        "PWL adoption fixed the never-arrived scale in the delayed path"
    )
    assert lr2._uva_weight is uva_j, "views valid after the delayed finalize"
    print("ok: delta stream (weight only, no scale) is carried by the "
          "delayed finalize (R7 safety net)")
finally:
    os.environ.pop("VLLM_PLE_QUANT_DIR", None)
    _vllm_config_mod.get_current_vllm_config = _orig_gcv

# (6g)/(6h) below call create_weights / process_weights_after_loading, which
# read get_current_vllm_config().load_config.load_format and .model_config.
# The `finally:` above restored the REAL accessor, which asserts outside a
# set_current_vllm_config() context, so re-arm the fake for the rest of the
# file (pre-existing gap: this harness could not reach these sections on HEAD
# at all — see sm70/03_history.md).
_vllm_config_mod.get_current_vllm_config = lambda: _FakeVllmConfig()

# (6g) env-off RL reload: the re-quantization branch cannot run during a
# reload (the arrival probes the loader on meta tensors; the re-quantization
# device handling routes outputs through the param's meta device and a copy
# out of meta is not allowed). It must fail loudly with actionable guidance —
# the reload-safe path is the sidecar branch (env on).
lr_off = object.__new__(Qwen4ExpPLEPinnedHostEmbedding)
nn.Module.__init__(lr_off)
lr_off.org_vocab_size = 16
lr_off.embedding_dim = 16
lr_off.data_parallel_rank = 0
lr_off.tp_rank = 0
lr_off.tp_size = 1
lr_off.shard_indices = SimpleNamespace(
    org_vocab_start_index=0, org_vocab_end_index=16
)
g_method = Qwen4ExpPLENvFp4EmbeddingMethod()
g_method.create_weights(
    lr_off,
    input_size_per_partition=16,
    output_partition_sizes=[16],
    input_size=16,
    output_size=16,
    params_dtype=torch.uint8,
    weight_loader=scale_loader_d,
)
assert g_method._ple_sidecar is None, "env off: no sidecar constructed"
record_metadata_for_reloading(lr_off)
initialize_layerwise_reload(lr_off)
payload_off = torch.randint(0, 200, (16, 16), dtype=torch.uint8).view(
    torch.float8_e4m3fn
)
try:
    lr_off.weight.weight_loader(lr_off.weight, payload_off, 0)
    raise AssertionError("expected the re-quantization-under-reload guard")
except ValueError as e:
    assert "VLLM_PLE_QUANT_DIR" in str(e), str(e)
print("ok: env-off RL reload fails loudly with sidecar guidance "
      "(no silent breakage)")

# (6h) A rank that owns NO org-rows (zero-width TP range) still has a
# padded, NON-EMPTY allocation — the mismatch branch must skip it (the
# numel==0 guard alone would not catch this):
degen = _PleLayer(1.99e-4)  # disagrees with the sidecar's scale
degen.shard_indices = SimpleNamespace(
    org_vocab_start_index=16, org_vocab_end_index=16
)
h_method = Qwen4ExpPLENvFp4EmbeddingMethod()
h_method._ple_sidecar = _PleNvfp4Sidecar(TINY, total_rows=16, dim=16)
h_method._ple_sidecar_loaded = False
h_method.process_weights_after_loading(degen)  # must not raise
print("ok: zero-org-rows rank (padded, non-empty table) skips the scale "
      "contract")

# (6i) Scalar n-gram layer id normalization: a config written with a bare
# int (including 0 — a valid layer index) must be treated as a one-layer
# list by every predicate (raw truthiness treated 0 as "no layers").
from vllm.config.engram import engram_layer_ids, model_has_engram_layers

cfg0 = SimpleNamespace(
    architecture="Qwen4ExpForCausalLM",
    hf_text_config=SimpleNamespace(ple_layer_ids=0),
)
assert engram_layer_ids(cfg0) == [0]
assert model_has_engram_layers(cfg0)
cfg_none = SimpleNamespace(
    architecture="Qwen4ExpForCausalLM",
    hf_text_config=SimpleNamespace(),
)
assert engram_layer_ids(cfg_none) is None
assert not model_has_engram_layers(cfg_none)
cfg_multi = SimpleNamespace(
    architecture="Qwen4ExpForCausalLM",
    hf_text_config=SimpleNamespace(ple_layer_ids=[3, 7]),
)
assert engram_layer_ids(cfg_multi) == [3, 7]
for bad_id in (2.5, "3,7", False):
    try:
        engram_layer_ids(SimpleNamespace(
            architecture="Qwen4ExpForCausalLM",
            hf_text_config=SimpleNamespace(ple_layer_ids=bad_id),
        ))
        raise AssertionError(f"expected rejection of {bad_id!r}")
    except ValueError:
        pass
print("ok: engram layer-id predicates normalize scalar/list, reject "
      "non-int")

shutil.rmtree(TINY_ROOT, ignore_errors=True)
print("ALL IPC-PATH TESTS PASSED")
