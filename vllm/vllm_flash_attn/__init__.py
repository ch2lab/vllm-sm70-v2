# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.machinery
import os
import sys
import types

# In symlink mode (VLLM_FLASH_ATTN_SRC_DIR), cute/ is a symlink to the real
# source tree and its files use `flash_attn.cute.*` imports (not rewritten).
# Register a virtual `flash_attn` package so those imports resolve.
_cute_dir = os.path.join(os.path.dirname(__file__), "cute")
if os.path.islink(_cute_dir) and "flash_attn" not in sys.modules:
    _fa_mod = types.ModuleType("flash_attn")
    _fa_mod.__path__ = [os.path.dirname(os.path.realpath(_cute_dir))]
    _fa_mod.__package__ = "flash_attn"
    _fa_mod.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", None, is_package=True
    )
    _fa_mod.__spec__.submodule_search_locations = _fa_mod.__path__
    sys.modules["flash_attn"] = _fa_mod

from vllm.vllm_flash_attn.flash_attn_interface import (  # noqa: E402
    FA2_AVAILABLE,
    FA3_AVAILABLE,
    compile_flash_attn_varlen_func_from_specs,
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)

if not (FA2_AVAILABLE or FA3_AVAILABLE):
    # On SM70 (Volta V100) the FA2/FA3 extensions cannot be built (no FP8/TF32
    # tensor cores), so this import would otherwise hard-fail before any
    # attention backend is chosen. FlashInfer's SM70 paged path is used instead,
    # so degrade to a warning rather than raising. Other configs are unchanged:
    # where FlashAttention is expected but its extensions are missing, the
    # original ImportError is still raised.
    from vllm.platforms import current_platform  # lazy: avoid an import cycle

    if current_platform.is_device_capability_family(70):
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "vllm_flash_attn: FA2/FA3 extensions unavailable on SM70; "
            "FlashAttention backends are disabled. Use the FlashInfer backend."
        )
    else:
        raise ImportError(
            "vllm.vllm_flash_attn requires the CUDA flash attention extensions "
            "(_vllm_fa2_C or _vllm_fa3_C). On ROCm, use upstream flash_attn."
        )

__all__ = [
    "compile_flash_attn_varlen_func_from_specs",
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]
