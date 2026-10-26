"""Per-stage GPU memory ledger (SM70 work, opt-in via VLLM_SM70_MEM_LEDGER=1).

vLLM already computes the full breakdown in MemoryProfilingResult but only emits
it at debug level, and nothing tracks the *runtime* activation peak against the
reservation made at startup. That gap is what turns into a late
OutOfMemoryError during a long-context prefill: the profile run only exercises
max_num_batched_tokens worth of short sequences, so buffers that scale with
context length (sparse-attention/indexer workspaces) are never measured.

Every line is one log record with the same shape so a log can be parsed back
into a budget table. When the env var is unset nothing here runs.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_SM70_MEM_LEDGER", "0") == "1"
_GIB = 2**30
# High-water mark of torch's allocated bytes since the last reset, per process.
_state = {"high_water": 0, "reported": 0, "steps": 0}


def enabled() -> bool:
    return _ENABLED


def _gib(value: float) -> str:
    return f"{value / _GIB:.3f}"


def report(stage: str, **terms: float | str) -> None:
    """Log one ledger line: device totals plus any caller-supplied terms."""
    if not _ENABLED:
        return
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)
    parts = [
        f"[memledger] stage={stage}",
        f"ctx+libs={_gib(total - free - reserved)}",
        f"torch_alloc={_gib(allocated)}",
        f"torch_reserved={_gib(reserved)}",
        f"torch_peak={_gib(peak)}",
        f"reserved_minus_alloc={_gib(reserved - allocated)}",
        f"device_free={_gib(free)}",
        f"device_total={_gib(total)}",
    ]
    for key, value in terms.items():
        if isinstance(value, str):
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={_gib(value)}")
    logger.info(" ".join(parts))


def reset_peak() -> None:
    """Forget the recorded peak so the next window measures only itself."""
    if not _ENABLED:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    _state["high_water"] = 0


def step_peak(num_scheduled_tokens: int = 0, every: int = 64) -> None:
    """Track the running activation peak during serving.

    Cheap: one counter read per step, and a log line only when a new
    high-water mark is set (or every `every` steps as a heartbeat).
    `num_scheduled_tokens` is recorded because the dominant transient term is
    the logits buffer, which scales with tokens-per-step times the vocab.
    """
    if not _ENABLED:
        return
    if not torch.cuda.is_available():
        return
    _state["steps"] += 1
    device = torch.cuda.current_device()
    peak = torch.cuda.max_memory_allocated(device)
    # Reserved, not allocated, is what actually presses against the device
    # limit: the allocator grabs segments ahead of use and long prefills fragment
    # them. A 260K prefill here grew device usage by 4.72 GiB while allocated
    # only rose 1.20 GiB.
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if peak <= _state["high_water"] and _state["steps"] % every:
        return
    grew = peak > _state["high_water"]
    _state["high_water"] = peak
    _state["reported"] += 1
    report(
        f"step{_state['steps']}" + ("_new_peak" if grew else "_heartbeat"),
        tokens_this_step=str(num_scheduled_tokens),
        step_peak_alloc=peak,
        step_peak_reserved=peak_reserved,
    )
