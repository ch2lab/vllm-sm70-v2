# SPDX-License-Identifier: Apache-2.0
"""File-valve host-side attribution for the HC (GatedResidual) blocks.

Same mechanism as the decoder-layer instrument in model.py (armed by the
EXISTENCE of VLLM_SM70_OPATTR_FILE; unset => every helper is a no-op), but with
its own accumulator and 'hc:'-prefixed tags so both instruments can be armed
together and analysed from the same /tmp/sm70_opattr_<pid>.jsonl stream.
Motivation (2026-10-18): the in-service region attribution puts the HC at 20.4%
of the prefill wall (6.1 ms per layer call), of which the six basic ops account
for only 2.32 ms; the AR all-reduces live INSIDE the ReplicatedLinear modules
(3 per HC block, 288 per step), so the split needs marks in here.
"""
from __future__ import annotations

import os

import torch

_SM70_OPATTR_FILE = os.environ.get("VLLM_SM70_OPATTR_FILE",
                                   "/tmp/sm70_opattr_mode")
_hc = {"ev": [], "tags": []}


def _hc_armed() -> bool:
    if not _SM70_OPATTR_FILE:
        return False
    try:
        return os.path.exists(_SM70_OPATTR_FILE)
    except OSError:
        return False


def hc_mark(tag: str) -> None:
    if not _hc_armed() or torch.cuda.is_current_stream_capturing():
        return
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    _hc["tags"].append(tag)
    _hc["ev"].append(ev)


def hc_flush(block: str) -> None:
    if not _hc["ev"]:
        return
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    _hc["ev"].append(end)
    _hc["tags"].append("hc:end")
    torch.cuda.synchronize()
    evs, tags = _hc["ev"], _hc["tags"]
    pairs = [f"{tags[i]}:{evs[i].elapsed_time(evs[i + 1]):.3f}"
             for i in range(len(evs) - 1)]
    try:
        with open(f"/tmp/sm70_opattr_{os.getpid()}.jsonl", "a") as fh:
            fh.write(f"{os.getpid()},-1,{block}," + ";".join(pairs) + "\n")
    except OSError:
        pass
    _hc["ev"] = []
    _hc["tags"] = []
