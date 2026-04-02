"""
FuseFullGraph_stream_overlap.py

Matches the ENTIRE forward computation:
    matmul = torch.matmul(in_2, in_3)
    tmp_3  = in_1.to(device(type='cuda'))
    tmp_4  = in_0.to(device(type='cuda'))
    return (tmp_4, tmp_3, matmul)

Replacement overlaps the GPU matmul with the CPU-blocking PCIe transfers
by launching the matmul on a secondary CUDA stream while the main thread
waits for the two host-to-device copies.  The final result is the same but
the wall-clock (and CUDA-event) time collapses to
    max(matmul_time, transfer_time)
instead of the sequential sum.
"""

import torch
from torch import device
from typing import Optional


# ---------------------------------------------------------------------------
# Pattern — mirrors the exact dataflow in model.py.
# NOTE: "device(type='cuda')" must be imported from torch (done above).
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    matmul = torch.matmul(in_2, in_3)
    tmp_3  = in_1.to(device(type='cuda'))
    tmp_4  = in_0.to(device(type='cuda'))
    return (tmp_4, tmp_3, matmul)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Module-level secondary CUDA stream (created once, reused every call).
# ---------------------------------------------------------------------------
_matmul_stream: Optional[torch.cuda.Stream] = None


def _get_matmul_stream() -> torch.cuda.Stream:
    global _matmul_stream
    if _matmul_stream is None:
        _matmul_stream = torch.cuda.Stream()
    return _matmul_stream


# ---------------------------------------------------------------------------
# Replacement kernel wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def stream_overlap_forward(in_0, in_1, in_2, in_3):
    """
    Runs matmul asynchronously on a secondary CUDA stream while the main
    thread (CPU) performs the blocking host-to-device copies for in_0/in_1.
    This overlaps PCIe DMA with GPU compute.
    """
    matmul_stream = _get_matmul_stream()

    # Launch matmul on secondary stream (returns immediately to CPU)
    with torch.cuda.stream(matmul_stream):
        matmul = torch.matmul(in_2, in_3)

    # Host-to-device copies on the default (current) stream.
    # These block the CPU but run concurrently with the GPU matmul above.
    tmp_4 = in_0.to(device(type='cuda'))
    tmp_3 = in_1.to(device(type='cuda'))

    # Ensure the default stream waits for the matmul stream to finish
    # before any downstream consumer uses `matmul`.
    torch.cuda.current_stream().wait_stream(matmul_stream)

    return (tmp_4, tmp_3, matmul)


# ---------------------------------------------------------------------------
# Replacement factory
# ---------------------------------------------------------------------------
def replacement_func():
    return stream_overlap_forward