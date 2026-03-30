"""
Fuse:  arange(0,1000,cuda) -> view(1,-1) -> repeat(2,1)
Into:  single Triton kernel that writes (2,1000) int64 tensor
       where each row is [0, 1, ..., 999].
Targets: float16 and float32 GAE graphs (N=1000).
"""
import torch
from torch import device
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern():
    tmp_0 = torch.arange(0, 1000, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args():
    return ()


# ---------------------------------------------------------------------------
# Triton kernel
# Each program handles one row (pid=0 or 1).
# BLOCK_SIZE=1024 >= 1000, so one block covers an entire row; mask trims the
# last 24 lanes that would exceed column 999.
# ---------------------------------------------------------------------------
@triton.jit
def _arange_repeat_1000_kernel(
    out_ptr,
    N,                          # runtime value (1000)
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)      # int32 [0..1023]
    mask = cols < N
    values = cols.to(tl.int64)           # int64 column indices
    out_offsets = row * N + cols         # element offsets into (2,1000)
    tl.store(out_ptr + out_offsets, values, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated so FX doesn't try to trace into it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _arange_repeat_1000():
    N = 1000
    BLOCK_SIZE = 1024          # smallest power-of-2 >= N
    out = torch.empty((2, N), dtype=torch.int64, device='cuda')
    # Grid: 2 programs (one per row); each block handles all 1000 columns
    _arange_repeat_1000_kernel[(2,)](out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# replacement_func — returns the callable (do NOT call it)
# ---------------------------------------------------------------------------
def replacement_func():
    return _arange_repeat_1000