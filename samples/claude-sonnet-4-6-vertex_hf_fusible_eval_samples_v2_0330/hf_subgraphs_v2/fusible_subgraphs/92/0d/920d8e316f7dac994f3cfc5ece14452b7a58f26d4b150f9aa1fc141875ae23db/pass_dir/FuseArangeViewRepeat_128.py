"""
Fuse:  arange(0,128,cuda) -> view(1,-1) -> repeat(2,1)
Into:  single Triton kernel that writes (2,128) int64 tensor
       where each row is [0, 1, ..., 127].
Targets: bfloat16 RECT_L graph (N=128).
"""
import torch
from torch import device
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern():
    tmp_0 = torch.arange(0, 128, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args():
    return ()


# ---------------------------------------------------------------------------
# Triton kernel
# Each program handles one row (pid=0 or 1).
# BLOCK_SIZE == 128, so tl.arange(0, BLOCK_SIZE) covers every column exactly.
# No mask needed.
# ---------------------------------------------------------------------------
@triton.jit
def _arange_repeat_128_kernel(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    N: tl.constexpr = 128
    cols = tl.arange(0, BLOCK_SIZE)          # int32 [0..127]
    values = cols.to(tl.int64)               # int64 [0..127]
    out_offsets = row * N + cols             # element offsets into (2,128)
    tl.store(out_ptr + out_offsets, values)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated so FX doesn't try to trace into it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _arange_repeat_128():
    N = 128
    out = torch.empty((2, N), dtype=torch.int64, device='cuda')
    # Grid: 2 programs (one per row)
    _arange_repeat_128_kernel[(2,)](out, BLOCK_SIZE=128)
    return out


# ---------------------------------------------------------------------------
# replacement_func — returns the callable (do NOT call it)
# ---------------------------------------------------------------------------
def replacement_func():
    return _arange_repeat_128