"""
Pass: FuseReduceSumDiv_dim3_keepdim
Fuses: x.sum(dim=3, keepdim=True) followed by x / sum
into a single Triton kernel.

Input:  in_3, shape [1, 2, 8, 8], dtype float16/bfloat16
Output: shape [1, 2, 8, 8] — each element divided by its row-sum along dim=3

Fixed shapes: n_rows = 1*2*8 = 16, n_cols = 8.
One Triton program per row; no autotune (overhead dominates at this scale).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  (mirror model.py exactly — keyword args for sum)
# ---------------------------------------------------------------------------
def pattern(x):
    s = x.sum(dim=3, keepdim=True)
    out = x / s
    return out


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------
def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel
#   Grid: (N_ROWS,) = (16,) — one program per row (leading dims flattened).
#   N_COLS = 8 elements per row, BLOCK_SIZE = 8 (power-of-2 == n_cols).
#   No masking needed: BLOCK_SIZE exactly matches n_cols.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_sum_div_kernel(
    x_ptr,
    out_ptr,
    stride_row,          # = n_cols = 8
    BLOCK_SIZE: tl.constexpr,   # == 8
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_start = row_id * stride_row

    x = tl.load(x_ptr + row_start + offsets)
    x_f32 = x.to(tl.float32)
    row_sum = tl.sum(x_f32, axis=0)
    out = x_f32 / row_sum
    tl.store(out_ptr + row_start + offsets, out)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fuse_reduce_sum_div_dim3(x):
    """
    x: shape [1, 2, 8, 8]
    returns: x / x.sum(dim=3, keepdim=True), shape [1, 2, 8, 8]
    """
    # Shapes are fixed: n_rows=16, n_cols=8
    N_COLS = x.shape[-1]          # 8
    N_ROWS = x.numel() // N_COLS  # 16
    out = torch.empty_like(x)
    reduce_sum_div_kernel[(N_ROWS,)](
        x, out, N_COLS, BLOCK_SIZE=8,
    )
    return out


# ---------------------------------------------------------------------------
# Replacement entry point
# ---------------------------------------------------------------------------
def replacement_func():
    return fuse_reduce_sum_div_dim3