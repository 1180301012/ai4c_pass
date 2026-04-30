"""
Shared Triton kernels for the AI4C cat + layer_norm optimization passes.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Layer Norm kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['N', 'BLOCK_SIZE'],
)
@triton.jit
def _layer_norm_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    eps,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row. Each program normalises one row of length N.
    Computes in fp32 for numerical stability, stores in original dtype.
    """
    pid = tl.program_id(0)
    x_ptr = x_ptr + pid * stride
    out_ptr = out_ptr + pid * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load row (out-of-range lanes get 0.0)
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Mean
    mean = tl.sum(x, axis=0) / N

    # Variance  (masked elements are 0 so contribute 0 to sum)
    x_c = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_c * x_c, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Scale & shift
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    out = x_c * rstd * w + b

    tl.store(out_ptr + cols, out.to(x_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Cat along dim=2 kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_ROWS': 32},  num_warps=2),
        triton.Config({'BLOCK_ROWS': 64},  num_warps=4),
        triton.Config({'BLOCK_ROWS': 128}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 256}, num_warps=8),
    ],
    key=['S_total', 'S_cur', 'S2', 'S3'],
)
@triton.jit
def _cat_dim2_kernel(
    in2_ptr, in3_ptr, in5_ptr,
    out_ptr,
    B, C,
    S_total, S_cur,   # S_cur = seq-len of first input (in_2 / a)
    S2, S23, S3,      # S2  = seq-len of second  (in_5 / b)
                       # S23 = S_cur+S2 — start of third  (in_3 / c)
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """
    Concatenate three tensors along dim=2:
      in_2: [B, 1, S_cur, C]
      in_5: [B, 1, S2,    C]
      in_3: [B, 1, S3,    C]
    → out: [B, 1, S_total, C]   where S_total = S_cur+S2+S3

    Grid: (B * ceil(S_total / BLOCK_ROWS),)
    """
    pid      = tl.program_id(0)
    n_blocks = tl.cdiv(S_total, BLOCK_ROWS)

    batch   = pid // n_blocks
    block_r = pid  % n_blocks

    row_start = block_r * BLOCK_ROWS
    rows      = row_start + tl.arange(0, BLOCK_ROWS)   # [BLOCK_ROWS]

    s_out = rows // S_cur    # which section (0=first, 1=second, 2=third)
    s_rem = rows % S_cur     # local row within that section

    out_off = (batch * S_total + rows) * C   # base offset in output
    batch2_off = batch * S_cur * C           # base offset in first input
    batch5_off = batch * S2    * C           # base offset in second input
    batch3_off = batch * S3    * C           # base offset in third input
    cols = tl.arange(0, BLOCK_COLS)         # [BLOCK_COLS]

    # ---- section 0: in_2  [B, 1, S_cur, C] ----
    valid2 = (rows < S_cur) & (cols < C)
    x2 = tl.load(
        in2_ptr + batch2_off + s_rem * C + cols,
        mask=valid2, other=0.0
    )
    tl.store(out_ptr + out_off + cols, x2, mask=valid2)

    # ---- section 1: in_5  [B, 1, S2, C] ----
    s5     = rows - S_cur
    valid5 = (s5 >= 0) & (rows < S_cur + S2) & (cols < C)
    x5 = tl.load(
        in5_ptr + batch5_off + s5 * C + cols,
        mask=valid5, other=0.0
    )
    tl.store(out_ptr + out_off + cols, x5, mask=valid5)

    # ---- section 2: in_3  [B, 1, S3, C] ----
    s3     = rows - S_cur - S2
    valid3 = (s3 >= 0) & (cols < C)
    x3 = tl.load(
        in3_ptr + batch3_off + s3 * C + cols,
        mask=valid3, other=0.0
    )
    tl.store(out_ptr + out_off + cols, x3, mask=valid3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_power_of_2(n):
    """Smallest power of 2 that is >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p