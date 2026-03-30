"""
Pass: FuseLayerNormDropout_unified

Fuses: layer_norm(any normalized_shape) + dropout(p=0.0)

into a single Triton kernel. Works for BOTH:
  - C=16, N=256  (tiny Swinv2, tiny swin)
  - C=96, N=65536 (gagan3012 swin_arocr_tiny)

The kernel handles non-contiguous input (tmp_7 is transposed from conv2d output),
computes layer norm with strided loads, and writes a contiguous output.
Subsequent view/pad/view/permute (window partition) remains in the graph.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: layer norm, bfloat16
# Single-row variant (for C=16, N=256 or general use)
# ---------------------------------------------------------------------------
@triton.jit
def _ln_bf16_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, eps,
    stride_row, stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_row
    Y_row = Y_ptr + row * C
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C

    x = tl.load(X_row + cols * stride_col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / C
    rstd = tl.rsqrt(var + eps)
    x_hat = xc * rstd

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * w + b
    tl.store(Y_row + cols, y.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Triton kernel: layer norm, float16
# Single-row variant
# ---------------------------------------------------------------------------
@triton.jit
def _ln_fp16_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, eps,
    stride_row, stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_row
    Y_row = Y_ptr + row * C
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C

    x = tl.load(X_row + cols * stride_col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / C
    rstd = tl.rsqrt(var + eps)
    x_hat = xc * rstd

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * w + b
    tl.store(Y_row + cols, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Triton kernel: layer norm, bfloat16, multi-row variant
# Processes BLOCK_ROWS consecutive rows per program.
# Amortizes weight/bias loads and reduces total grid size.
# ---------------------------------------------------------------------------
@triton.jit
def _ln_bf16_multirow_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, eps,
    stride_row, stride_col,
    BLOCK_ROWS: tl.constexpr,   # rows processed per program
    BLOCK_SIZE: tl.constexpr,   # >= C, power of 2
):
    row_start = tl.program_id(0) * BLOCK_ROWS
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C

    # Load weight & bias once; reused across BLOCK_ROWS rows
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    for r in tl.static_range(BLOCK_ROWS):
        row = row_start + r
        X_row = X_ptr + row * stride_row
        Y_row = Y_ptr + row * C

        x = tl.load(X_row + cols * stride_col, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / C
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / C
        rstd = tl.rsqrt(var + eps)
        x_hat = xc * rstd

        y = x_hat * w + b
        tl.store(Y_row + cols, y.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Triton kernel: layer norm, float16, multi-row variant
# ---------------------------------------------------------------------------
@triton.jit
def _ln_fp16_multirow_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, eps,
    stride_row, stride_col,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_ROWS
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    for r in tl.static_range(BLOCK_ROWS):
        row = row_start + r
        X_row = X_ptr + row * stride_row
        Y_row = Y_ptr + row * C

        x = tl.load(X_row + cols * stride_col, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / C
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / C
        rstd = tl.rsqrt(var + eps)
        x_hat = xc * rstd

        y = x_hat * w + b
        tl.store(Y_row + cols, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Pattern: layer_norm(any normalized_shape) + dropout(p=0.0)
# The normalized_shape is passed as an argument so it acts as a wildcard —
# matching BOTH (16,) for tiny swin and (96,) for gagan3012.
# ---------------------------------------------------------------------------
def pattern(tmp_7, in_2, in_1, normalized_shape):
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, normalized_shape, in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


def replacement_args(tmp_7, in_2, in_1, normalized_shape):
    # Do NOT pass normalized_shape to the replacement — C is read from tmp_7.shape
    return (tmp_7, in_2, in_1)


# ---------------------------------------------------------------------------
# Helper: choose BLOCK_SIZE (next power of 2 >= C)
# ---------------------------------------------------------------------------
def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Optimised replacement wrapper (handles any C dynamically)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_ln_unified(tmp_7, in_2, in_1):
    """
    tmp_7 : [B, N, C] with strides [N*C, 1, N]  (non-contiguous, post-transpose)
    in_2  : [C]  weight (gamma)
    in_1  : [C]  bias   (beta)
    Returns: tmp_9: [B, N, C] contiguous
    """
    B, N, C = tmp_7.shape
    _, stride_n, stride_c = tmp_7.stride()

    tmp_9 = torch.empty(B, N, C, dtype=tmp_7.dtype, device=tmp_7.device)
    total_rows = B * N

    if C == 96:
        # gagan3012: N=65536, C=96 — single-row kernel with BLOCK_SIZE=128
        # 65536 thread blocks for maximum GPU occupancy
        if tmp_7.dtype == torch.bfloat16:
            _ln_bf16_kernel[(total_rows,)](
                tmp_7, in_2, in_1, tmp_9,
                C, 1e-5,
                stride_n, stride_c,
                BLOCK_SIZE=128,
            )
        else:
            _ln_fp16_kernel[(total_rows,)](
                tmp_7, in_2, in_1, tmp_9,
                C, 1e-5,
                stride_n, stride_c,
                BLOCK_SIZE=128,
            )
    elif C == 16:
        # tiny swin: N=256, C=16 — exact BLOCK_SIZE=16 (no masking needed)
        if tmp_7.dtype == torch.bfloat16:
            _ln_bf16_kernel[(total_rows,)](
                tmp_7, in_2, in_1, tmp_9,
                C, 1e-5,
                stride_n, stride_c,
                BLOCK_SIZE=16,
                num_warps=1,
            )
        else:
            _ln_fp16_kernel[(total_rows,)](
                tmp_7, in_2, in_1, tmp_9,
                C, 1e-5,
                stride_n, stride_c,
                BLOCK_SIZE=16,
                num_warps=1,
            )
    else:
        # General fallback
        BLOCK_SIZE = _next_pow2(C)
        if tmp_7.dtype == torch.bfloat16:
            _ln_bf16_kernel[(total_rows,)](
                tmp_7, in_2, in_1, tmp_9,
                C, 1e-5,
                stride_n, stride_c,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            _ln_fp16_kernel[(total_rows,)](
                tmp_7, in_2, in_1, tmp_9,
                C, 1e-5,
                stride_n, stride_c,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return tmp_9


def replacement_func():
    return fused_ln_unified