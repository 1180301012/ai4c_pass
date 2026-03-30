"""
Pass: FuseLayerNormWindowPartition_C96_ws8

Fuses:
  layer_norm(normalized_shape=(96,)) + dropout(p=0.0)

into a single Triton kernel. The subsequent view/pad/view/permute (window
partition) remains in the graph and operates on the contiguous output tmp_9.

Targets:
  - gagan3012_swin_arocr_tiny (bfloat16 & float16)
    in_0: [1,3,1024,1024], conv stride=(4,4) => C=96, N=H*W=256*256=65536
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: layer norm bfloat16, strided input, contiguous output
# ---------------------------------------------------------------------------
@triton.jit
def _ln_c96_bf16_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, eps,
    stride_row, stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    # BLOCK_SIZE=128 > C=96: 32 masked elements — must zero them before variance
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_row
    Y_row = Y_ptr + row * C
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C

    x = tl.load(X_row + cols * stride_col, mask=mask, other=0.0).to(tl.float32)

    # Mean: masked positions loaded as 0.0, so sum is unaffected; divide by C (correct)
    mean = tl.sum(x, axis=0) / C

    # Centered: masked positions give (0 - mean) = -mean, which would corrupt variance
    xc = x - mean
    # *** Fix: zero out masked positions before summing squared deviations ***
    xc = tl.where(mask, xc, 0.0)

    var = tl.sum(xc * xc, axis=0) / C
    rstd = tl.rsqrt(var + eps)
    x_hat = xc * rstd          # 0 for masked positions

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * w + b
    tl.store(Y_row + cols, y.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Triton kernel: layer norm float16, strided input, contiguous output
# ---------------------------------------------------------------------------
@triton.jit
def _ln_c96_fp16_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, eps,
    stride_row, stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    # BLOCK_SIZE=128 > C=96: 32 masked elements — must zero them before variance
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_row
    Y_row = Y_ptr + row * C
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C

    x = tl.load(X_row + cols * stride_col, mask=mask, other=0.0).to(tl.float32)

    # Mean: masked positions loaded as 0.0, so sum is unaffected; divide by C (correct)
    mean = tl.sum(x, axis=0) / C

    # Centered: masked positions give (0 - mean) = -mean, which would corrupt variance
    xc = x - mean
    # *** Fix: zero out masked positions before summing squared deviations ***
    xc = tl.where(mask, xc, 0.0)

    var = tl.sum(xc * xc, axis=0) / C
    rstd = tl.rsqrt(var + eps)
    x_hat = xc * rstd          # 0 for masked positions

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * w + b
    tl.store(Y_row + cols, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Pattern: layer_norm + dropout  (no pad — avoids pad arg-matching issues)
# ---------------------------------------------------------------------------
def pattern(tmp_7, in_2, in_1):
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (96,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------
def replacement_args(tmp_7, in_2, in_1):
    return (tmp_7, in_2, in_1)


# ---------------------------------------------------------------------------
# Optimised replacement wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_ln_c96(tmp_7, in_2, in_1):
    """
    tmp_7 : [1, 65536, 96] with strides [96*65536, 1, 65536]  (non-contiguous)
    in_2  : [96]  weight (gamma)
    in_1  : [96]  bias   (beta)
    Returns: tmp_9: [1, 65536, 96] contiguous
    """
    B, N, C = tmp_7.shape            # 1, 65536, 96
    _, stride_n, stride_c = tmp_7.stride()   # 1, 65536

    tmp_9 = torch.empty(B, N, C, dtype=tmp_7.dtype, device=tmp_7.device)
    total_rows = B * N               # 65536

    if tmp_7.dtype == torch.bfloat16:
        _ln_c96_bf16_kernel[(total_rows,)](
            tmp_7, in_2, in_1, tmp_9,
            C, 1e-5,
            stride_n, stride_c,
            BLOCK_SIZE=128,
        )
    else:
        _ln_c96_fp16_kernel[(total_rows,)](
            tmp_7, in_2, in_1, tmp_9,
            C, 1e-5,
            stride_n, stride_c,
            BLOCK_SIZE=128,
        )

    return tmp_9


def replacement_func():
    return fused_ln_c96