"""
Pass: FuseLayerNormWindowPartition_C16_ws2

Fuses:
  layer_norm(normalized_shape=(16,)) + dropout(p=0.0)

into a single Triton kernel. The subsequent view/pad/view/permute (window
partition) remains in the graph and operates on the contiguous output tmp_9.

Targets:
  - hf-tiny-model-private_tiny-random-Swinv2ForImageClassification (bfloat16 & float16)
    in_0: [1,3,32,32], conv stride=(2,2) => C=16, N=H*W=16*16=256
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: layer norm bfloat16, strided input, contiguous output
# ---------------------------------------------------------------------------
@triton.jit
def _ln_c16_bf16_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, eps,
    stride_row, stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    # BLOCK_SIZE == C == 16: no masked elements, no variance bug
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_row
    Y_row = Y_ptr + row * C
    cols = tl.arange(0, BLOCK_SIZE)  # [0..15], all valid

    x = tl.load(X_row + cols * stride_col).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / C
    rstd = tl.rsqrt(var + eps)
    x_hat = xc * rstd

    w = tl.load(W_ptr + cols).to(tl.float32)
    b = tl.load(B_ptr + cols).to(tl.float32)
    y = x_hat * w + b
    tl.store(Y_row + cols, y.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Triton kernel: layer norm float16, strided input, contiguous output
# ---------------------------------------------------------------------------
@triton.jit
def _ln_c16_fp16_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    C, eps,
    stride_row, stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    # BLOCK_SIZE == C == 16: no masked elements, no variance bug
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_row
    Y_row = Y_ptr + row * C
    cols = tl.arange(0, BLOCK_SIZE)  # [0..15], all valid

    x = tl.load(X_row + cols * stride_col).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / C
    rstd = tl.rsqrt(var + eps)
    x_hat = xc * rstd

    w = tl.load(W_ptr + cols).to(tl.float32)
    b = tl.load(B_ptr + cols).to(tl.float32)
    y = x_hat * w + b
    tl.store(Y_row + cols, y.to(tl.float16))


# ---------------------------------------------------------------------------
# Pattern: layer_norm + dropout  (no pad — avoids pad arg-matching issues)
# ---------------------------------------------------------------------------
def pattern(tmp_7, in_2, in_1):
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
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
def fused_ln_c16(tmp_7, in_2, in_1):
    """
    tmp_7 : [1, 256, 16] with strides [4096, 1, 256]  (non-contiguous)
    in_2  : [16]  weight (gamma)
    in_1  : [16]  bias   (beta)
    Returns: tmp_9: [1, 256, 16] contiguous
    """
    B, N, C = tmp_7.shape            # 1, 256, 16
    _, stride_n, stride_c = tmp_7.stride()

    tmp_9 = torch.empty(B, N, C, dtype=tmp_7.dtype, device=tmp_7.device)
    total_rows = B * N               # 256

    if tmp_7.dtype == torch.bfloat16:
        _ln_c16_bf16_kernel[(total_rows,)](
            tmp_7, in_2, in_1, tmp_9,
            C, 1e-5,
            stride_n, stride_c,
            BLOCK_SIZE=16,
        )
    else:
        _ln_c16_fp16_kernel[(total_rows,)](
            tmp_7, in_2, in_1, tmp_9,
            C, 1e-5,
            stride_n, stride_c,
            BLOCK_SIZE=16,
        )

    return tmp_9


def replacement_func():
    return fused_ln_c16