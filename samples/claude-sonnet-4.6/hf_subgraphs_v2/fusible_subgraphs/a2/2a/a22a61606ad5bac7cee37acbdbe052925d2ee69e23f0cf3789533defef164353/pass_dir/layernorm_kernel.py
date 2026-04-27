"""
Shared Triton layer-norm forward kernel used by all FuseDropoutLayerNorm passes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_fwd_kernel(
    X_ptr,      # input  pointer  (M x N, row-major)
    W_ptr,      # weight pointer  (N,)
    B_ptr,      # bias   pointer  (N,)
    Y_ptr,      # output pointer  (M x N, row-major)
    stride,     # row stride == N for contiguous tensors
    N,          # hidden size (last dim)
    eps,        # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,   # must be >= N and a power of 2
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    row_start = row * stride
    # Load row – preserve original dtype for later cast-back
    x_orig = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)
    x = x_orig.to(tl.float32)

    # Mean
    mean = tl.sum(x, axis=0) / N

    # Variance (population)
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N

    # Normalise
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = xc * rstd

    # Weight and bias
    w = tl.load(W_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    y = xn * w + b

    # Store in the original dtype
    tl.store(Y_ptr + row_start + offsets, y.to(x_orig.dtype), mask=mask)