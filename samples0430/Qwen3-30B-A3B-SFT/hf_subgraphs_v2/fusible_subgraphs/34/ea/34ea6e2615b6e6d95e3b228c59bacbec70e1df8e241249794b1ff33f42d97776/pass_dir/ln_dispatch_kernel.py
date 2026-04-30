"""
Shared Triton layer-norm kernel + dispatch wrapper for C768 and C1024.
Both pass files import _dispatch from here so replacement_func() returns
the SAME Python object in both files (satisfying replacement_func_limit).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, C,
    sx1, sx2,   # x strides: spatial dim, channel dim
    sy1, sy2,   # y strides: spatial dim, channel dim
    eps,
    BLOCK_C: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    x  = tl.load(X_ptr + row * sx1 + cols * sx2, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    mean    = tl.sum(xf, axis=0) / C
    diff    = tl.where(mask, xf - mean, 0.0)
    var     = tl.sum(diff * diff, axis=0) / C
    inv_std = tl.rsqrt(var + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv_std * w + b

    tl.store(Y_ptr + row * sy1 + cols * sy2, y.to(x.dtype), mask=mask)


@torch.fx.wrap
def _dispatch(x, weight, bias, route):
    """Shared dispatch wrapper – identical across all route-specific passes."""
    C = x.shape[-1]
    N = x.numel() // C
    y   = torch.empty_like(x)
    sy1 = y.stride(-2)
    sy2 = y.stride(-1)
    eps = 1e-5

    if route == "C768":
        _ln_kernel[(N,)](
            x, weight, bias, y,
            N, C,
            x.stride(-2), x.stride(-1),
            sy1, sy2, eps,
            BLOCK_C=1024, num_warps=8,
        )
    elif route == "C1024":
        _ln_kernel[(N,)](
            x, weight, bias, y,
            N, C,
            x.stride(-2), x.stride(-1),
            sy1, sy2, eps,
            BLOCK_C=1024, num_warps=8,
        )
    return y