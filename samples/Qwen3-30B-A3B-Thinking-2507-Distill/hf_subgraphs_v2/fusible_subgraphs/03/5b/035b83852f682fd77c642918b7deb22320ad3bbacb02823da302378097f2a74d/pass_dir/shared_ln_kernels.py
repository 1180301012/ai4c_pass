"""
Shared LayerNorm kernels used by all LayerNorm passes.
Both LayerNorm_432 and LayerNorm_192 import from here so they
return the SAME replacement_func object, satisfying the
framework's "1 unique replacement_func" limit.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=1),
    ],
    key=['M'],
)
@triton.jit
def _ln_kernel_432(
    X, Y, W, B,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    N: tl.constexpr = 432
    EPS: tl.constexpr = 1e-6
    row = tl.program_id(0)
    X += row * N
    Y += row * N
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) * (1.0 / N)
    xc = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(xc * xc, axis=0) * (1.0 / N)
    xh   = xc * tl.rsqrt(var + EPS)
    w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + offs, xh * w + b, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8,  num_stages=1),
    ],
    key=['M'],
)
@triton.jit
def _ln_kernel_192(
    X, Y, W, B,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    N: tl.constexpr = 192
    EPS: tl.constexpr = 1e-6
    row = tl.program_id(0)
    X += row * N
    Y += row * N
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) * (1.0 / N)
    xc = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(xc * xc, axis=0) * (1.0 / N)
    xh   = xc * tl.rsqrt(var + EPS)
    w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + offs, xh * w + b, mask=mask)


# Pre-built dispatch table — avoids if/elif comparison overhead
_ROUTE = {
    "ln_432": _ln_kernel_432,
    "ln_192": _ln_kernel_192,
}


@torch.fx.wrap
def dispatch_layer_norm(x, weight, bias, route):
    """
    Unified entry point for all LayerNorm passes.
    `route` is "ln_432" or "ln_192" to select the right kernel.
    We assume x, weight, bias are already on CUDA (framework guarantees it).
    """
    out = torch.empty_like(x)
    if route == "ln_432":
        M = x.numel() // 432
        _ROUTE["ln_432"][(M,)](x, out, weight, bias, M)
    elif route == "ln_192":
        M = x.numel() // 192
        _ROUTE["ln_192"][(M,)](x, out, weight, bias, M)
    return out