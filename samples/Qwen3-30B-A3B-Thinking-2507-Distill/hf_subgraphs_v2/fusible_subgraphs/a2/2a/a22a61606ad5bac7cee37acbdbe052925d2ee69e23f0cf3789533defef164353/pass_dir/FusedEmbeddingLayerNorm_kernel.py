import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Optimized layer_norm kernel with loop-based reduction.
# BLOCK_D=256 divides all target D values (32, 64, 768, 1024) evenly →
# no masking overhead when BLOCK_D divides D.  Autotune selects best config.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
        triton.Config({'BLOCK_D': 512}, num_warps=4),
        triton.Config({'BLOCK_D': 512}, num_warps=8),
        triton.Config({'BLOCK_D': 1024}, num_warps=4),
        triton.Config({'BLOCK_D': 1024}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)

    # Pass 1: accumulate mean
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    n_iters = tl.cdiv(D, BLOCK_D)
    for i in range(n_iters):
        offs = i * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x
    mean = tl.sum(acc, axis=0) / D

    # Pass 2: accumulate variance
    var_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i in range(n_iters):
        offs = i * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        xc = tl.where(mask, x - mean, 0.0)
        var_acc += xc * xc
    var = tl.sum(var_acc, axis=0) / D
    rstd = tl.rsqrt(var + eps)

    # Pass 3: normalize, scale, bias → store
    for i in range(n_iters):
        offs = i * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + row * D + offs, out.to(x_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_layer_norm(x, w, b, route):
    if route == "ln_1024":
        D = 1024
    elif route == "ln_768":
        D = 768
    elif route == "ln_64":
        D = 64
    else:  # "ln_32"
        D = 32
    N = x.numel() // D
    out = torch.empty_like(x)
    layer_norm_kernel[(N,)](x, w, b, out, N, D, 1e-12)
    return out