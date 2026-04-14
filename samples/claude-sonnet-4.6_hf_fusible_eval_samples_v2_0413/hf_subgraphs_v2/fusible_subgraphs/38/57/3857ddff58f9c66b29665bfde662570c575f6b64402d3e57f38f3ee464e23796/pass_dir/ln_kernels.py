"""
Shared Triton kernels and dispatch function used by both
FuseAddReshapeLayerNorm_768 and FuseAddReshapeLayerNorm_16 passes.

Both pass files import `triton_layernorm_dispatch` from here so that
`replacement_func()` returns the SAME function object in every pass —
satisfying the replacement_func_limit.
"""

import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────
#  Triton kernel: layer-norm for N = 768 (BLOCK_SIZE = 1024)
# ──────────────────────────────────────────────────────────────
@triton.jit
def _ln_kernel_768(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, 0) / N
    xc   = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(xc * xc, 0) / N
    xn   = xc * (1.0 / tl.sqrt(var + eps))

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xn * w + b

    tl.store(out_ptr + row * N + cols, y, mask=mask)


# ──────────────────────────────────────────────────────────────
#  Triton kernel: layer-norm for N = 16 (BLOCK_SIZE = 32)
#  Using 32 threads (full warp) even though N = 16.
# ──────────────────────────────────────────────────────────────
@triton.jit
def _ln_kernel_16(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, 0) / N
    xc   = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(xc * xc, 0) / N
    xn   = xc * (1.0 / tl.sqrt(var + eps))

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xn * w + b

    tl.store(out_ptr + row * N + cols, y, mask=mask)


# ──────────────────────────────────────────────────────────────
#  Shared dispatch — the ONE replacement_func used by all passes
# ──────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_layernorm_dispatch(in_0, in_1, x, route):
    """
    in_0  : bias   [N]
    in_1  : weight [N]
    x     : input  [n_rows, N]  (already reshaped)
    route : "n768" or "n16"
    """
    out = torch.empty_like(x)          # same shape [n_rows, N] and dtype
    n_rows = x.shape[0]

    if route == "n768":
        _ln_kernel_768[(n_rows,)](
            x, in_1, in_0, out,
            768, 1e-5,
            BLOCK_SIZE=1024,
        )
    elif route == "n16":
        _ln_kernel_16[(n_rows,)](
            x, in_1, in_0, out,
            16, 1e-5,
            BLOCK_SIZE=32,
        )

    return out