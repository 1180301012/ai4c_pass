"""
Shared Triton layer-norm kernels and dispatch function for the
LayerNorm_* passes.  Having a single @torch.fx.wrap dispatch entry-point
with a route string argument prevents the pass-replacement func-limit drop.
"""

import torch
import triton
import triton.language as tl


# ── autotune: empirically select best num_warps per D ────────────────────────

@triton.jit
def _ln_fwd_32(X_ptr, W_ptr, B_ptr, Y_ptr, eps, N: tl.constexpr):
    """Optimized for D=32 with small n_rows."""
    row = tl.program_id(0)
    X_row = X_ptr + row * N
    Y_row = Y_ptr + row * N
    # N=32 is a power of 2 — directly usable in tl.arange
    offs = tl.arange(0, N)
    x = tl.load(X_row + offs).to(tl.float32)
    mean = tl.sum(x) / N
    x_c = x - mean
    var = tl.sum(x_c * x_c) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W_ptr + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)
    y = x_c * rstd * w + b
    tl.store(Y_row + offs, y)


# ── N=384 (next power of 2 = 512) ───────────────────────────────────────────

@triton.jit
def _ln_fwd_384_masked(X_ptr, W_ptr, B_ptr, Y_ptr, eps, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    X_row = X_ptr + row * N
    Y_row = Y_ptr + row * N
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X_row + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_c = x - mean
    var = tl.sum(x_c * x_c, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_c * rstd * w + b
    tl.store(Y_row + offs, y, mask=mask)


@triton.jit
def _ln_fwd_768_masked(X_ptr, W_ptr, B_ptr, Y_ptr, eps, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    X_row = X_ptr + row * N
    Y_row = Y_ptr + row * N
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X_row + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_c = x - mean
    var = tl.sum(x_c * x_c, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_c * rstd * w + b
    tl.store(Y_row + offs, y, mask=mask)


# ── fallback for BLOCK_N > D (masking) ───────────────────────────────────────
# kept for reference; actual kernel uses the above per-D specialisations

@triton.jit
def _ln_fwd_masked(X_ptr, W_ptr, B_ptr, Y_ptr, eps, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    X_row = X_ptr + row * N
    Y_row = Y_ptr + row * N
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X_row + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N          # padding elements are 0 → no bias
    x_c = x - mean
    var = tl.sum(x_c * x_c, axis=0) / N  # padding contributes (0-mean)^2
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_c * rstd * w + b
    tl.store(Y_row + offs, y, mask=mask)


# ── shared dispatch (single @torch.fx.wrap entry-point) ──────────────────────

@torch.fx.wrap
def dispatch_ln(in_0, in_1, in_4, route):
    """
    in_0  = bias   [D]
    in_1  = weight [D]
    in_4  = input  [..., D]
    route = string identifies the D to use
    """
    n_rows = in_4.shape[0] * in_4.shape[1]   # compute once
    y = torch.empty_like(in_4)                # allocate output once

    if route == "D384":
        # next_power_of_2(384) = 512; 2601 rows on A30 (108 SMs)
        _ln_fwd_384_masked[(n_rows,)](
            in_4, in_1, in_0, y, 1e-12,
            N=384, BLOCK_N=512, num_warps=4,
        )
    elif route == "D768":
        # next_power_of_2(768) = 1024; 2601 rows on A30 (108 SMs)
        _ln_fwd_768_masked[(n_rows,)](
            in_4, in_1, in_0, y, 1e-12,
            N=768, BLOCK_N=1024, num_warps=8,
        )
    else:  # "D32"
        _ln_fwd_32[(n_rows,)](
            in_4, in_1, in_0, y, 1e-12,
            N=32, num_warps=1,
        )
    return y