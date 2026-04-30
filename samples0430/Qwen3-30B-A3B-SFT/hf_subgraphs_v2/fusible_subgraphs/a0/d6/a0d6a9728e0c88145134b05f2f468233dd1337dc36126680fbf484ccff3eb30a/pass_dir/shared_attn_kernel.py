"""
Shared Triton kernels for fused attention:
  Pass 1: bmm(query, key) → softmax  (returns attn weights [B,1,D])
  Pass 2: bmm(attn_weights, value) → view → transpose → reshape  (returns flat [1,1,BHD])

Key insight: K=1 (key shape [B, D, 1]) → softmax is trivial (scalar per head).
"""

import torch
import triton
import triton.language as tl


# ── Pass 1 kernel: scaled dot-product → softmax  ──────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32}, num_warps=1),
        triton.Config({'BLOCK_D': 32}, num_warps=2),
        triton.Config({'BLOCK_D': 32}, num_warps=4),
    ],
    key=['D'],
)
@triton.jit
def _softmax_kernel_32(
    query_ptr, key_ptr, output_ptr,
    B, H,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    b  = bh // H
    d_range = tl.arange(0, BLOCK_D)

    q = tl.load(query_ptr + bh * D + d_range).to(tl.float32)
    k = tl.load(key_ptr   + b  * D + d_range).to(tl.float32)

    scale = 1.0 / tl.sqrt(D * 1.0)
    qk    = tl.sum(q * k, axis=0) * scale
    exp_qk = tl.exp(qk)
    # Trivial softmax of a scalar: exp(x) / 1 = exp(x)
    attn  = exp_qk  # single scalar = softmax weight

    # Store as scalar per (b, h)
    tl.store(output_ptr + bh, attn.to(query_ptr.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64}, num_warps=1),
        triton.Config({'BLOCK_D': 64}, num_warps=2),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
    ],
    key=['D'],
)
@triton.jit
def _softmax_kernel_64(
    query_ptr, key_ptr, output_ptr,
    B, H,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    b  = bh // H
    d_range = tl.arange(0, BLOCK_D)

    q = tl.load(query_ptr + bh * D + d_range).to(tl.float32)
    k = tl.load(key_ptr   + b  * D + d_range).to(tl.float32)

    scale = 1.0 / tl.sqrt(D * 1.0)
    qk    = tl.sum(q * k, axis=0) * scale
    exp_qk = tl.exp(qk)
    attn  = exp_qk

    tl.store(output_ptr + bh, attn.to(query_ptr.dtype.element_ty))


# ── Pass 2 kernel: weighted sum over values + flatten ─────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32}, num_warps=1),
        triton.Config({'BLOCK_D': 32}, num_warps=2),
        triton.Config({'BLOCK_D': 32}, num_warps=4),
    ],
    key=['D'],
)
@triton.jit
def _weighted_sum_kernel_32(
    attn_ptr, value_ptr, output_ptr,
    BH,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh     = tl.program_id(0)
    d_range = tl.arange(0, BLOCK_D)

    # Load single attention scalar for this (b, h)
    attn = tl.load(attn_ptr + bh).to(tl.float32)

    # Load value vector [b, 0, :]
    v = tl.load(value_ptr + bh * D + d_range).to(tl.float32)

    # Weighted sum
    out = v * attn

    # Store into flat output [1, 1, BHD]
    tl.store(output_ptr + bh * D + d_range, out.to(attn_ptr.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64}, num_warps=1),
        triton.Config({'BLOCK_D': 64}, num_warps=2),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
    ],
    key=['D'],
)
@triton.jit
def _weighted_sum_kernel_64(
    attn_ptr, value_ptr, output_ptr,
    BH,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh     = tl.program_id(0)
    d_range = tl.arange(0, BLOCK_D)

    attn = tl.load(attn_ptr + bh).to(tl.float32)
    v    = tl.load(value_ptr + bh * D + d_range).to(tl.float32)

    out  = v * attn

    tl.store(output_ptr + bh * D + d_range, out.to(attn_ptr.dtype.element_ty))


# ── Pass 2 kernel: weighted sum over values + flatten ─────────────────────────

@triton.jit
def _weighted_sum_kernel_32(
    attn_ptr, value_ptr, output_ptr,
    BH,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh     = tl.program_id(0)
    d_range = tl.arange(0, BLOCK_D)

    # Load attention scalar for this (b,h) from attn_weights [B,H,D]
    attn = tl.load(attn_ptr + bh * D).to(tl.float32)

    # Load value vector [b, 0, :]
    v = tl.load(value_ptr + bh * D + d_range).to(tl.float32)

    out = v * attn

    tl.store(output_ptr + bh * D + d_range, out.to(attn_ptr.dtype.element_ty))


@triton.jit
def _weighted_sum_kernel_64(
    attn_ptr, value_ptr, output_ptr,
    BH,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh     = tl.program_id(0)
    d_range = tl.arange(0, BLOCK_D)

    attn = tl.load(attn_ptr + bh * D).to(tl.float32)
    v    = tl.load(value_ptr + bh * D + d_range).to(tl.float32)

    out  = v * attn

    tl.store(output_ptr + bh * D + d_range, out.to(attn_ptr.dtype.element_ty))


# ── Shared dispatch wrapper (3 tensor args + route) ──────────────────────────

@torch.fx.wrap
def attn_fused_dispatch(a, b, route):
    """Direct FX-graph node dispatcher – keeps pattern matching correct."""
    if route == "route_32":
        return _launch_kernel_32(a, b, torch.empty((1, 1, 256), dtype=a.dtype, device=a.device))
    elif route == "route_64":
        return _launch_kernel_64(a, b, torch.empty((1, 1, 1024), dtype=a.dtype, device=a.device))
    elif route == "route_weighted_32":
        return _launch_weighted_kernel_32(a, b)
    else:  # route_weighted_64
        return _launch_weighted_kernel_64(a, b)

    _attn_kernel_32[(BH,)](
        query.view(BH, D), key.view(B, D), output,
        B=B, D=D, BLOCK_D=D,
        num_warps=1,
    )
    return output


@torch.fx.wrap
def _launch_kernel_64(query, key, output):
    B  = query.shape[0]
    H  = query.shape[1]
    D  = 64
    BH  = B * H
    _attn_kernel_64[(BH,)](
        query.view(BH, D), key.view(B, D), output,
        B=B, D=D, BLOCK_D=D,
        num_warps=1,
    )
    return output


@torch.fx.wrap
def _launch_weighted_kernel_32(a, b):
    B  = a.shape[0]
    H  = a.shape[1]
    D  = 32
    BH  = B * H
    BHD = BH * D
    output = torch.empty((1, 1, BHD), dtype=a.dtype, device=a.device)
    _weighted_sum_kernel_32[(BH,)](
        a, b, output, BH=BH, D=D, BLOCK_D=D,
        num_warps=1,
    )
    return output


@torch.fx.wrap
def _launch_weighted_kernel_64(a, b):
    B  = a.shape[0]
    H  = a.shape[1]
    D  = 64
    BH  = B * H
    BHD = BH * D
    output = torch.empty((1, 1, BHD), dtype=a.dtype, device=a.device)
    _weighted_sum_kernel_64[(BH,)](
        a, b, output, BH=BH, D=D, BLOCK_D=D,
        num_warps=1,
    )
    return output