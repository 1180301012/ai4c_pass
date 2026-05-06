"""
AI4C Pass: Replace spatial mean with a fast Triton kernel.
The graph uses call_method('mean', ...) for mean and call_method('silu_', ...) for silu.
Single-output pattern (mean only) works; silu is a separate op the replacement reads from.
"""

import torch
import triton
import triton.language as tl


# Pattern: spatial mean only (single output)
def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: fast spatial mean (sum in fp32, store in original dtype)
# Two-pass: 1) sum with atomic-fp32, 2) cast to output dtype (for fp16/bf16)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128},    num_warps=2),
        triton.Config({'BLOCK_HW': 256},    num_warps=4),
        triton.Config({'BLOCK_HW': 512},    num_warps=4),
        triton.Config({'BLOCK_HW': 1024},   num_warps=4),
        triton.Config({'BLOCK_HW': 2048},   num_warps=8),
        triton.Config({'BLOCK_HW': 4096},   num_warps=8),
        triton.Config({'BLOCK_HW': 8192},   num_warps=16),
        triton.Config({'BLOCK_HW': 16384},  num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _mean_fast_kernel(
    x_ptr,
    out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """1D grid: one CTA per (b,c) pair. Streaming load (evict_first), fp32 acc."""
    pid      = tl.program_id(0)
    base     = pid * HW
    ptr_base = x_ptr + base
    acc      = 0.0
    for start in range(0, HW, BLOCK_HW):
        offs   = start + tl.arange(0, BLOCK_HW)
        mask   = offs < HW
        acc    = acc + tl.sum(tl.load(ptr_base + offs, mask=mask, other=0.0,
                                      eviction_policy='evict_first').to(tl.float32))
    tl.store(out_ptr + pid, acc.to(out_ptr.dtype.element_ty))


@triton.jit
def _cast_mean_kernel(
    src_ptr,
    dst_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst_ptr + offs,
             tl.load(src_ptr + offs, mask=mask, other=0.0).
             to(dst_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def triton_mean(x):
    """
    Fast spatial mean: x[B, C, H, W] -> mean_out[B, C, 1, 1].
    Single-pass: 1D grid (B*C), fp32 accumulation, cast to output dtype.
    """
    B, C, H, W = x.shape
    HW     = H * W
    n_bc   = B * C

    out_fp32 = torch.empty((n_bc,), dtype=torch.float32, device=x.device)
    out      = torch.empty((B, C, 1, 1), dtype=x.dtype,     device=x.device)

    _mean_fast_kernel[(n_bc,)](
        x_ptr=x, out_ptr=out_fp32, HW=HW,
    )
    BLOCK_C = 256
    if x.dtype != torch.float32:
        _cast_mean_kernel[(triton.cdiv(n_bc, BLOCK_C),)](
            src_ptr=out_fp32, dst_ptr=out, n=n_bc, BLOCK=BLOCK_C,
        )
    return out


def replacement_func():
    return triton_mean