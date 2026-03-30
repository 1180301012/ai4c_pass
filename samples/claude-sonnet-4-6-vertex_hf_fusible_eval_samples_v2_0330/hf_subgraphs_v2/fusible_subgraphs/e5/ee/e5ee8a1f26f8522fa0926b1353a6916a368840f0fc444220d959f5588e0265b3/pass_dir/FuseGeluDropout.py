"""
Pass: FuseGeluDropout
Fuses torch.nn.functional.gelu(x) + torch.nn.functional.dropout(out, 0.0, False, False)
into a single optimized Triton GELU kernel, eliminating the no-op dropout call.

This matches the default GELU (no 'approximate' keyword argument).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _gelu_exact_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load in native dtype
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Upcast to float32 for accurate erf computation
    x_f32 = x.to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    cdf = 0.5 * (1.0 + tl.math.erf(x_f32 * inv_sqrt2))
    out_f32 = x_f32 * cdf

    # Cast back to original dtype and store
    out = out_f32.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_gelu_dropout(x):
    """Fused GELU + no-op dropout replacement."""
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _gelu_exact_kernel[grid](x, out, n_elements)
    return out


# ── Pattern & Replacement ────────────────────────────────────────────────────

def pattern(x):
    tmp = torch.nn.functional.gelu(x)
    out = torch.nn.functional.dropout(tmp, 0.0, False, False)
    return out


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_gelu_dropout