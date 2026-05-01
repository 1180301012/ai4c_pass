"""
Shared Triton GELU kernel used by FuseGeluDropout and FuseGeluApproxNoneDropout passes.
Both GELU variants (default and approximate='none') compute the exact same erf-based formula.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_fwd_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load with streaming hint (won't be reused after GELU)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    # Upcast to fp32 for numerical accuracy
    x_f32 = x.to(tl.float32)
    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out_f32 = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * INV_SQRT2))
    out = out_f32.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask, eviction_policy='evict_first')


@torch.fx.wrap
def shared_gelu_activation(x, route):
    """
    Shared dispatcher for all GELU+dropout fusion passes.
    'route' parameter differentiates which pass triggered this (not used for computation).
    """
    N = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 4096
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    _gelu_fwd_kernel[(num_blocks,)](
        x, out, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out