"""
Shared Triton kernels for fast spatial mean reduction over (H, W) dimensions.

Optimization: Replace torch's mean((2,3), keepdim=True) with a Triton kernel
that accumulates in float32 for numerical stability, then casts back to input dtype.

Grid: one program per (B*C) slice.
Loop over H*W in BLOCK_HW-sized chunks.
Dtype passed as a compile-time constant (DTYPE_CODE) to avoid a sentinel read.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=1),
        triton.Config({'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16),
    ],
    key=['HW'],   # DTYPE_CODE is constexpr → separate compiled variant per dtype
)
@triton.jit
def _spatial_mean_kernel(
    in_ptr,
    mean_ptr,
    HW,
    DTYPE_CODE: tl.constexpr,   # 0=float16, 1=bfloat16, 2=float32
    BLOCK_HW: tl.constexpr,
):
    """
    One program per (B*C) slice.
    'other=0.0' in tl.load already zeroes out-of-bounds lanes, so no
    tl.where is needed – saving a comparison + select per element.
    """
    pid = tl.program_id(0)
    base = pid * HW
    acc = 0.0

    for start in range(0, HW, BLOCK_HW):
        offsets = start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        # Invalid lanes get 0.0 → don't inflate the sum
        x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0)
        acc = acc + tl.sum(x.to(tl.float32), axis=0)

    mean_f32 = acc / HW
    if DTYPE_CODE == 0:
        tl.store(mean_ptr + pid, mean_f32.to(tl.float16))
    elif DTYPE_CODE == 1:
        tl.store(mean_ptr + pid, mean_f32.to(tl.bfloat16))
    else:
        tl.store(mean_ptr + pid, mean_f32)


_DTYPE_CODES = {
    torch.float16:  0,
    torch.bfloat16: 1,
    torch.float32:  2,
}


@torch.fx.wrap
def triton_spatial_mean(x):
    """
    Computes x.mean((2, 3), keepdim=True) using a fast Triton kernel.
    Returns shape (B, C, 1, 1) with same dtype as x.
    """
    B, C, H, W = x.shape
    BC = B * C
    HW = H * W
    mean = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)
    dtype_code = _DTYPE_CODES[x.dtype]
    _spatial_mean_kernel[(BC,)](x, mean, HW, DTYPE_CODE=dtype_code)
    return mean