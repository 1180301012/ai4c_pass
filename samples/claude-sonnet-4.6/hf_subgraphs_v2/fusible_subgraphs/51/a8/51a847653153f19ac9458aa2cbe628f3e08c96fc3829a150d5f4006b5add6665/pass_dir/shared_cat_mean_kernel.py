"""
Shared Triton kernel for spatial mean: [B,C,H,W] → [B,C,1,1]
Uses autotune to find best BLOCK_HW/num_warps/num_stages for each shape.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 16},   num_warps=1, num_stages=1),
        triton.Config({'BLOCK_HW': 32},   num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 32},   num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 64},   num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 64},   num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 64},   num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=4),
    ],
    key=['BC', 'HW'],
)
@triton.jit
def _spatial_mean_kernel(
    x_ptr, mean_ptr,
    BC, HW,
    DTYPE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    One program per (b*C + c) index.
    Computes mean over HW elements with a loop over BLOCK_HW-sized tiles.
    Accumulates in fp32 for numerical accuracy.
    """
    pid  = tl.program_id(0)
    base = x_ptr + pid * HW

    total = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        vals = tl.load(base + offs, mask=mask, other=0)
        total += vals.to(tl.float32)

    mean_f32 = tl.sum(total) / HW

    if DTYPE == 2:
        tl.store(mean_ptr + pid, mean_f32.to(tl.bfloat16))
    elif DTYPE == 1:
        tl.store(mean_ptr + pid, mean_f32.to(tl.float16))
    else:
        tl.store(mean_ptr + pid, mean_f32)


@torch.fx.wrap
def triton_spatial_mean(x):
    """Compute spatial mean [B,C,H,W] → [B,C,1,1] using Triton."""
    B, C, H, W = x.shape
    HW     = H * W
    BC     = B * C
    dtype  = x.dtype
    device = x.device
    dtype_id = 2 if dtype == torch.bfloat16 else (1 if dtype == torch.float16 else 0)

    mean = torch.empty((B, C, 1, 1), dtype=dtype, device=device)

    _spatial_mean_kernel[(BC,)](
        x, mean, BC, HW,
        DTYPE=dtype_id,
    )
    return mean