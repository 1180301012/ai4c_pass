"""
Pass: Fast spatial mean over dims (2, 3), keepdim=True.
Matches x.mean((2, 3), keepdim=True) → single tensor [N, C, 1, 1].
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
    ],
    key=['N_C'],
)
@triton.jit
def _mean_hw_kernel(
    x_ptr, out_ptr,
    N_C, HW,
    BLOCK_HW: tl.constexpr,
):
    """Each program reduces one (n,c) slice of size HW over BLOCK_HW chunks."""
    nc_id  = tl.program_id(0)
    hw_off = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask   = hw_off < HW
    base   = nc_id * HW
    vals   = tl.load(x_ptr + base + hw_off, mask=mask, other=0.0).to(tl.float32)
    total  = tl.sum(vals, axis=0)
    mean_v = total / HW
    tl.store(out_ptr + nc_id, mean_v)


@torch.fx.wrap
def triton_mean_hw(x):
    """Reduces [N, C, H, W] → [N, C, 1, 1] over last two dims."""
    N, C, H, W = x.shape
    HW  = H * W
    N_C = N * C
    out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    grid = lambda meta: (N_C, triton.cdiv(HW, meta['BLOCK_HW']))
    _mean_hw_kernel[grid](x, out, N_C, HW)
    return out


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean_hw