"""
Pass: Replace spatial mean over (H, W) dims with an optimized Triton kernel.
Matches: x.mean((2, 3), keepdim=True) for any 4-D NCHW tensor.
Single-output replacement — no multi-output subgraph issues.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _spatial_mean_kernel(
    x_ptr, out_ptr,
    NC, HW,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per (n, c) pair. Accumulates in float32 for precision."""
    pid = tl.program_id(0)
    base = pid * HW
    acc = 0.0

    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)

    mean_val = acc / HW
    tl.store(out_ptr + pid, mean_val)


@torch.fx.wrap
def triton_spatial_mean(x):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C
    out_f32 = torch.empty((N, C, 1, 1), dtype=torch.float32, device=x.device)
    _spatial_mean_kernel[(NC,)](x, out_f32, NC, HW)
    if x.dtype == torch.float32:
        return out_f32
    return out_f32.to(x.dtype)


def pattern(x):
    out = x.mean((2, 3), keepdim=True)
    return out


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_spatial_mean