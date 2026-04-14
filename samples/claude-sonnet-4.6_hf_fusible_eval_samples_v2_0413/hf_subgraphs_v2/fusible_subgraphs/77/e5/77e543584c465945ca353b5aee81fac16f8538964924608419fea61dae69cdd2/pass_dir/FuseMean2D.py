"""
Single-output pass: replace x.mean((2, 3), keepdim=True) with a Triton
reduction kernel.  Pattern has exactly ONE observable output (the mean),
which avoids the multi-output replacement issue entirely.

This single pass matches ALL test graphs regardless of conv stride/groups/dtype.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 256},   num_warps=4),
        triton.Config({"BLOCK_HW": 512},   num_warps=8),
        triton.Config({"BLOCK_HW": 1024},  num_warps=16),
        triton.Config({"BLOCK_HW": 2048},  num_warps=16),
        triton.Config({"BLOCK_HW": 4096},  num_warps=16),
        triton.Config({"BLOCK_HW": 8192},  num_warps=16),
        triton.Config({"BLOCK_HW": 16384}, num_warps=16),
    ],
    key=["HW"],
)
@triton.jit
def _mean_hw_kernel(
    x_ptr,
    out_ptr,
    HW,
    DTYPE: tl.constexpr,      # 0=fp16, 1=bf16, 2=fp32
    BLOCK_HW: tl.constexpr,
):
    """
    One program per (N, C) row.
    Reduces H*W elements to a single mean value.
    With BLOCK_HW >= HW the inner loop has a single iteration (no loop overhead).
    No tl.where needed: tl.load uses other=0.0 so masked elements contribute 0.
    """
    row  = tl.program_id(0)
    base = row * HW
    acc  = 0.0

    for s in range(0, HW, BLOCK_HW):
        idx  = s + tl.arange(0, BLOCK_HW)
        mask = idx < HW
        x    = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        acc  = acc + tl.sum(x)   # masked elements are 0.0, no tl.where needed

    mean = acc / HW
    if DTYPE == 0:
        tl.store(out_ptr + row, mean.to(tl.float16))
    elif DTYPE == 1:
        tl.store(out_ptr + row, mean.to(tl.bfloat16))
    else:
        tl.store(out_ptr + row, mean.to(tl.float32))


@torch.fx.wrap
def triton_mean_hw_keepdim(x):
    """
    Computes x.mean((2, 3), keepdim=True) via a Triton reduction kernel.
    x: [N, C, H, W]  →  returns [N, C, 1, 1]
    """
    N, C, H, W = x.shape
    NC    = N * C
    HW    = H * W
    DTYPE = 0 if x.dtype == torch.float16 else (1 if x.dtype == torch.bfloat16 else 2)

    out_flat = torch.empty((NC,), dtype=x.dtype, device=x.device)
    _mean_hw_kernel[(NC,)](x.contiguous(), out_flat, HW, DTYPE)

    return out_flat.view(N, C, 1, 1)


# ---------------------------------------------------------------------------
# Pattern, replacement_args, replacement_func
# ---------------------------------------------------------------------------

def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean_hw_keepdim