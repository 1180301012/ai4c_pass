import torch
import triton
import triton.language as tl
import functools


# Match ONLY the divide — the transpose stays in the graph as a free view.
# This avoids a physical memory transpose (we don't need to materialise the layout swap).
def pattern(in_0):
    tmp_0 = in_0 / 1.6817928305074292
    return tmp_0


def replacement_args(in_0):
    return (in_0,)


# Persistent-grid scale kernel:
# GRID_SIZE fixed blocks × 8 warps (256 threads) gives ~59 % SM occupancy,
# matching PyTorch's own element-wise CUDA kernels.
# For small tensors (N < GRID_SIZE * BLOCK), excess blocks check the loop bound
# and exit immediately — no wasted computation, just better latency hiding.
_GRID = 256
_BLOCK = 256


@triton.jit
def _persistent_scale_1_68(
    in_ptr, out_ptr, n_elements, scale,
    GRID_SIZE: tl.constexpr,
    BLOCK:     tl.constexpr,
):
    pid  = tl.program_id(0)
    step = GRID_SIZE * BLOCK
    # Each program strides across the input in chunks of BLOCK
    for start in range(pid * BLOCK, n_elements, step):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(in_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, x * scale, mask=mask)


@torch.fx.wrap
def triton_scale_1_6817928305074292(in_0):
    N   = in_0.numel()
    out = torch.empty_like(in_0)
    _persistent_scale_1_68[(_GRID,)](
        in_0, out, N, 1.0 / 1.6817928305074292,
        GRID_SIZE=_GRID, BLOCK=_BLOCK,
        num_warps=8,   # 256 threads per block → ~37 active warps/SM on A30
    )
    return out


def replacement_func():
    return triton_scale_1_6817928305074292