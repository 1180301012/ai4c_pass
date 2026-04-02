import torch
import triton
import triton.language as tl
import functools


# Match ONLY the divide — the transpose stays in the graph as a free view.
def pattern(in_0):
    tmp_0 = in_0 / 2.8284271247461903
    return tmp_0


def replacement_args(in_0):
    return (in_0,)


_GRID = 256
_BLOCK = 256


@triton.jit
def _persistent_scale_2_83(
    in_ptr, out_ptr, n_elements, scale,
    GRID_SIZE: tl.constexpr,
    BLOCK:     tl.constexpr,
):
    pid  = tl.program_id(0)
    step = GRID_SIZE * BLOCK
    for start in range(pid * BLOCK, n_elements, step):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(in_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, x * scale, mask=mask)


@torch.fx.wrap
def triton_scale_2_8284271247461903(in_0):
    N   = in_0.numel()
    out = torch.empty_like(in_0)
    _persistent_scale_2_83[(_GRID,)](
        in_0, out, N, 1.0 / 2.8284271247461903,
        GRID_SIZE=_GRID, BLOCK=_BLOCK,
        num_warps=8,
    )
    return out


def replacement_func():
    return triton_scale_2_8284271247461903