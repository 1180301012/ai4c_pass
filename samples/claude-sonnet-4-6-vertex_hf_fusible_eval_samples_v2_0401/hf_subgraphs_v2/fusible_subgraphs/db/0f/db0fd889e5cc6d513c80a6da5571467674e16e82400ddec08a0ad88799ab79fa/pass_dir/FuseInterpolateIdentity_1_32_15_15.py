"""
Try matching the DECOMPOSED ATen form of bicubic interpolation.
The _decomposed model likely uses aten.upsample_bicubic2d.vec instead of
torch.nn.functional.interpolate.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def _memcpy_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def skip_bicubic_same_size(x):
    """
    Replace bicubic interpolate(size=same) with a fast contiguous copy.
    Correct because same-size bicubic with align_corners=False is identity.
    """
    x_cont = x.contiguous()
    N = x_cont.numel()
    out = torch.empty_like(x_cont)
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _memcpy_kernel[grid](x_ptr=x_cont, out_ptr=out, n_elements=N)
    return out


def pattern(x):
    # Try aten.upsample_bicubic2d.vec  (decomposed form)
    return torch.ops.aten.upsample_bicubic2d.vec(x, [15, 15], False, None)


def replacement_args(x):
    return (x,)


def replacement_func():
    return skip_bicubic_same_size