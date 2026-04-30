import torch
import triton
import triton.language as tl


# Diagnostic: simplest possible pattern — tensor addition
def pattern(x, y):
    return x + y


def replacement_args(x, y):
    return (x, y)


@triton.jit
def _add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@torch.fx.wrap
def add_fused(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    _add_kernel[(num_blocks,)](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return add_fused