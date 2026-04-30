import torch
import triton
import triton.language as tl


def pattern(x, y):
    tmp = x + y
    out = torch.nn.functional.dropout2d(tmp, 0.1, False, False)
    return out


def replacement_args(x, y):
    return (x, y)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@torch.fx.wrap
def fused_add_dropout(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


def replacement_func():
    return fused_add_dropout