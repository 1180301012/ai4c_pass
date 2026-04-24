import torch
import triton
import triton.language as tl


def pattern(x):
    return x.to(dtype=torch.float16)


def replacement_args(x):
    return (x,)


@triton.jit
def cast_f16_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x.to(tl.float16), mask=mask)


@torch.fx.wrap
def triton_cast_to_f16(x):
    out = torch.empty_like(x, dtype=torch.float16)
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    cast_f16_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return triton_cast_to_f16