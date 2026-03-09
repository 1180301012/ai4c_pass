import torch
import triton
import triton.language as tl


# Pattern matching function - simplest pattern: just x + y
def pattern(in_0, in_1):
    return in_0 + in_1


# Extract arguments needed for replacement
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Triton kernel for element-wise add with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=1),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def add_wrapper(x, y):
    n = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    add_kernel[grid](x, y, out, n)
    return out


def replacement_func():
    return add_wrapper