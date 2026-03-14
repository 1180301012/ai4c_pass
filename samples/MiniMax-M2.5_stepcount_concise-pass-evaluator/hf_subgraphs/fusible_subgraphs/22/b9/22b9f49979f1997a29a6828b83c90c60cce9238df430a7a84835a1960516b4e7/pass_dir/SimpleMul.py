import torch
import triton
import triton.language as tl


def pattern(a, b):
    """Match just multiply operation."""
    return a * b


def replacement_args(a, b):
    return (a, b)


# Simple Triton kernel without autotuning to avoid conflicts
@triton.jit
def mul_kernel(
    a_ptr, b_ptr, out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a * b
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def mul_kernel_wrapper(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 4096
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(a)
    mul_kernel[(num_programs,)](
        a_ptr=a.flatten(),
        b_ptr=b.flatten(),
        out_ptr=out.flatten(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return mul_kernel_wrapper