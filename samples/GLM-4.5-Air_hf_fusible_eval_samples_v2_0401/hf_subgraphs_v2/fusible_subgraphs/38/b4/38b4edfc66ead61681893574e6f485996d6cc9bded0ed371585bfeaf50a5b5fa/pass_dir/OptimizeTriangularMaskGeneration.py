import torch
import triton
import triton.language as tl

# Simple pattern based on reference implementation
def pattern(x):
    return x + 1

def replacement_args(x):
    return (x,)

@triton.jit
def triton_add_one_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements # Mask to ensure we don't go out of bounds
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Calculate: add 1.0 to each element
    out = x + 1.0
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add_one(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    triton_add_one_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return triton_add_one