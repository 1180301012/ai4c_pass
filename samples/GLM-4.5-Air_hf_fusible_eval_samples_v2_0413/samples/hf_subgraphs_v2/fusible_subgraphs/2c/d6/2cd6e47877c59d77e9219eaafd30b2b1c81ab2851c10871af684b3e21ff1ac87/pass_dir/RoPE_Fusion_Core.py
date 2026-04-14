import torch
import triton
import triton.language as tl

def pattern(x, y):
    """
    Simple pattern to test pass loading and matching
    """
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def simple_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute and store result
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_add(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    simple_add_kernel[(num_programs,)](
        x,
        y,
        out,
        N,
        BLOCK_SIZE,
    )

    return out

def replacement_func():
    return simple_add