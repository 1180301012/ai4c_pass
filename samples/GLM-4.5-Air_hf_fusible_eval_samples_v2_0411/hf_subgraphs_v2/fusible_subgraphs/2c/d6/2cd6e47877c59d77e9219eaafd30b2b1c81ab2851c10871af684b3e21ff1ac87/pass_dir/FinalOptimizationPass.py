import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Simple multiplication pattern - very basic to ensure matching"""
    return x * y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def simple_mult_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple triton kernel following the reference implementation exactly"""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load data directly (no stride complications for this simple case)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate multiplication
    out = x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_mult_optimized(x, y):
    """Simple optimized multiplication using Triton - following reference exactly"""
    # Use the exact same approach as the reference
    N = x.numel()
    BLOCK_SIZE = 1024  # Use same block size as reference
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create output with same dtype and device as input
    out = torch.empty_like(x)

    # Use the exact same kernel call as reference
    simple_mult_kernel[(num_programs,)](
        x,
        y,
        out,
        N,
        BLOCK_SIZE,
    )

    return out

def replacement_func():
    return simple_mult_optimized