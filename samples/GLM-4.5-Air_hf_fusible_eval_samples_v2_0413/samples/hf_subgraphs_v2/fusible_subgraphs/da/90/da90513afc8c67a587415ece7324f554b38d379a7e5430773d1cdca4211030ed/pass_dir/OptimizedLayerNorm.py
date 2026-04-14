import torch
import triton
import triton.language as tl

@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for element-wise addition"""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Calculate
    out = x + y
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

# Pattern matching function - must mirror model.py exactly
def pattern(x, y):
    """Pattern matches addition operation that can be optimized"""
    return x + y

# Argument extraction function
def replacement_args(x, y):
    """Extract arguments needed for the replacement"""
    return (x, y)

@triton.jit
def triton_add_kernel_optimized(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for element-wise addition with better performance"""
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with vectorized access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    """Optimized addition using Triton with autotuning"""
    x, y = x, y
    N = x.numel()
    
    # Use smaller block size for better GPU occupancy
    BLOCK_SIZE = 256
    if N > 1000000:  # Large tensors
        BLOCK_SIZE = 512
    elif N < 10000:   # Small tensors
        BLOCK_SIZE = 128
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)

    triton_add_kernel_optimized[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

# Replacement function (returns function reference, not call)
def replacement_func():
    return triton_add