import torch
import triton
import triton.language as tl

# Simple test pattern: try to match just square operation
def pattern(x):
    return torch.square(x)

# Extract arguments for the replacement function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel for square operation only
@triton.jit
def square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Simple square operation
    squared = x * x
    
    # Store result
    tl.store(out_ptr + offsets, squared, mask=mask)

# Kernel wrapper function (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_square(x):
    # Get tensor properties
    n_elements = x.numel()
    
    # Configure block size for optimal GPU occupancy
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same properties as input
    out = torch.empty_like(x, dtype=x.dtype)
    
    # Launch the square kernel
    square_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function returns the optimized kernel function
def replacement_func():
    return triton_square