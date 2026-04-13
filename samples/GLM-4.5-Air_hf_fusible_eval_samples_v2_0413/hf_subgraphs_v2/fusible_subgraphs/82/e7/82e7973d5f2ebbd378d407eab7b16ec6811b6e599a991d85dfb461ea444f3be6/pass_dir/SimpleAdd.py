import torch
import triton
import triton.language as tl

# Simple pattern matching function for addition
def pattern(in_0, in_1):
    """Simple addition pattern"""
    tmp_1 = in_0 + in_1
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel with better memory coalescing
@triton.jit
def simple_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized for memory coalescing and parallel execution
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Use aligned memory access patterns for better performance
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Simple addition (hardware prefers this)
    out = x + y
    
    # Store result with same alignment
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_add(in_0, in_1):
    N = in_0.numel()
    
    # Use block size based on tensor size for better performance
    if N < 1024:
        BLOCK_SIZE = 256
    elif N < 10000:
        BLOCK_SIZE = 512
    elif N < 100000:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    simple_add_kernel[(num_programs,)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return simple_add