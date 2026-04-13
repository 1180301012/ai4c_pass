import torch
import triton
import triton.language as tl

# Pattern matching function for addition (same as before)
def pattern(in_0, in_1):
    """Simple addition pattern for larger tensors"""
    tmp_1 = in_0 + in_1
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel specifically for larger tensors with better performance
@triton.jit
def large_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with better memory pattern
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def large_add(in_0, in_1):
    """Optimized addition for larger tensors"""
    n_elements = in_0.numel()
    
    # Use optimal block size for larger tensors
    if n_elements >= 50000:
        # Very large tensors - use large block size for better occupancy
        BLOCK_SIZE = 4096
    elif n_elements >= 10000:
        # Medium-large tensors - use 2048
        BLOCK_SIZE = 2048
    else:
        # For smaller tensors, fall back to standard size
        BLOCK_SIZE = 1024
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    large_add_kernel[(num_programs,)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return large_add