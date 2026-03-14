import torch
import triton
import triton.language as tl

# Pattern matching function for scalar division
def pattern(in_1, divisor):
    """Match the computation pattern: in_1 / divisor
    This matches division by any scalar constant
    """
    result = in_1 / divisor
    return result

# Argument extraction function  
def replacement_args(in_1, divisor):
    """Extract arguments needed for the replacement"""
    return (in_1, divisor)

@triton.jit
def divide_by_constant_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    divisor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance kernel for element-wise division by constant"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Perform division by constant
    out = x / divisor
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def divide_by_constant_optimized(in_1, divisor):
    """Optimized function that replaces the division operation"""
    N = in_1.numel()
    BLOCK_SIZE = 256  # Smaller block size for better GPU utilization
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device as input
    out = torch.empty_like(in_1)
    
    # Launch the kernel with optimized grid configuration
    divide_by_constant_kernel[(num_programs,)](
        input_ptr=in_1,
        output_ptr=out, 
        n_elements=N,
        divisor=divisor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (must return function reference, not call it)
def replacement_func():
    return divide_by_constant_optimized