import torch
import triton
import triton.language as tl
import math

# Pattern matching function - only optimizes element-wise operations (multiply + add)
def pattern(in_0, in_1, in_2):
    """Match the element-wise computation pattern: multiply + add"""
    # Element-wise multiplication followed by addition
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    # Return just the fused result - the unbind/permute will be handled separately
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Simple and efficient Triton kernel for fused operations
@triton.jit
def simple_fused_kernel(
    out_ptr, in0_ptr, in1_ptr, in2_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Simple, direct memory access
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    in2 = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: (in2 * in1) + in0
    result = (in2 * in1) + in0
    
    tl.store(out_ptr + offsets, result, mask=mask)

# Optimized wrapper function with minimal overhead
@torch.fx.wrap  
def optimized_fused_op(in_0, in_1, in_2):
    """High-performance fused implementation with minimal overhead"""
    
    # Use simple, direct broadcasting that matches PyTorch's pattern
    # Create tensor with same shape as in_1 by expanding in_0 directly
    in_0_broadcast = in_0.expand_as(in_1)
    
    # Use optimized kernel with minimal overhead
    n_elements = in_1.numel()
    
    # Simple launch configuration - using optimal block size for this GPU
    BLOCK_SIZE = 512  # Smaller block size for better occupancy
    
    # Calculate grid size
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with minimal overhead
    output = torch.empty_like(in_1)
    simple_fused_kernel[grid_size](
        output, in_0_broadcast, in_1, in_2, n_elements, BLOCK_SIZE
    )
    
    return output





# Replacement function - returns the optimized kernel
def replacement_func():
    return optimized_fused_op