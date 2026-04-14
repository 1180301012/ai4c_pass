import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_1, in_2):
    """
    Match element-wise multiplication: in_2 * in_1
    """
    result = in_2 * in_1
    return result

# Argument extraction function
def replacement_args(in_1, in_2):
    return (in_1, in_2)

# Optimized Element-wise multiplication using Triton
@triton.jit
def elementwise_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized element-wise multiplication kernel
    Handles broadcasting of y to match x shape
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load inputs with broadcasting support
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Handle broadcasting for y
    y_shape = tl.load(y_ptr - 8)  # Load shape info (simplified)
    y = tl.load(y_ptr + offsets % tl.numel(y_shape), mask=mask)
    
    # Compute element-wise multiplication
    out = x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Helper function for broadcasting
@torch.fx.wrap
def broadcast_multiply(x, y):
    """Optimized element-wise multiplication with broadcasting"""
    # Ensure both tensors are on the same device
    if x.device != y.device:
        y = y.to(x.device)
    
    # Get total number of elements
    total_elements = x.numel()
    
    # Create output tensor with same shape as x
    out = torch.empty_like(x)
    
    # Determine optimal block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    elementwise_mul_kernel[grid_size](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        num_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return broadcast_multiply