import torch
import triton
import triton.language as tl

# Simple pattern to match subtraction operation
def pattern(x, y):
    """
    Simple pattern matching subtraction operation 
    """
    result = x - y
    return result

# Argument extraction for replacement
def replacement_args(x, y):
    return (x, y)

@triton.jit
def simple_sub_kernel(
    x_ptr,
    y_ptr, 
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized subtraction kernel with better memory access"""
    # Calculate program coordinates
    pid = tl.program_id(0)
    
    # Each program handles a contiguous block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both tensors with same mask for coalescing
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform subtraction
    out = x - y
    
    # Store result with mask to prevent out-of-bounds writes
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_sub_wrapper(x, y):
    """Optimized subtraction wrapper with autotuning based on tensor properties"""
    N = x.numel()
    
    # Adaptive block size selection based on tensor size
    if N < 4096:
        BLOCK_SIZE = 256
    elif N < 65536:
        BLOCK_SIZE = 512  
    elif N < 262144:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    simple_sub_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns kernel wrapper)  
def replacement_func():
    return simple_sub_wrapper