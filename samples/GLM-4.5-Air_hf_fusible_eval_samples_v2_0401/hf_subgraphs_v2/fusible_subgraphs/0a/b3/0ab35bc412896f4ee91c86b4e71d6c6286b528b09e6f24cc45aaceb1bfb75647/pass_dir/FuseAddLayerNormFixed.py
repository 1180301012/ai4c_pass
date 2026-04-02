import torch
import triton
import triton.language as tl

def pattern(in_3, in_2):
    """Pattern: element-wise addition - focusing on correctness"""
    tmp_2 = in_3 + in_2
    return tmp_2

def replacement_args(in_3, in_2):
    return (in_3, in_2)

@triton.jit
def efficient_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Highly efficient element-wise addition kernel"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data efficiently
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    result = x + y
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def efficient_add(x, y):
    """Efficient addition using optimal Triton configuration"""
    N = x.numel()
    
    # Use optimal block size for this tensor size
    if N >= 2048:  # Use larger blocks for efficiency
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 1024
        
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    result = torch.empty_like(x)
    
    # Launch kernel
    efficient_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=result,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result

def replacement_func():
    return efficient_add