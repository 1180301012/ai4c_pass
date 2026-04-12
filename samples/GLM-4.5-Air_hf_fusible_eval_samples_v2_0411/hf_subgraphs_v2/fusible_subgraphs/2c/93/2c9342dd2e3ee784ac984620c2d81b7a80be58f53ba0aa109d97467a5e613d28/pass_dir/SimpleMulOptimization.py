import torch
import triton
import triton.language as tl

def pattern(tmp_6):
    """Simple pattern: multiply by constant"""
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7

def replacement_args(tmp_6):
    return (tmp_6,)

@triton.jit
def simple_mul_kernel(
    input_ptr,
    output_ptr,
    scalar_value: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for multiplying by constant"""
    # More efficient program indexing
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Vectorized memory access with tl.arange
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input efficiently - using vectorized loads
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Vectorized multiplication with constant
    result = x * scalar_value
    
    # Store result efficiently
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_mul_optimization(tmp_6):
    """Optimized multiplication optimization"""
    n_elements = tmp_6.numel()
    output = torch.empty_like(tmp_6)
    
    # Optimize BLOCK_SIZE based on tensor size for better GPU utilization
    if n_elements < 8192:
        BLOCK_SIZE = 256  # For small tensors
    elif n_elements < 65536:
        BLOCK_SIZE = 512  # For medium tensors
    else:
        BLOCK_SIZE = 1024  # For large tensors
    
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_mul_kernel[(grid_size,)](
        tmp_6,
        output,
        3.141592653589793,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return simple_mul_optimization