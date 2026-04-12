import torch
import triton
import triton.language as tl

def pattern(tmp_5):
    """Fuse the final activation operations: subtract 0.25 → multiply by pi"""
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7

def replacement_args(tmp_5):
    return (tmp_5,)

@triton.jit
def fused_sub_mul_kernel(
    input_ptr,
    output_ptr,
    pi_value: tl.constexpr,
    sub_value: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for (x - 0.25) * pi"""
    # Efficient program indexing
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Vectorized memory access
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations: (x - 0.25) * pi
    result = (x - sub_value) * pi_value
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_sub_mul_optimization(tmp_5):
    """Optimized fusion of subtraction and multiplication"""
    n_elements = tmp_5.numel()
    output = torch.empty_like(tmp_5)
    
    # Use BLOCK_SIZE based on tensor size
    if n_elements < 32768:
        BLOCK_SIZE = 512  # For smaller tensors
    elif n_elements < 131072:
        BLOCK_SIZE = 1024  # For medium tensors  
    else:
        BLOCK_SIZE = 2048  # For large tensors
    
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_sub_mul_kernel[(grid_size,)](
        tmp_5,
        output,
        3.141592653589793,
        0.25,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_sub_mul_optimization