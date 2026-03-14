import torch
import triton
import triton.language as tl

# Pattern matching for the final multiplication
def pattern(arithmetic_result, in_2):
    """Match element-wise multiplication between processed features and input_2"""
    final_result = in_2 * arithmetic_result
    return final_result

def replacement_args(arithmetic_result, in_2):
    """Extract arguments for replacement"""
    return (arithmetic_result, in_2)

@triton.jit
def multiplication_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise multiplication kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Element-wise multiplication
    result = x * y
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def elementwise_multiply(x, y):
    """Optimized element-wise multiplication using Triton"""
    n_elements = x.numel()
    
    # Choose BLOCK_SIZE based on typical GPU preferences
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device as inputs
    result = torch.empty_like(x)
    
    # Launch multiplication kernel
    multiplication_kernel[(num_programs,)](
        x,
        y,
        result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result

def replacement_func():
    return elementwise_multiply