import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    x_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply sigmoid directly using Triton operations
    # Using a more numerically stable sigmoid computation
    neg_x = tl.minimum(x, 0.0)  # Only compute exp for negative values
    exp_neg_x = tl.exp(neg_x)
    result = tl.where(x >= 0, 1 / (1 + tl.exp(-x)), exp_neg_x / (1 + exp_neg_x))
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def custom_sigmoid(x):
    n_elements = x.numel()
    
    # Use optimized block size for better performance
    BLOCK_SIZE = 2048  # Larger block size for better GPU occupancy
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(x):
    """Simple sigmoid pattern"""
    result = x.sigmoid()
    return result

def replacement_args(x):
    return (x,)

def replacement_func():
    return custom_sigmoid