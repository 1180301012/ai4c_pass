import torch
import triton
import triton.language as tl

# Simple pattern matching function - matches just the cosine operation
def pattern(input_tensor):
    # Simple cosine operation
    result = input_tensor.cos()
    return result

# Argument extraction function  
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized kernel for cosine
@triton.jit
def simple_cosine_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and compute cosine
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    cos_vals = tl.cos(x)
    
    # Store result
    tl.store(output_ptr + offsets, cos_vals, mask=mask)

@torch.fx.wrap
def optimized_cosine(input_tensor):
    """Optimized cosine computation"""
    n_elements = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_cosine_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (must return a function reference)
def replacement_func():
    return optimized_cosine