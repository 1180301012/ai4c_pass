import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(input_tensor, weight):
    """Match PReLU activation operation"""
    result = torch.prelu(input_tensor, weight)
    return result

# Argument extraction function
def replacement_args(input_tensor, weight):
    return (input_tensor, weight)

# Optimized PReLU kernel using Triton
@triton.jit
def prelu_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load input
    x = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Load weight (single value for PReLU)
    weight = tl.load(weight_ptr, other=0.01)  # default weight if loading fails
    
    # PReLU: f(x) = x if x >= 0, else weight * x
    # Using select for conditional computation
    out = tl.where(x >= 0, x, weight * x)
    
    # Store result
    tl.store(output_ptr + offset, out, mask=mask)

@torch.fx.wrap
def optimized_prelu(input_tensor, weight):
    """Optimized PReLU activation using Triton kernel"""
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    prelu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference, not called directly)
def replacement_func():
    return optimized_prelu