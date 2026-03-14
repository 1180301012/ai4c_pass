import torch
import triton
import triton.language as tl

# Pattern matching function - matches both cosine and sine operations
def pattern(input_tensor):
    # Match both cosine and sine operations on the same input
    cos_result = input_tensor.cos()
    sin_result = input_tensor.sin()
    return (cos_result, sin_result)

# Argument extraction function  
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized kernel for computing both cosine and sine together
@triton.jit
def fused_trig_kernel(
    input_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input once and compute both trig functions
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    cos_vals = tl.cos(x)
    sin_vals = tl.sin(x)
    
    # Store both results
    tl.store(cos_out_ptr + offsets, cos_vals, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_vals, mask=mask)

@torch.fx.wrap
def optimized_trig_pair(input_tensor):
    """Optimized computation of both cosine and sine"""
    n_elements = input_tensor.numel()
    
    # Create output tensors
    cos_out = torch.empty_like(input_tensor)
    sin_out = torch.empty_like(input_tensor)
    
    # Launch kernel with shared computation
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_trig_kernel[(num_programs,)](
        input_ptr=input_tensor,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_out, sin_out

# Replacement function (must return a function reference)
def replacement_func():
    return optimized_trig_pair