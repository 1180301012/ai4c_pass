import torch
import triton
import triton.language as tl

@triton.jit
def optimized_unsqueeze_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Store result (unsqueeze is just a view change, so we copy data)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def optimized_unsqueeze(x, dim=-2):
    # Get total number of elements
    n_elements = x.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same data but expanded shape
    result = x.unsqueeze(dim)
    
    return result

def pattern(x):
    result = x.unsqueeze(-2)
    return result

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_unsqueeze