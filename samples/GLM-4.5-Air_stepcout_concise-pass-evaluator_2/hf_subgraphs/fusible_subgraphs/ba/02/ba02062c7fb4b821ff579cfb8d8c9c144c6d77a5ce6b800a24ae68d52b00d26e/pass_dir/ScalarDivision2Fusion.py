import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """
    Pattern matching for scalar division + view operation with scalar=2.0
    EXACT match of the target computation from Graph 2:
    tmp_0 = in_0
    tmp_1 = in_1 / 2.0
    tmp_2 = tmp_0.view(-1)
    return tmp_1, tmp_2
    """
    tmp_0 = in_0
    tmp_1 = in_1 / 2.0
    tmp_2 = tmp_0.view(-1)
    return tmp_1, tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def scalar_div_kernel_2(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for scalar division by 2.0
    Uses the fact that division by 2 can be optimized to bit shift
    """
    # Get program ID and compute memory offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform division by 2.0 (can use bit-level optimization)
    out = x * 0.5  # Multiply by 0.5 is often faster than division by 2
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_scalar_division_2(in_0, in_1):
    """
    Wrapper function for optimized scalar division by 2.0
    Handles both division and view operations
    """
    # Optimized scalar division for in_1
    n_elements = in_1.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out_1 = torch.empty_like(in_1)
    
    # Launch Triton kernel
    scalar_div_kernel_2[(num_programs,)](
        x_ptr=in_1,
        output_ptr=out_1,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Handle view operation (this is metadata-only, very efficient)
    out_2 = in_0.view(-1)
    
    return (out_1, out_2)

def replacement_func():
    return optimized_scalar_division_2