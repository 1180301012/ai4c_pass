import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern for scalar division operation"""
    return x / 8.0

def replacement_args(x):
    return (x,)

@triton.jit
def triton_div_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for division by 8.0"""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input with mask
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Division by 8.0 using multiplication for better performance
    out = x * 0.125
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_div(x):
    """Triton division by 8.0 wrapper function"""
    total_elements = x.numel()
    
    # Optimized block size for the given tensor shapes
    # [2, 12, 7, 7] = 1176 elements total
    if total_elements <= 2048:
        BLOCK_SIZE = 256  # Smaller block size for small tensors
    else:
        BLOCK_SIZE = 1024  # Default for larger tensors
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device as inputs
    out = torch.empty_like(x)
    
    # Launch the Triton kernel
    triton_div_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_div