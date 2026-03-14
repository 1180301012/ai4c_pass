import torch
import triton
import triton.language as tl

def pattern(in_2, in_3):
    return in_2 + in_3

def replacement_args(in_2, in_3):
    return (in_2, in_3)

@triton.jit
def optimized_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load with vectorized memory access for better performance
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Vectorized addition
    out = x + y
    
    # Store with vectorized memory access
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_triton_add(in_2, in_3):
    """Optimized Triton addition with block size selection"""
    if in_2.shape != in_3.shape:
        # Fallback to regular addition if shapes don't match
        return in_2 + in_3
    
    n_elements = in_2.numel()
    
    # Choose optimal block size based on tensor size
    if n_elements >= 16384:  # Larger tensors
        BLOCK_SIZE = 4096
    elif n_elements >= 4096:  # Medium tensors
        BLOCK_SIZE = 2048
    else:  # Small tensors
        BLOCK_SIZE = 1024
    
    # Adjust BLOCK_SIZE if it's larger than n_elements
    if BLOCK_SIZE > n_elements:
        BLOCK_SIZE = n_elements
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_2)
    
    # Launch optimized kernel
    optimized_add_kernel[(num_programs,)](
        in_2, in_3, out, n_elements, BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_triton_add