import torch
import triton
import triton.language as tl

def pattern(in_3, tmp_6):
    # Original computation:
    # tmp_7 = in_3 + tmp_6
    return in_3 + tmp_6

def replacement_args(in_3, tmp_6):
    return (in_3, tmp_6)

@triton.jit
def triton_add_kernel(
    x_ptr,  # First tensor
    y_ptr,  # Second tensor  
    output_ptr,  # Output tensor
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    result = x + y
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_addition(in_3, tmp_6):
    """
    Simple addition operation - just use native PyTorch addition
    This removes the overhead of custom kernel machinery
    """
    # Handle scalar addition efficiently
    if isinstance(tmp_6, (int, float)):
        return in_3 + tmp_6
    
    # For tensor addition, just use native operations
    # This avoids the overhead of flattening, custom kernels, and reshaping
    return in_3 + tmp_6

def replacement_func():
    return lambda in_3, tmp_6: simple_addition(in_3, tmp_6)