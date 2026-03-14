import torch
import triton
import triton.language as tl
from typing import Any

def pattern(x):
    """Matches the sequence: scalar_mult -> sigmoid -> scalar_mult"""
    # Must match exactly as in model.py:
    tmp_0 = 1.702 * x
    tmp_1 = torch.sigmoid(tmp_0)
    tmp_2 = x * tmp_1
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def fused_kernel_1_702_sigmoid_mult(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Fused kernel for: output = x * sigmoid(1.702 * x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: x * sigmoid(1.702 * x)
    scaled_x = 1.702 * x
    sigmoid_scaled_x = tl.sigmoid(scaled_x)
    result = x * sigmoid_scaled_x
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_computation(x):
    """Wrapper for the optimized fused Triton kernel"""
    n_elements = x.numel()
    
    # Optimized block size for element-wise operations
    # Smaller blocks are typically better for element-wise operations due to better occupancy
    if n_elements > 1000000:
        BLOCK_SIZE = 1024  # Very large tensors
        num_warps = 4
    elif n_elements > 200000:
        BLOCK_SIZE = 512   # Large tensors
        num_warps = 4
    elif n_elements > 50000:
        BLOCK_SIZE = 256   # Medium tensors  
        num_warps = 2
    else:
        BLOCK_SIZE = 128   # Small tensors
        num_warps = 1
    
    # For the specific input size [1, 197, 3072] = 605,184 elements
    if n_elements == 605184:
        # Tuned specifically for this tensor size - use power of 2
        BLOCK_SIZE = 512   # Power of 2, 605184/512 = 1182 programs
        num_warps = 4
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Launch kernel with optimized parameters
    fused_kernel_1_702_sigmoid_mult[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return out

def replacement_func():
    """Returns the fused computation function"""
    return fused_computation