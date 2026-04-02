import torch
import math
import triton
import triton.language as tl

# Pattern matching function for the complete activation computation
def pattern(in_0):
    """Match the complete Swish-like activation computation pattern"""
    # All operations from the original computation
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return (tmp_7,)

# Argument extraction function
def replacement_args(in_0):
    """Extract arguments needed for the replacement"""
    return (in_0,)

# Optimized fused kernel using Triton
@triton.jit
def fused_activation_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that implements the complete activation function in one pass"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Constants from the original computation
    a = 0.5
    b = 0.044715
    c = 0.7978845608028654
    
    # Fused computation:
    # 1. tmp_0 = a * x
    # 2. tmp_1 = x^3
    # 3. tmp_2 = b * tmp_1
    # 4. tmp_3 = x + tmp_2
    # 5. tmp_4 = c * tmp_3
    # 6. tmp_5 = tanh(tmp_4)
    # 7. tmp_6 = 1.0 + tmp_5
    # 8. tmp_7 = tmp_0 * tmp_6
    
    x_cubed = x * x * x
    scaled_cube = b * x_cubed
    inner_term = x + scaled_cube
    scaled_inner = c * inner_term
    # Custom tanh approximation using polynomial and basic operations
    # Using: tanh(x) ≈ x * (1.0 - abs(x) * (1.0/3.0 + abs(x) * abs(x) * (1.0/15.0)))
    safe_x = tl.where(scaled_inner > 1.0, 1.0, 
                      tl.where(scaled_inner < -1.0, -1.0, scaled_inner))
    x_abs = tl.abs(safe_x)
    tanh_result = safe_x * (1.0 - x_abs * (1.0/3.0 + x_abs * x_abs * (1.0/15.0)))
    sigmoid_like = 1.0 + tanh_result
    result = (a * x) * sigmoid_like
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_activation(x):
    """Kernel wrapper for the fused activation function"""
    n_elements = x.numel()
    # Optimized block size for the given tensor shapes [2, 1024, 3072]
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same properties as input
    out = torch.empty_like(x, dtype=x.dtype, device=x.device)
    
    # Launch the fused kernel
    fused_activation_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns callable)
def replacement_func():
    return fused_activation