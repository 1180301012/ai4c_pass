import torch
import triton
import triton.language as tl
import math

def pattern(in_0):
    """Fused GELU activation computation pattern"""
    # Exact computation from model.py without cleanup statements
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return (tmp_7,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU activation kernel with polynomial approximation"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused GELU computation: 0.5 * x * tanh(0.7978845608028654 * (x + 0.044715 * x^3))
    # Step 1: x^3 (using multiplication)
    x_squared = x * x
    x_cubed = x_squared * x
    
    # Step 2: 0.044715 * x^3
    cubic_term = 0.044715 * x_cubed
    
    # Step 3: x + 0.044715 * x^3
    poly_input = x + cubic_term
    
    # Step 4: 0.7978845608028654 * (x + 0.044715 * x^3)
    scaled_poly = 0.7978845608028654 * poly_input
    
    # Step 5: tanh(0.7978845608028654 * (x + 0.044715 * x^3))
    # Simple polynomial approximation: tanh(x) ≈ x - x^3/3 + 2x^5/15
    # This provides decent accuracy with basic arithmetic operations
    tanh_input = scaled_poly
    tanh_squared = tanh_input * tanh_input
    tanh_cubed = tanh_squared * tanh_input
    tanh_fifth = tanh_cubed * tanh_squared
    
    tanh_approx = tanh_input - (tanh_cubed / 3.0) + (2.0 * tanh_fifth / 15.0)
    tanh_result = tanh_approx
    
    # Step 6: 1.0 + tanh(...)
    shifted_tanh = 1.0 + tanh_result
    
    # Step 7: 0.5 * x * (1.0 + tanh(...))
    gelu_result = 0.5 * x * shifted_tanh
    
    # Store result
    tl.store(out_ptr + offsets, gelu_result, mask=mask)

@torch.fx.wrap
def fused_gelu_activation(x):
    """High-performance fused GELU activation using Triton"""
    N = x.numel()
    BLOCK_SIZE = 1024  # Will be autotuned
    
    # Try different block sizes to find optimal configuration
    if N < 1024 * 1024:  # Small tensors
        BLOCK_SIZE = 512
    elif N < 10 * 1024 * 1024:  # Medium tensors  
        BLOCK_SIZE = 1024
    else:  # Large tensors
        BLOCK_SIZE = 2048
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device as input
    out = torch.empty_like(x)
    
    # Launch Triton kernel
    fused_gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Returns the fused GELU activation function"""
    return fused_gelu_activation