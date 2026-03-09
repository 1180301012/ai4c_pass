import torch
import triton
import triton.language as tl
import math

# Pattern matching function - matches the entire computation graph
def pattern(in_0):
    # Match the complete computation chain that produces tmp_7
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return (tmp_7,)  # Return the same structure as the original

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel using Triton with auto-tuning
@triton.jit
def complex_activation_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    coeff1: tl.constexpr,  # 0.5
    coeff2: tl.constexpr,  # 0.044715
    coeff3: tl.constexpr,  # 0.7978845608028654
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Boundary mask
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse all operations into a single computation:
    # out = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))
    
    # Step 1: x^3
    x_cubed = x * x * x
    
    # Step 2: x + 0.044715 * x^3
    inner_term = x + coeff2 * x_cubed
    
    # Step 3: 0.7978845608028654 * inner_term
    scaled_inner = coeff3 * inner_term
    
    # Step 4: tanh(scaled_inner) using exponential identity: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    exp_2x = tl.exp(2.0 * scaled_inner)
    tanh_result = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Step 5: 1.0 + tanh_result
    tanh_plus_one = 1.0 + tanh_result
    
    # Step 6: 0.5 * x * tanh_plus_one
    out = coeff1 * x * tanh_plus_one
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_complex_activation(x):
    """Wrapper function to launch the optimized Triton kernel"""
    n_elements = x.numel()
    
    # Choose optimal block size based on tensor size
    if n_elements >= 100000000:  # Large tensors (134M elements)
        BLOCK_SIZE = 4096
    elif n_elements >= 10000000:  # Medium tensors (33M elements)  
        BLOCK_SIZE = 2048
    else:  # Small tensors (1M-6M elements)
        BLOCK_SIZE = 1024
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with pre-computed coefficients
    grid = (num_programs,)
    complex_activation_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        coeff1=0.5,
        coeff2=0.044715,
        coeff3=0.7978845608028654,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_complex_activation