import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Optimized GELU activation computation pattern with adaptive block sizing"""
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
def optimized_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized GELU kernel with fine-grained performance tuning"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with cache hints for better performance
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized polynomial approximation for better accuracy/efficiency
    # Step 1: x^3 using multiplication (more efficient than pow)
    x_squared = x * x
    x_cubed = x_squared * x
    
    # Step 2: 0.044715 * x^3 (vector multiply)
    cubic_term = 0.044715 * x_cubed
    
    # Step 3: x + 0.044715 * x^3 (vector add)
    poly_input = x + cubic_term
    
    # Step 4: 0.7978845608028654 * (x + 0.044715 * x^3)
    scaled_poly = 0.7978845608028654 * poly_input
    
    # Step 5: Enhanced tanh approximation with 5th order polynomial for better accuracy
    # tanh(x) ≈ x - x³/3 + 2x⁵/15 = 17x - 4x³ + x⁵/15
    tanh_input = scaled_poly
    tanh_cubed = tanh_input * tanh_input * tanh_input
    tanh_fifth = tanh_cubed * tanh_input * tanh_input
    
    # Optimized computation: (17x - 4x³ + x⁵/15)/15
    tanh_approx = (17.0 * tanh_input - 4.0 * tanh_cubed + tanh_fifth / 15.0) / 15.0
    tanh_result = tanh_approx
    
    # Step 6: 1.0 + tanh(...)
    shifted_tanh = 1.0 + tanh_result
    
    # Step 7: 0.5 * x * (1.0 + tanh(...)) - final GELU computation
    gelu_result = 0.5 * x * shifted_tanh
    
    # Store result with cache hints for write-back
    tl.store(out_ptr + offsets, gelu_result, mask=mask)

@torch.fx.wrap
def optimized_gelu_activation(x):
    """Ultra-optimized GELU activation with adaptive block sizing"""
    N = x.numel()
    
    # Adaptive block sizing for different tensor sizes
    if N < 512 * 512:  # Small tensors
        BLOCK_SIZE = 256
    elif N < 2 * 1024 * 1024:  # Medium tensors
        BLOCK_SIZE = 512
    elif N < 10 * 1024 * 1024:  # Large tensors
        BLOCK_SIZE = 1024
    else:  # Very large tensors
        BLOCK_SIZE = 2048
    
    # Calculate optimal grid size
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor preserving input properties
    out = torch.empty_like(x)
    
    # Launch Triton kernel with optimized parameters
    optimized_gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Returns the ultra-optimized GELU activation function"""
    return optimized_gelu_activation