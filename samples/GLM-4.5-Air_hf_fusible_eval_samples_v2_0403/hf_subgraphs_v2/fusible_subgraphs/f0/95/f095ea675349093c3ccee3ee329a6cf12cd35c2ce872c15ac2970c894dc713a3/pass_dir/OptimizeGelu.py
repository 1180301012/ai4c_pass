import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern to match GELU activation"""
    return torch.nn.functional.gelu(x)

def replacement_args(x):
    return (x,)

@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized GELU kernel using Triton"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized GELU computation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using approximations for better performance
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    gelu_constant = 0.044715
    
    # Compute GELU using the optimized formula
    x_cubed = x * x * x
    inner = x + gelu_constant * x_cubed
    tanh_arg = sqrt_2_over_pi * inner
    tanh_result = tl.tanh(tanh_arg)
    gelu_result = 0.5 * x * (1.0 + tanh_result)
    
    # Store result
    tl.store(out_ptr + offsets, gelu_result, mask=mask)

@torch.fx.wrap
def triton_gelu(x):
    """Optimized GELU using Triton kernel"""
    n_elements = x.numel()
    
    # Choose block size for optimal GPU occupancy
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    gelu_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_gelu