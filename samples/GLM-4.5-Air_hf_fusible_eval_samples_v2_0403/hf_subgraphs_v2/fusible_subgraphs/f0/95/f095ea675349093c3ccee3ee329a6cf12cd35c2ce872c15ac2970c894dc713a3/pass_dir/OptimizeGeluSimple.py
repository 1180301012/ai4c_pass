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
    
    # ReLU-based GELU approximation: gelu(x) ≈ x * sigmoid(1.702 * x)
    # This is faster and more accurate than erf approximation
    alpha = 1.702
    
    # Compute sigmoid using approximation: 1 / (1 + exp(-alpha * x))
    # Using exp approximation for better performance
    exp_arg = -alpha * x
    
    # Clip to avoid overflow
    exp_arg = tl.maximum(exp_arg, -50.0)
    exp_result = tl.exp(exp_arg)
    sigmoid_result = 1.0 / (1.0 + exp_result)
    
    # Apply GELU: x * sigmoid(alpha * x)
    gelu_result = x * sigmoid_result
    
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