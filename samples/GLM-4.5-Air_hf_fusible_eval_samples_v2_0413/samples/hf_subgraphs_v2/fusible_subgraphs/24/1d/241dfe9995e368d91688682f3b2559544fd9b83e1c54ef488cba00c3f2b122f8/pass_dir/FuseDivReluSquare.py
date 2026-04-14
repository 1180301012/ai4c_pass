import torch
import triton
import triton.language as tl

@triton.jit
def fused_div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: divide, relu, square
    x_div = x / scale
    x_relu = tl.maximum(x_div, 0.0)
    x_square = x_relu * x_relu  # More efficient than pow(x_relu, 2)
    
    # Store result
    tl.store(out_ptr + offsets, x_square, mask=mask)

@torch.fx.wrap
def fused_div_relu_square(x, scale):
    # Calculate grid configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with scale as float32 (will be converted in kernel)
    fused_div_relu_square_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        scale=float(scale),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Pattern matching function
def pattern(x):
    """Pattern: division by constant -> ReLU -> square"""
    tmp_0 = x / 11.313708498984761
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = torch.square(tmp_1)
    return tmp_2

# Argument extraction function
def replacement_args(x):
    return (x,)

# Replacement function
def replacement_func():
    return fused_div_relu_square