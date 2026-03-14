import torch
import triton
import triton.language as tl

# Pattern matching function - must match exactly the computation in model.py
def pattern(in_0, in_1):
    """
    Match the SiLU operation from the computation graph
    """
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    # Pattern must include observables that appear in the model's return
    return tmp_0, tmp_1

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized SiLU kernel using Triton
@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Optimized SiLU kernel: y = x * sigmoid(x)
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute SiLU: x * sigmoid(x)
    # Use fast sigmoid approximation for better performance
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid_x
    
    # Store output
    tl.store(y_ptr + offsets, y, mask=mask)

# Kernel wrapper with different optimization strategies for different input shapes
@torch.fx.wrap
def optimized_silu(x):
    """
    Optimized SiLU implementation with shape-specific optimizations
    """
    # Handle different input shapes efficiently
    if x.ndim == 3:
        # For shape [B, H, W] - our primary case
        B, H, W = x.shape
        n_elements = B * H * W
        
        # Optimize block size based on tensor size
        if n_elements > 1000000:
            BLOCK_SIZE = 1024
        elif n_elements > 100000:
            BLOCK_SIZE = 512
        else:
            BLOCK_SIZE = 256
    else:
        # Fallback for other shapes
        n_elements = x.numel()
        BLOCK_SIZE = min(1024, (n_elements + 1023) // 1024 * 1024)
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Launch optimized kernel
    silu_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y

# Replacement function - must return a callable function
def replacement_func():
    return optimized_silu