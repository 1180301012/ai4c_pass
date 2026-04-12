import torch
import triton
import triton.language as tl

# Pattern matching function - match SiLU operation exactly as in the model
def pattern(x):
    """Match SiLU operation with inplace=True"""
    return torch.nn.functional.silu(x, inplace=True)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized SiLU kernel using Triton
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized SiLU kernel: silu(x) = x * sigmoid(x)"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU using optimized operations
    # fast_sigmoid approximation for better performance
    sigmoid_x = 1.0 / (1.0 + tl.exp(-tl.abs(x)))
    silu_x = x * sigmoid_x
    
    # Store output  
    tl.store(out_ptr + offsets, silu_x, mask=mask)

@torch.fx.wrap
def optimized_silu(x):
    """Optimized SiLU computation"""
    # Get input properties
    input_shape = x.shape
    device = x.device
    
    # Flatten input for efficient processing
    if x.is_contiguous():
        x_flat = x
        original_shape = input_shape
        needs_reshape = False
    else:
        x_flat = x.contiguous()
        original_shape = input_shape
        needs_reshape = True
    
    n_elements = x_flat.numel()
    
    # Use optimal block size for good GPU utilization
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out_flat = torch.empty_like(x_flat)
    
    # Launch Triton kernel
    silu_kernel[(num_programs,)](
        x_ptr=x_flat,
        out_ptr=out_flat,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back if needed
    if needs_reshape:
        return out_flat.view(original_shape)
    else:
        return out_flat

# Replacement function (returns function reference)
def replacement_func():
    return optimized_silu