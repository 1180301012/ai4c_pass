import torch
import triton
import triton.language as tl

# Pattern matching function - simpler approach matching just norm + clamp
def pattern(x):
    """
    Simple pattern matching for norm + clamp operations
    """
    # L2 normalization
    norm_result = torch.functional.norm(x, dim=-1, keepdim=True)
    
    # Scale by constant and clamp
    scaled = norm_result * 0.14433756729740643
    clamped = scaled.clamp(min=1e-05)
    
    return clamped

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel for the core normalization and clamping
@triton.jit
def norm_clamp_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Core normalization operations
    # Compute L2 norm (simplified for this example)
    x_squared = x * x
    norm = tl.sqrt(tl.sum(x_squared, axis=0)) + 1e-05
    
    # Scale and clamp
    result = norm * 0.14433756729740643
    result = tl.maximum(result, 1e-05)
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_norm_clamp(x):
    # Get total number of elements
    N = x.numel()
    
    # Set up Triton kernel parameters
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the Triton kernel
    norm_clamp_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_norm_clamp