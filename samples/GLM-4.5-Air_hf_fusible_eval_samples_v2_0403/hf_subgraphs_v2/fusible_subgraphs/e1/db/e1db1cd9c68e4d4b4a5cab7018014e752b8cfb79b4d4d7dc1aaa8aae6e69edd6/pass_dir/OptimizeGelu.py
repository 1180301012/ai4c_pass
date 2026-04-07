import torch
import triton
import triton.language as tl

# Pattern matching function - just match GELU operation
def pattern(x):
    return torch.nn.functional.gelu(x)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel for GELU
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # High-performance GELU computation using standard math operations
    # tl.exp is more stable than tl.math.exp for this computation
    x_scaled = 1.702 * x
    sigmoid_approx = tl.exp(-x_scaled)
    sigmoid_approx = sigmoid_approx / (1.0 + sigmoid_approx)
    gelu_val = x * sigmoid_approx
    
    # Store result
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

@torch.fx.wrap
def triton_gelu(x):
    """Wrapper function to launch Triton GELU kernel with optimizations"""
    n_elements = x.numel()
    
    # Optimized block size based on tensor characteristics
    if n_elements >= 1048576:  # Large tensors
        BLOCK_SIZE = 2048
    elif n_elements >= 262144:  # Medium tensors 
        BLOCK_SIZE = 1024
    else:  # Small tensors
        BLOCK_SIZE = 512
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Use different grid configurations for optimal GPU utilization
    if num_programs > 1024:
        grid = (1024, (num_programs + 1023) // 1024)  # 2D grid for large workloads
    else:
        grid = (num_programs,)
    
    gelu_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference, not the actual call)
def replacement_func():
    return triton_gelu