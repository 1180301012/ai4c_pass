import torch
import triton
import triton.language as tl
import math

# Pattern matching function - matches the exact computation sequence
def pattern(in_0):
    # Exact computation pattern from the model
    tmp_0 = in_0 * 0.5
    tmp_1 = in_0 / 1.4142135623730951  # sqrt(2)
    tmp_2 = torch.erf(tmp_1)
    tmp_3 = 1.0 + tmp_2
    tmp_4 = tmp_0 * tmp_3
    return tmp_4  # Return the final result (same as model output)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized GELU kernel with autotuning
@triton.jit
def gelu_erf_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Constants - precomputed in kernel for better performance
    sqrt2 = 1.4142135623730951
    half = 0.5
    
    # Fused GELU computation: 0.5 * x * (1 + erf(x / sqrt(2)))
    # Optimized order of operations for better numerical stability and performance
    x_scaled = x / sqrt2  # x / sqrt(2)
    erf_result = tl.erf(x_scaled)  # erf(x / sqrt(2))
    gelu_val = half * x * (1.0 + erf_result)  # 0.5 * x * (1 + erf_result)
    
    # Store result
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

# Kernel wrapper with adaptive block sizing (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_gelu_erf(x):
    N = x.numel()
    
    # Adaptive block sizing based on tensor size for optimal performance
    if N < 8192:
        # Small tensors: smaller blocks to reduce overhead
        BLOCK_SIZE = 256
    elif N < 65536:
        # Medium tensors: moderate blocks
        BLOCK_SIZE = 512  
    elif N < 262144:
        # Large tensors: standard blocks
        BLOCK_SIZE = 1024
    else:
        # Very large tensors: larger blocks for better GPU utilization
        BLOCK_SIZE = 2048
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x, device=x.device)
    
    # Launch kernel with optimized configuration
    gelu_erf_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_gelu_erf