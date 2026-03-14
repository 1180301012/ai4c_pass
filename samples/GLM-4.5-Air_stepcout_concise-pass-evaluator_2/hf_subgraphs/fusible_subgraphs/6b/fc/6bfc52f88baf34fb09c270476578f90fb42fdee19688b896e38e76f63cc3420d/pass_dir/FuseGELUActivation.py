import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Matches GELU computation: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    tmp_0 = in_0 * 0.5
    tmp_1 = in_0 / 1.4142135623730951  # sqrt(2)
    tmp_2 = torch.erf(tmp_1)
    tmp_3 = 1.0 + tmp_2
    tmp_4 = tmp_0 * tmp_3
    return (tmp_4,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized GELU kernel using Triton
    
    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Constants - precomputed and optimized
    inv_sqrt_2 = 0.7071067811865476  # 1.0 / sqrt(2) for faster computation
    half = 0.5
    
    # Optimized GELU computation: 0.5 * x * (1 + erf(x * inv_sqrt_2))
    # Reduced operations by combining x/sqrt(2) into multiplication
    x_scaled = x * inv_sqrt_2
    erf_result = tl.erf(x_scaled)
    out = half * x * (1.0 + erf_result)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def optimized_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Optimized GELU kernel with configurable warp and stage settings"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Constants
    sqrt_2 = 1.4142135623730951
    half = 0.5
    
    # Optimized GELU computation fused into single kernel
    x_div_sqrt2 = x / sqrt_2
    erf_result = tl.erf(x_div_sqrt2)
    one_plus_erf = 1.0 + erf_result
    out = half * x * one_plus_erf
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_gelu(x):
    """Fused GELU computation using Triton"""
    n_elements = x.numel()
    
    # Choose block size dynamically based on tensor dimensions
    if n_elements > 10000000:
        BLOCK_SIZE = 4096
        num_warps = 16
        num_stages = 3
    elif n_elements > 1000000:
        BLOCK_SIZE = 2048
        num_warps = 16
        num_stages = 3
    elif n_elements > 100000:
        BLOCK_SIZE = 1024
        num_warps = 8
        num_stages = 2
    else:
        BLOCK_SIZE = 512
        num_warps = 4
        num_stages = 2
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch optimized kernel with tuned parameters
    optimized_gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages
    )
    
    return out

def replacement_func():
    return fused_gelu