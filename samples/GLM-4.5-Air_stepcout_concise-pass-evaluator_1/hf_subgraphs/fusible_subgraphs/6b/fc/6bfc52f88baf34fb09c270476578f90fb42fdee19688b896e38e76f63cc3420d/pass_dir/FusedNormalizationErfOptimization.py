import torch
import triton
import triton.language as tl
import math

# Pattern matching function - matches the exact computation in model.py
def pattern(in_0):
    tmp_0 = in_0 * 0.5
    tmp_1 = in_0 / 1.4142135623730951
    tmp_2 = torch.erf(tmp_1)
    tmp_1 = None
    tmp_3 = 1.0 + tmp_2
    tmp_2 = None
    tmp_4 = tmp_0 * tmp_3
    tmp_0 = tmp_3 = None
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel with block size selection based on tensor size
@triton.jit
def fused_normalization_erf_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Constants precomputed for better performance
    scale_factor = 0.5
    sqrt2_reciprocal = 1.0 / 1.4142135623730951
    
    # Compute fused operation in optimal order for arithmetic units
    x_div_sqrt2 = x * sqrt2_reciprocal  # First: division by √2 (multiplication)
    erf_result = tl.erf(x_div_sqrt2)    # Second: error function
    one_plus_erf = 1.0 + erf_result     # Third: add 1.0
    x_scaled = x * scale_factor          # Fourth: scale by 0.5
    out = x_scaled * one_plus_erf       # Final multiplication
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper with dynamic block size selection
@torch.fx.wrap
def fused_normalization_erf(in_0):
    # Get tensor properties
    N = in_0.numel()
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Select optimal block size based on tensor size
    if N < 100000:  # Small tensors
        BLOCK_SIZE = 256
    elif N < 1000000:  # Medium tensors
        BLOCK_SIZE = 1024
    else:  # Large tensors
        BLOCK_SIZE = 2048
    
    # Calculate grid size
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_normalization_erf_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_normalization_erf