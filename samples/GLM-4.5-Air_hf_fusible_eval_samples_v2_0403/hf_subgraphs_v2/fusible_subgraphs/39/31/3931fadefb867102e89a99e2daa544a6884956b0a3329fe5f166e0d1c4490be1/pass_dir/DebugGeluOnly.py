import torch
import triton
import triton.language as tl

# Pattern matching function - match GELU operation on input tensor
def pattern(in_0):
    # Match GELU operation
    out = torch.nn.functional.gelu(in_0)
    return out

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Simple Triton kernel for GELU
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply GELU using simple approximation: 0.5 * x * (1.0 + erf(x / sqrt(2)))
    # Using polynomial approximation for erf(x): x * (1.0 - x^2 * (1.0 - x^2 * 0.551))
    x_sq = x * x
    erf_approx = x * (1.0 - x_sq * (1.0 - x_sq * 0.551))
    sqrt_2 = 1.41421356237
    erf_scaled = erf_approx / sqrt_2
    
    # GELU = 0.5 * x * (1.0 + erf_scaled)
    gelu_out = 0.5 * x * (1.0 + erf_scaled)
    
    # Store result
    tl.store(out_ptr + offsets, gelu_out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def triton_gelu(x):
    n_elements = x.numel()
    block_size = 1024
    num_programs = (n_elements + block_size - 1) // block_size
    
    out = torch.empty_like(x)
    
    gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=block_size
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_gelu