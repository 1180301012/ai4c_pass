import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance GELU kernel using Triton"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(0.7978845608028654 * x * (1 + 0.044715 * x * x)))
    x_cubed = x * x * x
    inner = 0.7978845608028654 * x * (1.0 + 0.044715 * x_cubed)
    tanh_val = tl.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_val)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_gelu(x):
    """Optimized GELU using Triton kernel"""
    N = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for good GPU occupancy
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch Triton kernel
    gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(x):
    """Match GELU operation"""
    return torch.nn.functional.gelu(x)

def replacement_args(x):
    """Extract arguments for GELU optimization"""
    return (x,)

def replacement_func():
    """Return optimized GELU function"""
    return triton_gelu