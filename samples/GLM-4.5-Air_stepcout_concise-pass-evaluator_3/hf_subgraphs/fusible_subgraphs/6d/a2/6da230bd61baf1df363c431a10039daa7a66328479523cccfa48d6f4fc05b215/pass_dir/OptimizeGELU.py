import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Optimized GELU activation kernel using Triton with better memory access patterns"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with better alignment considerations
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Use GELU with more accurate formulation that matches PyTorch better
    # GELU(x) = x * 0.5 * (1.0 + erf(x / sqrt(2)))
    sqrt_2 = 1.41421356237
    erf_arg = x / sqrt_2
    erf_out = tl.erf(erf_arg)
    gelu_out = x * 0.5 * (1.0 + erf_out)
    
    # Store output with proper memory access
    tl.store(out_ptr + offsets, gelu_out, mask=mask)

@torch.fx.wrap
def triton_gelu(x):
    """Triton-optimized GELU implementation with autotuning"""
    n_elements = x.numel()
    
    # Optimize block size based on input size for better GPU utilization
    if n_elements < 4096:
        BLOCK_SIZE = 128  # Small inputs - smaller blocks
    elif n_elements < 65536:
        BLOCK_SIZE = 256  # Medium inputs - medium blocks
    else:
        BLOCK_SIZE = 1024 if (n_elements % 1024) == 0 else 512  # Large inputs - optimized blocks
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Ensure output has correct dtype (match input)
    out = torch.empty_like(x, dtype=x.dtype)
    
    gelu_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(x):
    """Match GELU operation"""
    # Apply GELU exactly as in the original model
    tmp_0 = torch.nn.functional.gelu(x)
    return tmp_0

def replacement_args(x):
    """Extract arguments for replacement"""
    return (x,)

def replacement_func():
    """Return optimized GELU function"""
    return triton_gelu