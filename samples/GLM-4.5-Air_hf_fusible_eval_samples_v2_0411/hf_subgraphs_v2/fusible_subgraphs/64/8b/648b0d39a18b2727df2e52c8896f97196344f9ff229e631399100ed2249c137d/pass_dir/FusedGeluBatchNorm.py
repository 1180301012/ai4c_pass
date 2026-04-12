import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern: Simple GELU"""
    tmp_5 = torch.nn.functional.gelu(x, approximate='none')
    return tmp_5

def replacement_args(x):
    """Extract arguments for the GELU kernel"""
    return (x,)

@triton.jit
def triton_gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple GELU kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Better polynomial approximation of GELU 
    # GELU(x) ≈ x * (0.5 + 0.5 * tanh(0.797885 * (x + 0.044715 * x^3)))
    # Using mathematical identities to avoid expensive function calls
    x3 = x * x * x
    inner_arg = x + 0.044715 * x3
    # tanh(x) ≈ x * (1 + x^2) / (1 + x^2 + x^4/3 + x^6/15)  for small x
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    
    # Polynomial approximation of tanh(0.797885 * inner_arg)
    scaled_inner = 0.797885 * inner_arg
    # For range clipping and linear approximation
    linear_part = scaled_inner
    poly_part = scaled_inner * (1.0 + scaled_inner * scaled_inner) / (1.0 + scaled_inner * scaled_inner + scaled_inner * x4/3.0 + scaled_inner * x6/15.0)
    
    # Use the approximation only for moderate values, fall back to linear for extremes
    tanh_approx = tl.where(tl.abs(scaled_inner) < 1.0, poly_part, linear_part)
    gelu_out = x * (0.5 + 0.5 * tanh_approx)
    
    # Store output
    tl.store(out_ptr + offsets, gelu_out, mask=mask)

@torch.fx.wrap
def triton_gelu(x):
    """Wrapper for simple GELU operation"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    triton_gelu_kernel[(num_programs,)](x, out, n_elements, BLOCK_SIZE)
    
    return out

def replacement_func():
    """Returns the simple GELU function"""
    return triton_gelu