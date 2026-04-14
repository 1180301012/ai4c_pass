import torch
import triton
import triton.language as tl

def pattern(x):
    """Simple pattern: scalar multiplication by 0.0625"""
    result = 0.0625 * x
    return result

def replacement_args(x):
    """Extract arguments for scalar multiplication"""
    return (x,)

@triton.jit
def scalar_multiply_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Scalar multiplication kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scale
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def debug_scalar_multiply(x):
    """Debug scalar multiplication using Triton"""
    # Create output tensor
    output = torch.empty_like(x)
    
    # Use proper scalar multiply kernel
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    scalar_multiply_kernel[(grid,)](
        x_ptr=x,
        out_ptr=output,
        n_elements=n_elements,
        scale=0.0625,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return debug scalar multiply function"""
    return debug_scalar_multiply