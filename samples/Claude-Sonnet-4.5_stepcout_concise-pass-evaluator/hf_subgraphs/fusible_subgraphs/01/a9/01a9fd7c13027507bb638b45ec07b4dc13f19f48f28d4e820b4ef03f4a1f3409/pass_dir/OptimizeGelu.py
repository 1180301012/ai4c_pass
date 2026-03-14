import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    """
    Match a simple GELU operation
    """
    return torch.nn.functional.gelu(x, approximate='none')


def replacement_args(x):
    """Extract arguments for the replacement function"""
    return (x,)


@triton.jit
def optimized_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized GELU kernel
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt_2 = 1.4142135623730951
    result = 0.5 * x * (1.0 + tl.math.erf(x / sqrt_2))
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def optimized_gelu(x):
    """
    Wrapper for optimized GELU
    """
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    optimized_gelu_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    """Return the replacement function"""
    return optimized_gelu