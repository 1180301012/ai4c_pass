import torch
import triton
import triton.language as tl

# Pattern matching function - match simple Add operation
def pattern(x1, x2):
    """
    Match Add pattern:
    - result = x1 + x2
    """
    result = x1 + x2
    return result

# Argument extraction function
def replacement_args(x1, x2):
    return (x1, x2)

@triton.jit
def add_kernel_small(
    x1_ptr, x2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Add kernel for small inputs"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    out = x1 + x2
    
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def add_kernel_large(
    x1_ptr, x2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Add kernel optimized for large inputs - uses vectorized loads"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Load and compute
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    out = x1 + x2
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def add_triton(x1, x2):
    """Triton Add wrapper - optimized for various sizes"""
    n_elements = x1.numel()
    out = torch.empty_like(x1)
    
    # Use larger block sizes for better performance on large tensors
    if n_elements >= 524288:  # 128*64*64 or larger
        BLOCK_SIZE = 8192
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        add_kernel_large[grid](x1, x2, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        add_kernel_small[grid](x1, x2, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

def replacement_func():
    return add_triton