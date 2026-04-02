import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Very simple pattern that just does arithmetic operations.
    This should definitely match something in the computation.
    """
    # Simple addition pattern
    result = x + 1.0
    return result

def replacement_args(x):
    """Extract the input tensor"""
    return (x,)

@triton.jit
def simple_add_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel that adds 1.0 to each element"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x + 1.0
    tl.store(y_ptr + offsets, y, mask=mask)

def simple_add(x):
    """Simple function that adds 1.0 to all elements using Triton"""
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    y = torch.empty_like(x)
    simple_add_kernel[grid](
        x, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y

@torch.fx.wrap
def simple_add_wrapper(x):
    """Wrapper function for simple addition"""
    return simple_add(x)

def replacement_func():
    """Returns the replacement function"""
    return simple_add_wrapper