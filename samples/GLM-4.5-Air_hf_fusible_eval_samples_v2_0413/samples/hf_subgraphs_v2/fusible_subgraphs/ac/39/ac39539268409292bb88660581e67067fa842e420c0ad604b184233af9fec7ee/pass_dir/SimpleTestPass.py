import torch
import triton
import triton.language as tl

@triton.jit
def simple_cat_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    x_elements: tl.constexpr,
    y_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized concatenation kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (x_elements + y_elements)
    
    # Optimized: Load both tensors with better bounds checking
    x_load = tl.load(x_ptr + offsets, mask=offsets < x_elements, other=0)
    y_load = tl.load(y_ptr + offsets - x_elements, mask=offsets >= x_elements, other=0)
    
    # Use tl.where for conditional selection - this is well-optimized in Triton
    result = tl.where(offsets < x_elements, x_load, y_load)
    tl.store(out_ptr + offsets, result, mask)

@torch.fx.wrap
def optimized_concat(x, y):
    """Optimized concatenation function"""
    x_elements = x.numel()
    y_elements = y.numel()
    total_elements = x_elements + y_elements
    
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(total_elements, dtype=x.dtype, device=x.device)
    
    simple_cat_kernel[(grid_size,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        x_elements=x_elements,
        y_elements=y_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape to match the expected concatenation result
    # The original torch.cat([x, y]) should produce [x.shape[0] + y.shape[0], x.shape[1]] for 2D tensors
    if x.dim() == 2 and y.dim() == 2 and x.shape[1] == y.shape[1]:
        out = out.view(x.shape[0] + y.shape[0], x.shape[1])
    elif x.dim() == 1 and y.dim() == 1:
        out = out  # Already 1D, no reshape needed
    else:
        out = out.view(x.shape[0] + y.shape[0], -1)  # General case
    
    return out

def pattern(x, y):
    """Pattern matching tensor concatenation"""
    result = torch.cat([x, y])
    return result

def replacement_args(x, y):
    """Extract arguments for concatenation"""
    return (x, y)

def replacement_func():
    """Return optimized concatenation function"""
    return optimized_concat