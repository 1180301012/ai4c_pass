import torch
import triton
import triton.language as tl

def pattern(x, y):
    # tmp_2 = in_2 + in_3
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles a contiguous block of data
    start_pid = pid * BLOCK_SIZE
    offsets = start_pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    """Optimized addition using Triton"""
    if x.shape != y.shape or x.dtype != y.dtype:
        # Fallback to regular addition if shapes or dtypes don't match
        return x + y
    
    out = torch.empty_like(x)
    
    # Calculate launch configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_add