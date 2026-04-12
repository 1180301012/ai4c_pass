import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    # Get total number of elements
    n_elements = x.numel()
    
    # Try multiple block sizes and choose the fastest
    # Use smaller block sizes for better cache utilization on large tensors
    if n_elements > 1024 * 1024:  # Large tensor (>1M elements)
        BLOCK_SIZE = 512
    elif n_elements > 256 * 1024:  # Medium tensor (>256K elements)
        BLOCK_SIZE = 256
    else:  # Small tensor (<=256K elements)
        BLOCK_SIZE = 128
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with 2D grid for better utilization
    triton_add_kernel[(num_programs, 1)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_add