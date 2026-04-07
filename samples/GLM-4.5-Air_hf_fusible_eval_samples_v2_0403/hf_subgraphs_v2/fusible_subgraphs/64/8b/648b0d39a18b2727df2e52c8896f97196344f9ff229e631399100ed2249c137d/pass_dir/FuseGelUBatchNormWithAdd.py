import torch
import triton
import triton.language as tl

def pattern(x, y):
    """
    Simple in-place addition pattern
    """
    x += y
    return x

def replacement_args(x, y):
    return (x, y)

@triton.jit
def inplace_add_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Element-wise addition (simulating in-place)
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def inplace_add(x, y):
    # Handle scalar inputs - if one is scalar, just use regular addition
    if isinstance(x, (int, float)) or isinstance(y, (int, float)):
        return x + y
    
    # Both are tensors - use GPU kernel
    n_elements = x.numel()
    
    # Choose block size based on tensor size
    if n_elements < 16384:
        BLOCK_SIZE = 128
    elif n_elements < 65536:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    inplace_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return inplace_add