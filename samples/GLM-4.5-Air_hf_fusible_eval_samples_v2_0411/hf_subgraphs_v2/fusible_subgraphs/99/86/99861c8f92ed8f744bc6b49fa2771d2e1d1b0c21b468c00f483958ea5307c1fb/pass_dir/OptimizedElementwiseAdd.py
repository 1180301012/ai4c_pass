import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def optimized_add_kernel(
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
def optimized_add(x, y):
    # Use large block sizes to minimize kernel launch overhead
    # This is key for beating optimized library implementations
    n_elements = x.numel()
    
    # Use very large block sizes for best amortization of launch overhead
    # Our tensors are large (e.g., 1 x 249 x 1024 = 255K elements)
    if n_elements > 1024 * 1024:  # Extra large
        BLOCK_SIZE = 4096
    elif n_elements > 512 * 1024:  # Very large
        BLOCK_SIZE = 2048
    elif n_elements > 128 * 1024:  # Large
        BLOCK_SIZE = 1024
    else:  # Medium/small
        BLOCK_SIZE = 512
    
    # Calculate minimal grid size for maximum utilization
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x, device=x.device)
    
    # Launch kernel with 1D grid and large blocks - minimal computation overhead
    optimized_add_kernel[grid_size, 1](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_add