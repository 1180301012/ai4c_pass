import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def optimized_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements # Mask to ensure we don't go out of bounds
    
    # Load with proper memory coalescing
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Vectorized addition
    out = x + y
    
    # Store with proper memory coalescing
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    # For small tensors, use smaller block size to reduce overhead
    N = x.numel()
    
    # Optimized block size for small tensors - better occupancy
    if N <= 512:
        BLOCK_SIZE = 256  # Smaller block for better GPU occupancy
    else:
        BLOCK_SIZE = 512  # Medium block for larger tensors
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    optimized_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return triton_add