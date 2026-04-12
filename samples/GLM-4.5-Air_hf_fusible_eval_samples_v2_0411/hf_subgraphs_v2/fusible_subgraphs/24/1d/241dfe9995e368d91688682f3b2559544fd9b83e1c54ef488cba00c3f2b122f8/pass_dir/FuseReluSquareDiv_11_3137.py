import torch
import triton
import triton.language as tl

# Pattern matching function - start with simple square operation
def pattern(x):
    return torch.square(x)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized kernel for square operation only
@triton.jit
def fast_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to avoid out-of-bounds access
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fast square operation: x * x
    out = x * x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fast_square(x):
    N = x.numel()
    
    # Adaptive block size based on tensor size for better occupancy
    if N < 1024:
        BLOCK_SIZE = 128
    elif N < 10000:
        BLOCK_SIZE = 256
    elif N < 100000:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
        
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch fast square kernel with better grid configuration
    if num_programs > 1:
        # Use 2D grid for better GPU utilization
        fast_square_kernel[(num_programs, 1)](
            x_ptr=x,
            out_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Use 1D grid for small tensors
        fast_square_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fast_square