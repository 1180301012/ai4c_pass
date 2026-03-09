import torch
import triton
import triton.language as tl

# Simple addition pattern
def pattern(x, y):
    return x + y

# Extract arguments for the fused kernel
def replacement_args(x, y):
    return (x, y)

# Optimized addition kernel with autotuning configs
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise addition with better vectorization
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)



@torch.fx.wrap
def triton_add(x, y):
    N = x.numel()
    
    # Choose optimal block size based on tensor size
    if N <= 4096:
        BLOCK_SIZE = 128
    elif N <= 16384:
        BLOCK_SIZE = 256
    elif N <= 65536:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    # Use basic kernel - remove autotuning for stability
    add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return triton_add