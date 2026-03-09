import torch
import triton
import triton.language as tl

def pattern(x):
    return x == 0

def replacement_args(x):
    return (x,)

@triton.jit
def zero_comparison_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute zero comparison directly without where operation
    # Store 1.0 for zero values, 0.0 for non-zero values
    result = tl.where(x == 0.0, 1.0, 0.0)
    
    # Store result directly
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_zero_comparison(x):
    N = x.numel()
    BLOCK_SIZE = 512  # Smaller block size for better occupancy
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=torch.float32)
    
    zero_comparison_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Convert back to boolean
    return out > 0.5

def replacement_func():
    return optimized_zero_comparison