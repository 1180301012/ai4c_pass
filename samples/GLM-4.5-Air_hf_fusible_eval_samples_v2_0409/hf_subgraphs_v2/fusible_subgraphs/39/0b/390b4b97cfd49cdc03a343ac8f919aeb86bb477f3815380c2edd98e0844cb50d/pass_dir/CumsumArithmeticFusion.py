import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_1 = x.cumsum(-1)
    tmp_2 = tmp_1 - 1
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def cumsum_arithmetic_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of BLOCK_SIZE elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    
    # Compute cumsum + arithmetic in one step
    # For cumsum, we need to handle the sequential nature
    cumsum = tl.cumsum(x)
    result = cumsum - 1
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_cumsum_arithmetic(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    cumsum_arithmetic_kernel[(num_programs,)](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_cumsum_arithmetic