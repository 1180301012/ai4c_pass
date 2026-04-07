import torch
import triton
import triton.language as tl

def pattern(x, divisor1, divisor2):
    # Match the actual computation pattern from the model:
    # result = x / divisor1, then tmp = result, then tmp = tmp / divisor2
    result = x / divisor1
    tmp = result
    tmp = tmp / divisor2
    return tmp

def replacement_args(x, divisor1, divisor2):
    return (x, divisor1, divisor2)

@triton.jit
def optimized_div_fusion_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse division operations: x / divisor1 / divisor2 = x * (1.0 / (divisor1 * divisor2))
    # In our case: divisor1 = 16, divisor2 = 0.05, so multiplier = 1.0 / (16 * 0.05) = 1.25
    out = x * multiplier
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_division_fusion(x, divisor1, divisor2):
    """
    Optimized division fusion that replaces x / divisor1 / divisor2 with x * multiplier
    where multiplier = 1.0 / (divisor1 * divisor2)
    """
    if divisor1 == 16 and divisor2 == 0.05:
        # In our specific case: 1.0 / (16 * 0.05) = 1.25
        multiplier = 1.25
    elif divisor1 == 0.05 and divisor2 == 0.05:
        # Handle other possible cases
        multiplier = 1.0 / (divisor1 * divisor2)
    else:
        # Generic case: compute the multiplier
        multiplier = 1.0 / (divisor1 * divisor2)
    
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    optimized_div_fusion_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        multiplier=multiplier,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_division_fusion