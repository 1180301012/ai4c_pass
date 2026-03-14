import torch
import triton
import triton.language as tl

def pattern(x):
    y = x.to(torch.float32)
    return y

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_type_conversion_kernel(
    out_ptr,
    ptr_x,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with vectorized loading for better performance
    x_val = tl.load(ptr_x + offsets, mask=mask, other=0.0)
    
    # Store result with explicit type conversion to float32
    tl.store(out_ptr + offsets, x_val.to(tl.float32), mask=mask)

@torch.fx.wrap
def optimized_type_conversion(x):
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024  # Power of 2 for better GPU utilization
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=torch.float32)
    
    optimized_type_conversion_kernel[(num_programs,)](
        out_ptr=out,
        ptr_x=x,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_type_conversion