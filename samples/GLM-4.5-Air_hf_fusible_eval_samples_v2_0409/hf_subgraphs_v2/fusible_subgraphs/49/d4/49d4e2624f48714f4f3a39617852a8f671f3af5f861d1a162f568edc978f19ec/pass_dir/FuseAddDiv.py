import torch
import triton
import triton.language as tl

def pattern(x, y):
    tmp_2 = x + y
    tmp_3 = tmp_2 / 2
    return tmp_3

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_add_div_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load with proper data type handling
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized fused operation: add then multiply by 0.5 (faster than divide)
    out = (x + y) * 0.5
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_div(x, y):
    n_elements = x.numel()
    # Use larger block size for better GPU occupancy
    BLOCK_SIZE = 1024  # Optimal for the small tensor size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use torch.empty_like to match original tensor properties
    out = torch.empty_like(x)
    
    fused_add_div_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_div