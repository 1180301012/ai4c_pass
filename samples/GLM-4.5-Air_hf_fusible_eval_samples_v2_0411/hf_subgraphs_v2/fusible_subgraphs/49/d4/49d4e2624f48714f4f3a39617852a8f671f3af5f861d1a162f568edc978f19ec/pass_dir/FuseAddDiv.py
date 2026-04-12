import torch
import triton
import triton.language as tl

# Pattern: tmp_2 = in_2 + in_3; tmp_3 = tmp_2 / 2
def pattern(in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    return tmp_3

def replacement_args(in_2, in_3):
    return (in_2, in_3)

@triton.jit
def add_div_kernel(
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
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute (x + y) / 2
    out = (x + y) * 0.5
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def add_div(x, y):
    n_elements = x.numel()
    
    # For small tensors, use a power of 2 block size close to tensor size
    if n_elements <= 512:
        BLOCK_SIZE = 512  # Power of 2, good for tensors <= 512
        num_programs = 1
    elif n_elements <= 1024:
        BLOCK_SIZE = 512  # Power of 2, better than 1024 for small tensors
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    else:
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    add_div_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return add_div