import torch
import triton
import triton.language as tl

# Minimal pattern - just match relu 
def pattern(x):
    return torch.relu(x)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Simple ReLU kernel
@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def relu_wrapper(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    relu_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return relu_wrapper