import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0 / 11.313708498984761
    return (tmp_0,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def div_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    constant: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Division by constant
    divided = x / constant
    
    # Store result
    tl.store(out_ptr + offsets, divided, mask=mask)

@torch.fx.wrap
def div_by_constant(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Launch the division kernel
    div_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        constant=11.313708498984761,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return div_by_constant