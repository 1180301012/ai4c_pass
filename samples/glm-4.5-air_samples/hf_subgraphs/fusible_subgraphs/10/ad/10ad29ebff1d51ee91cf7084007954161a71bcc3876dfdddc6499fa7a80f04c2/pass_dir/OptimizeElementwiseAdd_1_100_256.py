import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def elementwise_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_elementwise_add(x, y):
    N = x.numel()
    
    # For small tensors, use optimal block size
    if N <= 256:
        BLOCK_SIZE = N
    elif N <= 1024:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 128
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    elementwise_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_elementwise_add