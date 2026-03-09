import torch
import triton
import triton.language as tl

def pattern(x, y):
    return torch.sigmoid(x + y)

def replacement_args(x, y):
    return (x, y)

@triton.jit
def sigmoid_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate
    result = x + y
    out = tl.where(result >= 0, 
                   result / (1 + tl.exp(-result)), 
                   tl.exp(result) / (1 + tl.exp(result)))
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_sigmoid_add(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    sigmoid_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return triton_sigmoid_add