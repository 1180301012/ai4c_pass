import torch
import triton
import triton.language as tl

def pattern(x):
    return torch.nn.functional.relu(x)

def replacement_args(x):
    return (x,)

@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    linear_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < n_elements
    
    # Load input
    x = tl.load(x_ptr + linear_idx, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(out_ptr + linear_idx, out, mask=mask)

@torch.fx.wrap
def triton_relu(x):
    # Handle different tensor shapes
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    relu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_relu