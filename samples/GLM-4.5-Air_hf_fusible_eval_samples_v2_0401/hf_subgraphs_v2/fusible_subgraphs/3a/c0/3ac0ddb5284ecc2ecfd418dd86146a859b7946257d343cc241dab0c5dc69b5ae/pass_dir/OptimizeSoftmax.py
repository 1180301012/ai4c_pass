import torch
import triton
import triton.language as tl

# Pattern matching function for softmax operation
def pattern(x):
    return torch.nn.functional.softmax(x, dim=-1)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized softmax kernel using Triton
@triton.jit
def triton_softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Apply softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    max_val = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / (sum_exp + 1e-20)
    
    tl.store(out_ptr + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def triton_softmax(x):
    # Handle the case where x might need reshaping
    # We expect x to be in shape [8, 300, 625] from the model
    original_shape = x.shape
    
    # Flatten for processing while keeping track of original shape
    total_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    triton_softmax_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return triton_softmax