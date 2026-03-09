import torch
import triton
import triton.language as tl

def pattern(conv_result, pos_embed):
    # Pattern: conv2d result + position embedding addition
    # This captures the element-wise addition of position embeddings
    out = conv_result + pos_embed
    return out

def replacement_args(conv_result, pos_embed):
    return (conv_result, pos_embed)

@triton.jit
def elementwise_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements preserving original data type
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_elementwise_add(x, y):
    # Input tensors should have the same shape
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    n_elements = x.numel()
    out = torch.empty_like(x, dtype=x.dtype)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    elementwise_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_elementwise_add