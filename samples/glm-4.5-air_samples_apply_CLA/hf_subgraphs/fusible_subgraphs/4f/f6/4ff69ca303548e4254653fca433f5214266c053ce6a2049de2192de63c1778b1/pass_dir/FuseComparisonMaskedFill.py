import torch
import triton
import triton.language as tl

def pattern(x, mask, value):
    return x.masked_fill(mask, value)

def replacement_args(x, mask, value):
    return (x, mask, value)

@triton.jit
def optimized_masked_fill_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data efficiently
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=False)
    
    # Apply masked fill using conditional move
    out = tl.where(mask_val, value, x)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_masked_fill(x, mask, value):
    n_elements = x.numel()
    
    # Optimized block size for better performance with this tensor size
    if n_elements >= 1000000:
        BLOCK_SIZE = 4096  # Large block for big tensors
    elif n_elements >= 500000:
        BLOCK_SIZE = 2048  # Medium block
    else:
        BLOCK_SIZE = 1024  # Small block
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Pre-allocate output tensor
    out = torch.empty_like(x, dtype=x.dtype)
    
    # Launch autotuned kernel
    optimized_masked_fill_kernel[(num_programs,)](
        x_ptr=x,
        mask_ptr=mask,
        out_ptr=out,
        n_elements=n_elements,
        value=value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_masked_fill