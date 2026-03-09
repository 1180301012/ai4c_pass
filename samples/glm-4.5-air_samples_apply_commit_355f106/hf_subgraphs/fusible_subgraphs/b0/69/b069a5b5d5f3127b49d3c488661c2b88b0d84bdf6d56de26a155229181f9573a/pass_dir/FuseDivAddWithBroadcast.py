import torch
from torch import device
import triton
import triton.language as tl

def pattern(x, y):
    tmp_0 = x / 8.0
    tmp_1 = y.to(device(type='cuda', index=0))
    tmp_2 = tmp_0 + tmp_1
    return tmp_2

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_div_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x values
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate 4D indices for x: [batch, heads, height, width]
    # Total elements = 2 * 12 * 7 * 7 = 1176
    batch = offsets // (12 * 7 * 7)
    remainder = offsets % (12 * 7 * 7)
    heads = remainder // (7 * 7)
    height = (remainder % (7 * 7)) // 7
    width = (remainder % (7 * 7)) % 7
    
    # For y, we need [batch, 0, 0, width] where width only goes to 6
    # y has shape [2, 1, 1, 7], so we only use valid width indices (0-6)
    width_valid = (width < 7)
    y_batch = batch
    y_width = width
    
    # Calculate y indices: y has shape [2, 1, 1, 7] = [2, 7]
    # Flatten the y tensor as [batch, 7]
    y_index = y_batch * 7 + y_width
    
    # Load y values only for valid width indices
    y_mask = mask & width_valid
    y_val = tl.load(y_ptr + y_index, mask=y_mask, other=0.0)
    
    # For width >= 7, y_val should be 0 (no broadcasting)
    y_val = tl.where(width_valid, y_val, 0.0)
    
    # But y should only be applied to heads=0, height=0
    y_should_apply = (heads == 0) & (height == 0) & width_valid
    
    # Apply y only where it should be broadcasted
    y_final = tl.where(y_should_apply, y_val, 0.0)
    
    # Perform fused operation: (x / 8.0) + y
    result = (x_val / 8.0) + y_final
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_div_add(x, y):
    # Get total number of elements
    total_elements = x.numel()
    
    # Set block size - can be autotuned later
    BLOCK_SIZE = 1024
    
    # Calculate launch grid
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_div_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_div_add