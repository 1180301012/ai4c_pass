import torch
import triton
import triton.language as tl

def pattern(x, fill_value):
    mask = x != 0
    result = x.masked_fill(mask, fill_value)
    return result, mask

def replacement_args(x, fill_value):
    return (x, fill_value)

@triton.jit
def fused_masked_kernel(
    x_ptr,
    out_ptr, 
    mask_ptr,
    n_elements,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mask and fill in one operation
    is_zero = x == 0.0
    filled_x = tl.where(is_zero, x, fill_value)
    
    # Store results
    tl.store(out_ptr + offsets, filled_x, mask=mask)
    # Store boolean mask as float32 (to match torch.bool behavior in context)
    tl.store(mask_ptr + offsets, tl.where(is_zero, 0.0, 1.0), mask=mask)

@torch.fx.wrap
def fused_masked_fill(x, fill_value):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    mask = torch.empty_like(x, dtype=torch.float32)  # Store as float32 for compatibility
    
    fused_masked_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        mask_ptr=mask,
        n_elements=N,
        fill_value=fill_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out, mask > 0.5  # Convert back to boolean

def replacement_func():
    return fused_masked_fill