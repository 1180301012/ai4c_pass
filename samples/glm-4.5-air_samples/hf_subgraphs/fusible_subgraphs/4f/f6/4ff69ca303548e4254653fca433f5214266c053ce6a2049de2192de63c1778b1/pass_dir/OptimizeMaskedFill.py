import torch
import triton
import triton.language as tl

def pattern(x, mask, fill_value):
    return x.masked_fill(mask, fill_value)

def replacement_args(x, mask, fill_value):
    return (x, mask, fill_value)

@triton.jit
def optimized_masked_fill_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data and mask
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    
    # Convert mask to boolean equivalent for where operation
    # In Triton, use 1.0 for True, 0.0 for False
    condition = mask_vals > 0.5
    
    # Apply masked fill using where
    result = tl.where(condition, fill_value, x)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_masked_fill(x, mask, fill_value):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Convert boolean mask to float32 if needed for kernel
    if mask.dtype == torch.bool:
        mask_float = mask.to(torch.float32)
    else:
        mask_float = mask
    
    optimized_masked_fill_kernel[(num_programs,)](
        x_ptr=x,
        mask_ptr=mask_float,
        out_ptr=out,
        n_elements=N,
        fill_value=fill_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_masked_fill