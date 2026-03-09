import torch
import triton
import triton.language as tl

def pattern(x):
    mask1 = x != 0
    mask2 = x == 0
    return mask2, mask1

def replacement_args(x):
    return (x,)

@triton.jit
def dual_mask_kernel(
    x_ptr,
    mask1_ptr,  # != 0
    mask2_ptr,  # == 0
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both masks simultaneously - they are complements
    is_zero = x == 0.0
    is_non_zero = ~is_zero  # This is equivalent to x != 0
    
    # Store both masks as float32 (for compatibility with torch.bool)
    tl.store(mask1_ptr + offsets, tl.where(is_non_zero, 1.0, 0.0), mask=mask)
    tl.store(mask2_ptr + offsets, tl.where(is_zero, 1.0, 0.0), mask=mask)

@torch.fx.wrap
def dual_mask_computation(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    mask1 = torch.empty_like(x, dtype=torch.float32)  # != 0 result
    mask2 = torch.empty_like(x, dtype=torch.float32)  # == 0 result
    
    dual_mask_kernel[(num_programs,)](
        x_ptr=x,
        mask1_ptr=mask1,
        mask2_ptr=mask2,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Convert back to boolean
    return mask2 > 0.5, mask1 > 0.5

def replacement_func():
    return dual_mask_computation