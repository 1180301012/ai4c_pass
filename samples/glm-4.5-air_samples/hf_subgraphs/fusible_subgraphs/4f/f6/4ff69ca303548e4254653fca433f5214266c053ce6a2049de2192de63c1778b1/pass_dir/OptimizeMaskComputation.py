import torch
import triton
import triton.language as tl

def pattern(x):
    mask0 = x != 0
    mask1 = x == 0
    tmp1 = x.masked_fill(mask0, -1000.0)
    return (mask1, tmp1)

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_mask_kernel(
    x_ptr,
    zero_mask_ptr,  # x == 0 result (first return value)
    filled_result_ptr,  # x.masked_fill(x != 0, -1000.0) result (second return value)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both masks and result in one pass
    is_zero = x == 0.0
    is_non_zero = ~is_zero
    
    # Create filled result: fill non-zero elements with -1000.0
    filled_x = tl.where(is_non_zero, -1000.0, x)
    
    # Store results
    tl.store(zero_mask_ptr + offsets, tl.where(is_zero, 1.0, 0.0), mask=mask)
    tl.store(filled_result_ptr + offsets, filled_x, mask=mask)

@torch.fx.wrap
def optimized_mask_computation(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    zero_mask = torch.empty_like(x, dtype=torch.float32)  # x == 0 mask
    filled_result = torch.empty_like(x)  # x.masked_fill(x != 0, -1000.0) result
    
    optimized_mask_kernel[(num_programs,)](
        x_ptr=x,
        zero_mask_ptr=zero_mask,
        filled_result_ptr=filled_result,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Convert zero mask back to boolean and return both results
    return zero_mask > 0.5, filled_result

def replacement_func():
    return optimized_mask_computation