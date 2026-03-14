import torch
import triton
import triton.language as tl
import operator

# Pattern matching function - matches the exact computation in model.py
def pattern(in_0):
    """
    Match pattern:
    tmp_0 = in_0 != 0
    tmp_1 = in_0.masked_fill(tmp_0, -1000.0)
    tmp_2 = in_0 == 0
    return tmp_2, tmp_1
    """
    tmp_0 = in_0 != 0
    tmp_1 = in_0.masked_fill(tmp_0, -1000.0)
    tmp_2 = in_0 == 0
    return tmp_2, tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Triton kernel for fused operation with autotune
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_kernel(
    input_ptr,
    out_mask_ptr,
    out_filled_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: is_zero = (x == 0)
    is_zero = x == 0.0
    
    # When is_zero is True, x is 0.0, so we can directly use 0.0
    filled = tl.where(is_zero, 0.0, -1000.0)
    
    # Store outputs
    tl.store(out_mask_ptr + offsets, is_zero, mask=mask)
    tl.store(out_filled_ptr + offsets, filled, mask=mask)

# Combined wrapper
@torch.fx.wrap
def fused_impl(in_0):
    n = in_0.numel()
    out_mask = torch.empty(in_0.shape, dtype=torch.bool, device=in_0.device)
    out_filled = torch.empty_like(in_0)
    
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_kernel[grid](in_0, out_mask, out_filled, n)
    
    return (out_mask, out_filled)

# Wrapper that matches the pattern output structure
def replacement_wrapper(in_0):
    result = fused_impl(in_0)
    return operator.getitem(result, 0), operator.getitem(result, 1)

# Replacement function
def replacement_func():
    return replacement_wrapper