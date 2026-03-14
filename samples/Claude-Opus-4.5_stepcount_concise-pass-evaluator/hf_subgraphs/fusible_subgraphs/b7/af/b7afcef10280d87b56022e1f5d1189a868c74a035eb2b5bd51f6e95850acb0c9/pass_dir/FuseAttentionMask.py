import torch
import triton
import triton.language as tl

# Pattern matching function - match just the to(bool) and masked_fill operations
def pattern(tmp_2):
    tmp_3 = tmp_2.to(torch.bool)
    tmp_4 = tmp_2.masked_fill(tmp_3, -3.4028234663852886e+38)
    return tmp_4

# Argument extraction function
def replacement_args(tmp_2):
    return (tmp_2,)

# Constants
FILL_VALUE = -3.4028234663852886e+38

# Triton kernel for fused to(bool) + masked_fill computation
@triton.jit
def bool_masked_fill_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load, compare, and select in one pass
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # Where x != 0 (True in bool), fill with large negative value
    result = tl.where(x != 0.0, -3.4028234663852886e+38, x)
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper decorated with torch.fx.wrap
@torch.fx.wrap
def bool_masked_fill_wrapper(tmp_2):
    N = tmp_2.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    out = torch.empty_like(tmp_2)
    bool_masked_fill_kernel[grid](tmp_2, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

# Replacement function - returns the wrapper function
def replacement_func():
    return bool_masked_fill_wrapper