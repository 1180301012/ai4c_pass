import torch
import triton
import triton.language as tl

@triton.jit
def fuse_clamp_sqr_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input - convert to float32 for computation
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute: tmp_1 = 1.0 - tmp_0
    tmp_1 = 1.0 - x
    
    # Original pattern:
    # tmp_2 = tmp_1.bool() -> True for non-zero values
    # tmp_3 = tmp_1.masked_fill(tmp_2, -inf) -> -inf for non-zero, tmp_1 for zero
    # tmp_4 = tmp_3 * tmp_1:
    #   - if tmp_1 > 0: -inf * positive = -inf
    #   - if tmp_1 == 0: 0 * 0 = 0
    #   - if tmp_1 < 0: -inf * negative = inf
    
    # Use sign to compute: sign(tmp_1) * infinity for non-zero, 0 for zero
    result = tl.where(tmp_1 > 0, float('-inf'), 
                tl.where(tmp_1 < 0, float('inf'), 0.0))
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fuse_clamp_sqr(x):
    """
    Fused kernel for:
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    
    This computes: sign(1-x) * infinity for non-zero, 0 for zero
    """
    N = x.numel()
    # Block size 256 works best for this tensor size
    BLOCK_SIZE = 256
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if num_programs < 1:
        num_programs = 1

    # Allocate output with correct dtype matching original pattern
    out = torch.empty_like(x, dtype=torch.float32)

    fuse_clamp_sqr_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def pattern(in_0):
    """
    Pattern matching the fused computation:
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return (tmp_4,)
    """
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return (tmp_4,)

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return fuse_clamp_sqr