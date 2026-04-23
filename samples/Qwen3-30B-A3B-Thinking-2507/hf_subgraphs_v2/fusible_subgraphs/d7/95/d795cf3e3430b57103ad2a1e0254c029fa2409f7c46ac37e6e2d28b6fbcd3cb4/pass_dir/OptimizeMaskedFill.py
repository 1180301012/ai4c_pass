import torch
import triton
import triton.language as tl

def pattern(tmp_12):
    mask_non_zero = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(mask_non_zero, -1000.0)
    mask_zero = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(mask_zero, 0.0)
    return tmp_16

def replacement_args(tmp_12):
    return (tmp_12,)

@triton.jit
def masked_fill_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl.where(x != 0, -1000.0, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def optimized_masked_fill(x):
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    masked_fill_kernel[grid](
        x,
        out,
        n,
        BLOCK_SIZE
    )
    return out

def replacement_func():
    return optimized_masked_fill