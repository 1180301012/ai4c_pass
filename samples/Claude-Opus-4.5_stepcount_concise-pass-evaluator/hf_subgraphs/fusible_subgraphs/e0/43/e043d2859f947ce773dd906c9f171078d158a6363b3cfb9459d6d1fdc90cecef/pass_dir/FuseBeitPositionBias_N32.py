import torch
import triton
import triton.language as tl

@triton.jit
def triton_cat_kernel(
    in_1_ptr, in_0_ptr, out_ptr,
    n1, n0, cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total = (n1 + n0) * cols
    mask = offsets < total
    
    row = offsets // cols
    col = offsets % cols
    
    in_1_mask = (row < n1) & mask
    in_0_mask = (row >= n1) & mask
    
    val_1 = tl.load(in_1_ptr + row * cols + col, mask=in_1_mask, other=0.0)
    val_0 = tl.load(in_0_ptr + (row - n1) * cols + col, mask=in_0_mask, other=0.0)
    
    result = tl.where(row < n1, val_1, val_0)
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_cat(in_0, in_1):
    # Concatenate using Triton: [in_1, in_0] along dim 0
    n1, cols = in_1.shape
    n0 = in_0.shape[0]
    total = n1 + n0
    out = torch.empty((total, cols), dtype=in_1.dtype, device=in_1.device)
    
    n_elements = total * cols
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    triton_cat_kernel[grid](
        in_1, in_0, out,
        n1, n0, cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    return tmp_0

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return optimized_cat