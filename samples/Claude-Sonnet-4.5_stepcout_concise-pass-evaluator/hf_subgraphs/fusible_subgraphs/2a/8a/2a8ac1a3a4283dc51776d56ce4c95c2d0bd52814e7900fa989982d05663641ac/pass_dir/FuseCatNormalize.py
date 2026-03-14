import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_cat_normalize_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Calculate row offset
    row_offset = row_idx * n_cols
    
    # Compute L2 norm for this row
    norm_sq = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        values = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0)
        norm_sq += tl.sum(values * values)
    
    norm = tl.sqrt(norm_sq)
    # Avoid division by zero
    norm = tl.maximum(norm, 1e-12)
    
    # Normalize and store
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        values = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0)
        normalized = values / norm
        tl.store(output_ptr + row_offset + offsets, normalized, mask=mask)

@torch.fx.wrap
def fused_cat_normalize(in_0):
    n_rows, n_cols = in_0.shape
    output = torch.empty_like(in_0)
    
    grid = (n_rows,)
    
    fused_cat_normalize_kernel[grid](
        in_0,
        output,
        n_rows,
        n_cols,
    )
    
    return output

def pattern(in_0):
    tmp_0 = torch.cat([in_0], 1)
    return (tmp_0,)

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return fused_cat_normalize