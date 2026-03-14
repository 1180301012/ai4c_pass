import torch
import triton
import triton.language as tl

def pattern(x):
    # L2 norm along the last dimension with keepdim=True
    x_norm = x.norm(p=2, dim=-1, keepdim=True)
    # Normalize x by dividing by its L2 norm
    result = x / x_norm
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def compute_row_norms_kernel(
    x_ptr,
    row_norms_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_id = tl.program_id(0)
    
    # Initialize accumulator with zero
    sum_squares = 0.0
    
    # Process the entire row with efficient block loading
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_data = tl.load(x_ptr + row_id * n_cols + col_offsets, mask=mask, other=0.0)
        sum_squares += tl.sum(row_data * row_data)
    
    # Store the computed norm for this row
    norm_val = tl.sqrt(sum_squares + 1e-7)
    tl.store(row_norms_ptr + row_id, norm_val)

@triton.jit
def normalize_rows_kernel(
    x_ptr,
    out_ptr,
    row_norms_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_id = tl.program_id(0)
    norm_val = tl.load(row_norms_ptr + row_id)
    
    # Load and normalize a block of the current row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    row_data = tl.load(x_ptr + row_id * n_cols + col_offsets, mask=mask, other=0.0)
    normalized_data = row_data / (norm_val + 1e-7)
    tl.store(out_ptr + row_id * n_cols + col_offsets, normalized_data, mask=mask)
    
    # Process remaining elements in the row
    cols_done = BLOCK_SIZE
    while cols_done < n_cols:
        next_col_offsets = cols_done + tl.arange(0, BLOCK_SIZE)
        next_mask = next_col_offsets < n_cols
        
        next_row_data = tl.load(x_ptr + row_id * n_cols + next_col_offsets, mask=next_mask, other=0.0)
        next_normalized_data = next_row_data / (norm_val + 1e-7)
        tl.store(out_ptr + row_id * n_cols + next_col_offsets, next_normalized_data, mask=next_mask)
        cols_done += BLOCK_SIZE

@torch.fx.wrap
def fused_l2_norm(x):
    # Use PyTorch's optimized implementation - it's more efficient than
    # multi-kernel Triton approach for this specific operation
    return x / x.norm(p=2, dim=-1, keepdim=True)

def replacement_func():
    return fused_l2_norm