import torch
import triton
import triton.language as tl

def pattern(x):
    # Simplified pattern: just match normalize operation
    return torch.nn.functional.normalize(x, p=2, dim=1)

def replacement_args(x):
    return (x,)

@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute program id based on 2D grid
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Create offsets within the block
    row_start = m * BLOCK_SIZE_M
    col_offsets = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Ensure we don't go out of bounds
    row_mask = row_start < n_rows
    col_mask = col_offsets < n_cols
    
    # Load the row data if within bounds
    if row_start < n_rows:
        x_row = tl.load(x_ptr + row_start * n_cols + col_offsets, mask=col_mask, other=0.0)
        # Compute L2 norm for this row
        row_norm = tl.sqrt(tl.sum(x_row * x_row))
        # Avoid division by zero
        row_norm = tl.where(row_norm == 0, 1.0, row_norm)
        # Normalize the row
        out_row = x_row / row_norm
        # Store the results
        tl.store(out_ptr + row_start * n_cols + col_offsets, out_row, mask=col_mask)

@torch.fx.wrap
def l2_normalize_triton(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    # Configure block sizes based on problem size
    BLOCK_SIZE_N = min(256, n_cols)  # Each row is processed independently for columns
    
    # Grid configuration: (num_row_blocks, num_col_blocks)
    num_row_blocks = (n_rows + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_col_blocks = 1  # We process all columns for each row in one kernel launch
    
    l2_normalize_kernel[(num_row_blocks, num_col_blocks)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return l2_normalize_triton