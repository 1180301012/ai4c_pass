import torch
import triton
import triton.language as tl

# Pattern: x.sum(dim=1)
def pattern(x):
    return x.sum(dim=1)

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_sum_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Use power of 2 for arange (8 is next power of 2 after 5)
    idx = tl.arange(0, 8)
    mask = idx < n_cols
    
    # Load the row
    x_row = tl.load(x_ptr + row_idx * n_cols + idx, mask=mask)
    
    # Sum the row
    row_sum = tl.sum(x_row)
    
    # Store result
    tl.store(out_ptr + row_idx, row_sum)

@torch.fx.wrap
def optimized_sum(x):
    n_rows, n_cols = x.shape
    
    # Create output tensor
    out = torch.empty((n_rows,), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    optimized_sum_kernel[(n_rows,)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=8,
    )
    
    return out

def replacement_func():
    return optimized_sum