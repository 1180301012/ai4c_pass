import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    return x

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized L2 normalization kernel
@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (L2 norm is computed along columns per row)
    row_idx = tl.program_id(0)
    offset = row_idx * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_rows * n_cols
    
    # Load the row data
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    
    # Compute L2 norm (sum of squares) for this row
    sq_sum = tl.sum(x * x)
    norm = tl.sqrt(sq_sum + 1e-8)  # Add epsilon for numerical stability
    
    # Normalize by dividing by the norm
    out = x / norm
    
    # Store the result
    tl.store(out_ptr + offset, out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_l2_normalize(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    
    # Adjust grid size based on tensor dimensions
    grid = (n_rows,)
    
    out = torch.empty_like(x)
    
    l2_normalize_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_l2_normalize