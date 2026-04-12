import torch
import triton
import triton.language as tl

@triton.jit
def l2norm_normalize_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple and efficient L2 norm and normalization kernel for small matrices"""
    # Each program handles one row - optimal for small matrices with few rows
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Load entire row (efficient for small matrices with many columns)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row_idx * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute L2 norm for the row
    x_squared = x * x
    sum_x_squared = tl.sum(x_squared)
    norm = tl.sqrt(sum_x_squared)
    
    # Normalize the row (avoid division by zero)
    norm_safe = tl.where(norm == 0, 1.0, norm)
    out = x / norm_safe
    
    # Store normalized row
    tl.store(out_ptr + row_idx * n_cols + cols, out.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def l2norm_normalize_triton(x):
    """High-performance L2 norm and normalization function"""
    n_rows, n_cols = x.shape
    # Use next power of 2 for BLOCK_SIZE (required by Triton arange)
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2
    
    out = torch.empty_like(x)
    
    num_programs = n_rows
    
    # Launch kernel with one program per row (optimal for small matrices)
    l2norm_normalize_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(x):
    """Match L2 norm followed by division pattern"""
    tmp_0 = x.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = x / tmp_0
    return tmp_1  # Only return the observable intermediate

def replacement_args(x):
    """Extract argument for replacement"""
    return (x,)

def replacement_func():
    """Return the optimized kernel wrapper"""
    return l2norm_normalize_triton