import torch
import triton
import triton.language as tl

# Pattern matching function - matches L2 norm + division normalization
def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Optimized L2 normalization kernel using Triton
@triton.jit
def l2norm_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
):
    # Each program handles one row for reduction
    row_idx = tl.program_id(0)
    
    # For small rows, compute norm with a simple loop
    sum_sq = 0.0
    for col in range(n_cols):
        x = tl.load(x_ptr + row_idx * n_cols + col)
        sum_sq += x * x
    
    # Compute norm
    norm = tl.sqrt(sum_sq)
    
    # Normalize each element in the row
    for col in range(n_cols):
        x = tl.load(x_ptr + row_idx * n_cols + col)
        out = x / norm
        tl.store(out_ptr + row_idx * n_cols + col, out)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_l2norm(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    num_programs = n_rows
    
    l2norm_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_l2norm