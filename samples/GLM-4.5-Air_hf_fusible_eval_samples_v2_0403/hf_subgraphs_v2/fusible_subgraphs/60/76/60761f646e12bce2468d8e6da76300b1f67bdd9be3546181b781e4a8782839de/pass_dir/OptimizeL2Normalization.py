import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Match the L2 normalization pattern"""
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return tmp_1

def replacement_args(in_0):
    """Extract the input tensor argument"""
    return (in_0,)

@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for L2 normalization along dimension 1"""
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # Compute start pointer for this row
    row_start_ptr = x_ptr + row_idx * n_cols
    
    # Load the entire row
    row = tl.load(row_start_ptr + tl.arange(0, n_cols), mask=tl.arange(0, n_cols) < n_cols, other=0.0)
    
    # Compute L2 norm squared
    norm_squared = tl.sum(row * row)
    
    # Add epsilon for numerical stability and compute norm
    norm = tl.sqrt(norm_squared + eps)
    
    # Normalize the row
    normalized_row = row / norm
    
    # Store the result
    out_start_ptr = out_ptr + row_idx * n_cols
    tl.store(out_start_ptr + tl.arange(0, n_cols), normalized_row, mask=tl.arange(0, n_cols) < n_cols)

@torch.fx.wrap
def triton_l2_normalize(x, eps=1e-12):
    """High-performance L2 normalization using Triton"""
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    # Choose block size based on feature dimension for optimal performance
    BLOCK_SIZE = min(1024, n_cols)
    
    # Calculate grid size (one program per row)
    grid = (n_rows,)
    
    # Launch the kernel
    l2_normalize_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized L2 normalization function"""
    return triton_l2_normalize