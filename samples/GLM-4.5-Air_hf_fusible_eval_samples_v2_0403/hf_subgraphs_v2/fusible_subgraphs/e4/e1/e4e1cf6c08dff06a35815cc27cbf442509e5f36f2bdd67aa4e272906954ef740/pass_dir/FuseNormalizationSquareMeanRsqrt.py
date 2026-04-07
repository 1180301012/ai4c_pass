import torch
import triton
import triton.language as tl
import math

def pattern(x, scalar):
    """Match the normalization pattern: square -> mean -> epsilon -> rsqrt -> multiply"""
    # x is already float() from the original computation
    tmp_5 = x.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = x * tmp_8
    return tmp_9

def replacement_args(x, scalar):
    return (x, scalar)

@triton.jit
def fused_normalization_kernel(
    x_ptr,
    scalar_ptr,
    out_ptr,
    n_cols,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # Load the scalar (same for all elements)
    scalar = tl.load(scalar_ptr)
    
    # Load the row of x
    x = tl.load(x_ptr + row_idx * n_cols + cols, mask=mask, other=0.0)
    
    # Step 1: Square the values
    x_squared = x * x
    
    # Step 2: Compute mean of the row (by summing and dividing)
    block_sum = tl.sum(x_squared, axis=0)
    total_sum = tl.sum(block_sum)
    mean = total_sum / n_cols
    
    # Step 3: Add epsilon
    mean_plus_epsilon = mean + EPSILON
    
    # Step 4: Compute inverse square root
    inv_sqrt_mean = 1.0 / tl.sqrt(mean_plus_epsilon)
    
    # Step 5: Apply normalization
    out = x * inv_sqrt_mean
    
    # Store the result
    tl.store(out_ptr + row_idx * n_cols + cols, out, mask=mask)

@torch.fx.wrap
def fused_normalization(x, scalar):
    """Fused normalization: square -> mean -> epsilon -> rsqrt -> multiply"""
    # Compute shapes
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Block size for columns
    BLOCK_SIZE = 128
    
    # Launch grid
    n_row_blocks = n_rows
    n_col_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_normalization_kernel[(n_row_blocks, n_col_blocks)](
        x_ptr=x,
        scalar_ptr=scalar,
        out_ptr=out,
        n_cols=n_cols,
        n_rows=n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
        EPSILON=1e-06,
    )
    
    return out

def replacement_func():
    return fused_normalization