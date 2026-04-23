import torch
import triton
import triton.language as tl

# Pattern: match the softmax operation
def pattern(in_0):
    result = in_0.softmax(dim=-1)
    return result

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def triton_softmax_kernel(
    x_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    last_dim_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one "row" along the last dimension
    row_idx = tl.program_id(0)
    
    # Compute offsets for this row
    row_start_ptr = x_ptr + row_idx * last_dim_size
    
    # Create offsets for the last dimension
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < last_dim_size
    
    # Load values with -inf for out-of-bounds
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=float('-inf'))
    
    # Compute softmax
    max_val = tl.max(x, axis=0)
    x_minus_max = x - max_val
    exp_x = tl.exp(x_minus_max)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp
    
    # Store the result
    out_row_start_ptr = out_ptr + row_idx * last_dim_size
    tl.store(out_row_start_ptr + col_offsets, softmax_out, mask=mask)

@torch.fx.wrap
def triton_softmax(x):
    """Optimized softmax using Triton kernel
    
    For input tensor of shape [B, N, N], applies softmax along the last dimension.
    """
    *batch_dims, last_dim_size = x.shape
    
    # Calculate total number of rows to process
    n_rows = 1
    for dim_size in batch_dims:
        n_rows *= dim_size
    
    # Determine BLOCK_SIZE (must be at least last_dim_size for full load)
    BLOCK_SIZE = max(1024, last_dim_size)
    
    out = torch.empty_like(x)
    
    # Launch grid: one program per "row"
    grid = (n_rows,)
    
    triton_softmax_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        last_dim_size=last_dim_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_softmax