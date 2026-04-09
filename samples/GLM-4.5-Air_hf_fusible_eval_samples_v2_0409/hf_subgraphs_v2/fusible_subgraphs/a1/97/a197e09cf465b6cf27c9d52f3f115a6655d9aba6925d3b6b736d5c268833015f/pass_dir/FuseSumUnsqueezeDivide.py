import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    return (tmp_0,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def sum_last_dim_kernel(
    x_ptr,
    output_ptr,
    n_channels,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output tensor
    # Output shape is [batch_size, n_channels, n_rows]
    col_idx = tl.program_id(0)  # Which column to sum
    idx = tl.program_id(1)      # Which channel/position in remaining dims
    
    # Calculate which input column this corresponds to
    output_offset = idx * n_rows + col_idx
    base_offset = idx * n_rows * n_cols + col_idx
    
    # Load the column data
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < base_offset + n_rows
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum along the column (last dimension)
    col_sum = tl.sum(x)
    
    # Store the sum result
    output_idx = idx * n_rows + col_idx
    tl.store(output_ptr + output_idx, col_sum)

@torch.fx.wrap
def sum_last_dim_triton(x):
    # Get input dimensions
    batch_size, n_channels, n_rows, n_cols = x.shape
    
    # Output tensor for sum - shape should be [batch_size, n_channels, n_rows]
    output_shape = (batch_size, n_channels, n_rows)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Use power of 2 block size for arange compatibility
    BLOCK_SIZE = 128  # Power of 2, reasonable size for performance
    if BLOCK_SIZE > n_rows:
        BLOCK_SIZE = 64  # Smaller power of 2 if needed
    
    # Create grid: (n_cols, batch_size * n_channels)
    n_total_cols = n_cols
    n_groups = batch_size * n_channels
    grid = lambda meta: (n_total_cols, n_groups)  # (column_idx, group_idx)
    
    # Launch kernel
    sum_last_dim_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_channels=n_groups,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return sum_last_dim_triton