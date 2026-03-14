import torch
import triton
import triton.language as tl

def pattern(tmp_7):
    """Fuses the sequential max operations: max(0) followed by max(-1, keepdim=True)"""
    tmp_8 = tmp_7.max(0, keepdim=False)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_9.max(-1, keepdim=True)
    tmp_11 = tmp_10[0]
    return tmp_9, tmp_11

def replacement_args(tmp_7):
    return (tmp_7,)

@triton.jit
def fused_max_kernel(
    input_ptr,
    intermediate_ptr,
    final_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    """Fused kernel that computes max over dim 0 and then max over last dim"""
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Load the entire column for max over dim 0
    col_offset = col_idx * n_rows
    offsets = col_offset + tl.arange(0, BLOCK_SIZE_X)
    mask = offsets < n_rows * n_cols
    
    # Load column data for max over dim 0
    column_data = tl.load(input_ptr + offsets, mask=mask, other=-1e20)
    max_val = tl.max(column_data)
    
    # Store intermediate result (max over dim 0)
    intermediate_ptr[col_idx] = max_val
    
    # For the final max over last dimension, find global max of column results
    if row_idx == 0:  # Only one thread needed per column for final max
        final_ptr[col_idx] = max_val

@torch.fx.wrap
def fused_max_operations(tmp_7):
    n_rows, n_cols = tmp_7.shape
    
    # Create output tensors
    intermediate_out = torch.empty(n_cols, dtype=tmp_7.dtype, device=tmp_7.device)
    final_out = torch.empty(n_cols, dtype=tmp_7.dtype, device=tmp_7.device)
    
    # Set up grid dimensions
    block_size_x = 1024
    block_size_y = 1
    
    # Number of programs needed
    n_programs_x = (n_rows + block_size_x - 1) // block_size_x
    grid_x = n_programs_x
    grid_y = n_cols
    
    fused_max_kernel[(grid_x, grid_y)](
        input_ptr=tmp_7,
        intermediate_ptr=intermediate_out,
        final_ptr=final_out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )
    
    return intermediate_out, final_out

def replacement_func():
    return fused_max_operations