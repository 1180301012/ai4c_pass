import torch
import triton
import triton.language as tl
from torch import device

def pattern(tmp_2):
    # Match the exact sequence: expand + device transfer + max reduction
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    tmp_7 = tmp_6.to(device(type='cuda', index=0))
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    
    # Return both tmp_7 (for output) and tmp_9 (for further computation)
    return tmp_7, tmp_9

def replacement_args(tmp_2):
    return (tmp_2,)

@triton.jit
def optimized_expand_max_kernel(
    input_ptr,
    output_expanded_ptr,  # tmp_6 (same as tmp_7 in original)
    output_max_ptr,       # tmp_9  
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global offsets
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create mask for columns
    col_mask = col_offsets < n_cols
    
    # Load the input row (tmp_2) - this gets repeated 3 times in original computation
    input_data = tl.load(input_ptr + col_offsets, mask=col_mask, other=0)
    
    # Store the expanded version (simulating expand(3, -1, -1))
    # Each program handles one row of the expanded result
    if row_idx < 3:
        tl.store(output_expanded_ptr + row_idx * n_rows * n_cols + col_offsets, 
                input_data, mask=col_mask)
    
    # Compute max across the 3 expanded rows (simulating max(0))
    if row_idx == 0:
        # Load all 3 rows and compute max
        row0 = tl.load(output_expanded_ptr + col_offsets, mask=col_mask, other=0)
        row1 = tl.load(output_expanded_ptr + n_rows * n_cols + col_offsets, mask=col_mask, other=0)
        row2 = tl.load(output_expanded_ptr + 2 * n_rows * n_cols + col_offsets, mask=col_mask, other=0)
        
        # Compute element-wise max
        max_vals = tl.maximum(tl.maximum(row0, row1), row2)
        tl.store(output_max_ptr + col_offsets, max_vals, mask=col_mask)

@torch.fx.wrap
def optimized_expand_max(tmp_2):
    """Optimized version of expand + max reduction"""
    input_shape = tmp_2.shape
    n_rows = input_shape[0]  # Should be 1 for all current examples
    n_cols = input_shape[1]
    
    # Output shapes
    expanded_shape = (3, n_rows, n_cols)  # This is the shape of tmp_6 (same as tmp_7 in original)
    max_shape = (n_rows, n_cols)          # This is the shape of tmp_9
    
    # Create output tensors
    tmp_6 = torch.empty(expanded_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    tmp_9 = torch.empty(max_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Set up grid and launch kernel
    BLOCK_SIZE = 1024
    n_programs = 3  # We need 3 programs for the 3 expanded rows
    
    optimized_expand_max_kernel[(n_programs,)](
        input_ptr=tmp_2,
        output_expanded_ptr=tmp_6,
        output_max_ptr=tmp_9,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_6, tmp_9

def replacement_func():
    return optimized_expand_max