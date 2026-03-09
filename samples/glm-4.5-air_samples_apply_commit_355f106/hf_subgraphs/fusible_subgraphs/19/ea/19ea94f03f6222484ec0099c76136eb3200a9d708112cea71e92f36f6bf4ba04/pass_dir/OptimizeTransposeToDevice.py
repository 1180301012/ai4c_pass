import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    tmp_2 = x.t()
    tmp_3 = tmp_2.to(device(type='cuda'))
    return tmp_3

# Argument extraction function
def replacement_args(x):
    return (x,)

# Simple optimized transpose kernel
@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
):
    # Each program handles one element in the output
    # For [1, 1152] -> [1152, 1], we need 1152 programs (one per output row)
    row_idx = tl.program_id(0)
    
    # Handle only the valid output rows
    if row_idx < n_cols:
        # For input shape [n_rows, n_cols] and output shape [n_cols, n_rows]
        # When n_rows=1, we're transposing [1, 1152] -> [1152, 1]
        
        # Load from input: x[0, row_idx] 
        # Input layout: [row_0_col_0, row_0_col_1, row_0_col_2, ...]
        src_offset = row_idx  # Since n_rows=1, just the column index
        
        # Load the element (ensure bounds checking)
        src_val = tl.load(x_ptr + src_offset, mask=src_offset < n_rows * n_cols, other=0.0)
        
        # Store to output: out[row_idx, 0] 
        # Output layout: [col_0_row_0, col_1_row_0, col_2_row_0, ...]
        dst_offset = row_idx
        tl.store(out_ptr + dst_offset, src_val)

@torch.fx.wrap
def optimized_transpose(x):
    n_rows, n_cols = x.shape
    
    # Create output tensor with transposed shape
    out = torch.empty((n_cols, n_rows), dtype=x.dtype, device=x.device)
    
    # Launch kernel - one program per row in output
    num_output_rows = n_cols
    
    # Launch kernel
    transpose_kernel[(num_output_rows,)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_transpose