import torch
import triton
import triton.language as tl
import math
import numpy as np

# Pattern matching function
def pattern(softmax_input, scale_tensor):
    # Pattern matches: softmax(x, dim=1) * y followed by sum(dim=1)
    tmp_0 = torch.nn.functional.softmax(softmax_input, dim=1)
    tmp_2 = tmp_0 * scale_tensor
    tmp_3 = tmp_2.sum(dim=1)
    return tmp_3

# Argument extraction function
def replacement_args(softmax_input, scale_tensor):
    return (softmax_input, scale_tensor)

# Triton kernel for fused softmax * scale + sum
@triton.jit
def fused_softmax_scale_sum_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID determines which row block we're processing
    row_block_idx = tl.program_id(0)
    row_start = row_block_idx * BLOCK_SIZE_M
    
    # Load scale tensor values
    scale_vals = tl.load(scale_ptr + tl.arange(0, n_cols), mask=(tl.arange(0, n_cols) < n_cols))
    
    # Process multiple rows per program (BLOCK_SIZE_M)
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < n_rows
    
    # For each row in the block
    for row_idx in range(BLOCK_SIZE_M):
        if row_start + row_idx >= n_rows:
            break
            
        # Load a full row of input data
        input_row = tl.load(input_ptr + (row_start + row_idx) * n_cols + tl.arange(0, n_cols), 
                           mask=(tl.arange(0, n_cols) < n_cols))
        
        # Compute softmax on the row
        max_val = tl.max(input_row, mask=(tl.arange(0, n_cols) < n_cols))
        shifted_exp = tl.exp(input_row - max_val, mask=(tl.arange(0, n_cols) < n_cols))
        sum_exp = tl.sum(shifted_exp, mask=(tl.arange(0, n_cols) < n_cols))
        softmax_row = shifted_exp / sum_exp
        
        # Compute weighted sum: softmax_row * scale summed
        weighted_sum = tl.sum(softmax_row * scale_vals, mask=(tl.arange(0, n_cols) < n_cols))
        
        # Store the result
        tl.store(output_ptr + (row_start + row_idx), weighted_sum, mask=row_offsets >= row_start)

@torch.fx.wrap
def fused_softmax_scale_sum(input_tensor, scale_tensor):
    n_rows = input_tensor.shape[0]
    n_cols = input_tensor.shape[1]
    
    # Use block sizes that work well for small tensors like [1, 5]
    BLOCK_SIZE_M = 64  # Process 64 rows per program
    BLOCK_SIZE_N = n_cols  # Use full columns for this small case
    
    num_programs = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    output = torch.empty(n_rows, dtype=input_tensor.dtype, device=input_tensor.device)
    
    fused_softmax_scale_sum_kernel[(num_programs,)](
        input_ptr=input_tensor,
        scale_ptr=scale_tensor,
        output_ptr=output,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_softmax_scale_sum