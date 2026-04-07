import torch
import triton
import triton.language as tl

def pattern(tmp_9):
    # Match the sequential max operations and arithmetic
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    
    # Return tmp_13 which is one of the outputs
    return tmp_13

def replacement_args(tmp_9):
    return (tmp_9,)

@triton.jit
def optimized_max_arithmetic_kernel(
    input_ptr,      # tmp_9
    output_ptr,     # tmp_13
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global offsets
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create mask for columns
    col_mask = col_offsets < n_cols
    
    # For max along dim -1 (last dimension), each row is independent
    # So we need to process each row in parallel
    
    # Find the max in this row (along the last dimension)
    row_data = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=col_mask, other=0)
    
    # For max along last dimension, each element is independent
    # In this case, since we're max along dim=-1 with keepdim=True,
    # and the input is (n_rows, n_cols), the result should be (n_rows, 1)
    # Each element in a row gets the max of that row
    
    # Compute max for this row (reduction along column dimension)
    # Since each program handles one row, we need to compute the max within that row
    row_max = tl.max(row_data, axis=0)  # This gives us the max for this row
    
    # The arithmetic operations: +1, -9
    result = row_max + 1 - 9
    
    # Store the result - should be in shape (n_rows, 1)
    if row_idx < n_rows:
        tl.store(output_ptr + row_idx * 1 + 0, result, mask=True)

def compute_row_max(row_data, col_mask, n_cols):
    """Helper function to compute max along a row"""
    # Simple max computation for a row
    max_val = row_data[0] if col_mask[0] else 0
    
    # Loop through elements to find max (this is simplified for demonstration)
    # In a real implementation, we'd use a more efficient reduction
    for i in range(1, n_cols):
        if col_mask[i]:
            if row_data[i] > max_val:
                max_val = row_data[i]
    return max_val

@torch.fx.wrap
def optimized_max_arithmetic(tmp_9):
    """Optimized version of sequential max operations and arithmetic"""
    input_shape = tmp_9.shape
    n_rows = input_shape[0]
    n_cols = input_shape[1]
    
    # Output shape should be (n_rows, 1) due to max(-1, keepdim=True)
    output_shape = (n_rows, 1)
    
    # Create output tensor
    tmp_13 = torch.empty(output_shape, dtype=tmp_9.dtype, device=tmp_9.device)
    
    # For small tensors, use simple CPU computation
    if n_rows * n_cols <= 1024:
        # Simple CPU-based computation for small tensors
        max_vals = tmp_9.max(dim=-1, keepdim=True)[0]
        tmp_13 = max_vals + 1 - 9
        return tmp_13
    
    # Set up grid and launch kernel for larger tensors
    BLOCK_SIZE = min(1024, n_cols)
    n_programs = n_rows
    
    optimized_max_arithmetic_kernel[(n_programs,)](
        input_ptr=tmp_9,
        output_ptr=tmp_13,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_13

def replacement_func():
    return optimized_max_arithmetic