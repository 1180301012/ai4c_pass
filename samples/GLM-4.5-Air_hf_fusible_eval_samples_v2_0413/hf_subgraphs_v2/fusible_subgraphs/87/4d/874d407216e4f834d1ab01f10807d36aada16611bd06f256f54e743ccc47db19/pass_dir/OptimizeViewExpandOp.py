import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, tmp_1):
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, tmp_1):
    return (in_0, tmp_1.shape)

# Triton kernel for optimized view and expand operation
@triton.jit
def optimized_expand_kernel(
    vector_ptr,
    output_ptr,
    target_rows,
    target_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the target matrix
    row_id = tl.program_id(0) % target_rows
    col_id = tl.program_id(0) // target_rows
    
    # Check bounds
    mask = (row_id < target_rows) & (col_id < target_cols)
    
    # Load the vector element (same for all columns)
    vector_val = tl.load(vector_ptr + row_id, mask=mask)
    
    # Store to the target position (broadcast to all columns for this row)
    output_idx = row_id * target_cols + col_id
    tl.store(output_ptr + output_idx, vector_val, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_expand(vector, target_shape):
    target_rows, target_cols = target_shape
    
    # Determine optimal block size
    BLOCK_SIZE = 1024
    total_elements = target_rows * target_cols
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty(target_shape, dtype=vector.dtype, device=vector.device)
    
    optimized_expand_kernel[(num_programs,)](
        vector_ptr=vector,
        output_ptr=output,
        target_rows=target_rows,
        target_cols=target_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_expand