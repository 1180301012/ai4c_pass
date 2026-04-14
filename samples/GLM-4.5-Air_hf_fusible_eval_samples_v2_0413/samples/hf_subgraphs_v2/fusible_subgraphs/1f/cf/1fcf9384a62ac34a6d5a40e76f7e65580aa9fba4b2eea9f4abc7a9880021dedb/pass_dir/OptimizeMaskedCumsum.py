import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation in model.py
def pattern(in_0):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_masked_cumsum_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid positions
    mask = col_offset < n_cols
    
    # Load input data
    input_data = tl.load(input_ptr + row_idx * n_cols + col_offset, mask=mask, other=0)
    
    # Create binary mask efficiently
    is_valid = (input_data != 1).to(tl.int32)
    
    # Use simple vectorized approach that's efficient
    cumsum_val = tl.cumsum(is_valid, axis=0)
    result = tl.where(is_valid == 1, cumsum_val + 1, 1)
    
    # Store result
    tl.store(output_ptr + row_idx * n_cols + col_offset, result, mask=mask)

@torch.fx.wrap
def optimized_masked_cumsum(input_tensor):
    n_rows, n_cols = input_tensor.shape
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed for columns
    n_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty((n_rows, n_cols), dtype=torch.int64, device=input_tensor.device)
    
    # Launch kernel
    optimized_masked_cumsum_kernel[(n_rows, n_blocks)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function returns the optimized kernel
def replacement_func():
    return optimized_masked_cumsum