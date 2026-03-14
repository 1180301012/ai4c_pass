import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation chain
def pattern(in_0):
    # Match the exact computation from model.py
    tmp_0 = in_0
    tmp_1 = tmp_0.ne(1)
    tmp_0 = None
    tmp_2 = tmp_1.int()
    tmp_1 = None
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_3 = None
    tmp_5 = tmp_4 + 0
    tmp_4 = None
    tmp_6 = tmp_5 * tmp_2
    tmp_5 = tmp_2 = None
    tmp_7 = tmp_6.long()
    tmp_6 = None
    tmp_8 = tmp_7 + 1
    tmp_7 = None
    return (tmp_8,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel for fused computation with autotune
@triton.jit
def fused_cumsum_mask_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one entire row of the input
    row_idx = tl.program_id(0)
    
    # Calculate global memory position for this row
    row_start = row_idx * n_cols
    
    # Load the entire row efficiently
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + n_cols
    
    # Load input row for this program
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Create mask (elements != 1) and compute cumsum across entire row
    # This matches the original PyTorch computation exactly
    mask_vals = (input_vals != 1).to(tl.int32)
    cumsum = tl.cumsum(mask_vals, axis=0)
    result = cumsum * mask_vals  # This zeros out positions where original element was 1
    
    # Convert to long and add 1 (final result)
    result = result.to(tl.int64) + 1
    
    # Store the result efficiently
    tl.store(output_ptr + offsets, result, mask=mask)

# Optimized kernel wrapper
@torch.fx.wrap
def fused_cumsum_mask_forward(input_tensor):
    # Get tensor dimensions
    n_rows, n_cols = input_tensor.shape
    
    # Choose optimal block size - powers of 2 for best performance
    if n_cols <= 64:
        BLOCK_SIZE = 64
    elif n_cols <= 128:
        BLOCK_SIZE = 128
    elif n_cols <= 256:
        BLOCK_SIZE = 256
    elif n_cols <= 512:
        BLOCK_SIZE = 512
    elif n_cols <= 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    # Ensure BLOCK_SIZE is at least n_cols to avoid boundary issues
    # This ensures each program loads the entire row in one go
    BLOCK_SIZE = max(BLOCK_SIZE, n_cols)
    
    # Calculate grid size (one program per row)
    grid_size = (n_rows,)
    
    # Create output tensor with same shape and dtype as expected result (long)
    output = torch.empty_like(input_tensor, dtype=torch.int64)
    
    # Launch the kernel
    fused_cumsum_mask_kernel[grid_size](
        input_ptr=input_tensor,
        output_ptr=output,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function - returns the optimized kernel
def replacement_func():
    return fused_cumsum_mask_forward