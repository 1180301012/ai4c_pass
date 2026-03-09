import torch
import triton
import triton.language as tl

def pattern(x, y):
    """
    Pattern: Simple slicing operation
    """
    # Direct slice without intermediate variable
    result = x[:, 0:128]
    return result

def replacement_args(x, y):
    # For the pattern we're matching, slice from 0 to 128
    slice_start, slice_end = 0, 128
    return (x, slice_start, slice_end)

@triton.jit
def optimized_slice_kernel(
    input_ptr,
    output_ptr,
    input_rows,
    input_cols,
    output_rows,
    output_cols,
    slice_start,
    slice_end,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for direct slicing without intermediate tensor creation
    """
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate slice dimensions
    slice_dim = slice_end - slice_start
    
    # Column offsets within this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < min(slice_dim, output_cols)
    input_col_offsets = col_offsets[col_mask] + slice_start
    
    # Calculate memory indices
    input_indices = row_idx * input_cols + input_col_offsets
    output_indices = row_idx * output_cols + col_offsets[col_mask]
    
    # Load input data directly and store to output
    input_data = tl.load(
        input_ptr + input_indices,
        mask=input_indices < input_rows * input_cols,
        other=0
    )
    
    tl.store(
        output_ptr + output_indices,
        input_data,
        mask=col_mask
    )

@torch.fx.wrap
def optimized_slice_wrapper(input_tensor, slice_start, slice_end):
    """
    Wrapper function that launches the optimized slicing kernel
    """
    input_rows, input_cols = input_tensor.shape
    slice_dim = slice_end - slice_start
    
    # Create output tensor
    output = torch.empty(
        input_rows, slice_dim,
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )
    
    # Optimal block size
    BLOCK_SIZE = min(1024, slice_dim)
    
    # Grid size (one program per row)
    num_rows = (input_rows + 1)  # ceil division
    
    # Launch kernel
    optimized_slice_kernel[(num_rows,)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_rows=input_rows,
        input_cols=input_cols,
        output_rows=input_rows,
        output_cols=slice_dim,
        slice_start=slice_start,
        slice_end=slice_end,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_slice_wrapper