import torch
import triton
import triton.language as tl

def pattern(x, y):
    """
    Pattern: Simple slice operation (first step)
    tmp_2 = x[slice(None, None, None), slice(None, 7, None)]
    """
    # Simple slice operation
    result = x[:, :7]
    return result

def replacement_args(x, y):
    # For simple slice pattern, just return the tensor and slice info
    slice_dim = 7
    return (x, slice_dim)

@triton.jit
def fused_slice_expand_kernel(
    input_ptr,
    output_ptr,
    input_rows,
    input_cols,
    output_rows,
    output_cols,
    slice_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that directly computes expanded result without intermediate tensor
    Each program handles one row of the output
    """
    row_idx = tl.program_id(0)
    
    # Calculate the starting position in the input for this row
    input_base = row_idx * slice_dim
    
    # For each column in the output row (limited by slice_dim)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < min(slice_dim, output_cols)
    
    # Load data from input (caching across row broadcasts)
    input_col_offsets = col_offsets[col_mask]
    input_indices = input_base + input_col_offsets
    input_data = tl.load(
        input_ptr + input_indices,
        mask=input_indices < input_rows * slice_dim,
        other=0
    )
    
    # Store to output - each input element is broadcasted across all rows
    output_indices = row_idx * output_cols + input_col_offsets
    tl.store(
        output_ptr + output_indices,
        input_data,
        mask=col_mask
    )

@torch.fx.wrap
def optimized_slice_wrapper(input_tensor, slice_dim):
    """
    Wrapper function that launches the optimized slice kernel
    """
    input_rows, input_cols = input_tensor.shape
    
    # Create output tensor
    output = torch.empty(
        input_rows, slice_dim,
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )
    
    # Optimal block size
    BLOCK_SIZE = min(1024, slice_dim)
    
    # Grid size (one program per input row)
    num_rows = (input_rows + 1)  # ceil division
    
    # Launch kernel
    fused_slice_expand_kernel[(num_rows,)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_rows=input_rows,
        input_cols=input_cols,
        output_rows=input_rows,
        output_cols=slice_dim,
        slice_dim=slice_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_slice_wrapper