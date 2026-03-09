import torch
import triton
import triton.language as tl

def pattern(x, y):
    """
    Pattern: Complex indexing with new dimensions added
    This reshapes from [rows, cols] to [rows, 1, 1, cols]
    """
    # Direct indexing without intermediate variable
    result = x[:, None, None, :]
    return result

def replacement_args(x, y):
    return (x,)

@triton.jit
def optimized_reshape_kernel(
    input_ptr,
    output_ptr,
    input_rows,
    input_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for reshaping [rows, cols] to [rows, 1, 1, cols] without intermediate tensor
    """
    # Each program handles one row of the original input
    row_idx = tl.program_id(0)
    
    # Since we have new dimensions [1, 1], each original row maps to 1 output row
    # and we need to handle all columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < min(input_cols, BLOCK_SIZE)
    
    # Calculate input and output indices
    input_indices = row_idx * input_cols + col_offsets[col_mask]
    
    # For output, the layout is [rows, 1, 1, cols]
    # Since the new dimensions are size 1, output row = input row, output col = input col
    output_indices = row_idx * (1 * 1 * input_cols) + col_offsets[col_mask]
    
    # Load input data and store directly to output reshaped layout
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
def optimized_reshape_wrapper(input_tensor):
    """
    Wrapper function that launches the optimized reshape kernel
    """
    input_rows, input_cols = input_tensor.shape
    
    # Create output tensor with new dimension layout [rows, 1, 1, cols]
    output = torch.empty(
        input_rows, 1, 1, input_cols,
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )
    
    # Optimal block size
    BLOCK_SIZE = min(1024, input_cols)
    
    # Grid size (one program per input row)
    num_rows = (input_rows + 1)  # ceil division
    
    # Launch kernel
    optimized_reshape_kernel[(num_rows,)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_rows=input_rows,
        input_cols=input_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_reshape_wrapper