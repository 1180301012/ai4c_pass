import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Original pattern: unsqueeze followed by expand
    tmp_5 = x.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    return tmp_6

def replacement_args(x, y):
    return (x, y)

@triton.jit
def optimized_expand_kernel(
    input_ptr,
    output_ptr,
    input_rows,
    input_cols,
    output_rows,
    output_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1) 
    col_idx = tl.program_id(2)
    
    # Calculate offset for this position
    output_offset = batch_idx * output_rows * output_cols + row_idx * output_cols + col_idx
    
    # Create mask for boundary conditions
    mask = (row_idx < input_rows) & (col_idx < input_cols) & (batch_idx < output_rows)
    
    if mask:
        # For expand operation: we copy from input position (0, row_idx, col_idx)
        # to all positions (batch_idx, row_idx, col_idx) where batch_idx < output_rows
        input_offset = row_idx * input_cols + col_idx
        input_val = tl.load(input_ptr + input_offset, other=0.0)
        tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def optimized_unsqueeze_expand(x, y):
    input_shape = x.shape
    
    # The original sequence: x.unsqueeze(0).expand(3, -1, -1)
    # This expands a 2D tensor [rows, cols] to [3, rows, cols]
    
    if len(input_shape) == 2:
        input_rows, input_cols = input_shape
        output_shape = (3, input_rows, input_cols)
    elif len(input_shape) == 1:
        input_size = input_shape[0]
        output_shape = (3, 1, input_size)
        # Reshape 1D to 2D first
        x_2d = x.reshape(1, -1)
        input_rows, input_cols = 1, input_size
        input_ptr = x_2d
    else:
        # Fallback for other shapes
        return x.unsqueeze(0).expand(3, -1, -1)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    if input_rows > 0 and input_cols > 0:
        # Calculate grid dimensions
        batch_dim = output_shape[0]
        BLOCK_SIZE = 32  # Optimal block size for 3D expand
        
        # Use Triton kernel for optimized expansion
        optimized_expand_kernel[
            (batch_dim, (input_rows + BLOCK_SIZE - 1) // BLOCK_SIZE, (input_cols + BLOCK_SIZE - 1) // BLOCK_SIZE),
        ](
            input_ptr=x if len(input_shape) == 2 else x_2d,
            output_ptr=output,
            input_rows=input_rows,
            input_cols=input_cols,
            output_rows=batch_dim,
            output_cols=input_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Handle edge cases (empty tensors)
        if len(input_shape) == 2:
            output = x.unsqueeze(0).expand(3, -1, -1)
        else:
            output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    return output

def replacement_func():
    return optimized_unsqueeze_expand