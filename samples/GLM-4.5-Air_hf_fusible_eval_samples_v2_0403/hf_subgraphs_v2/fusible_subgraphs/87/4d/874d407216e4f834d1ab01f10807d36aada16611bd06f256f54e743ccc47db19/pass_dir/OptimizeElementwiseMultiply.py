import torch
import triton
import triton.language as tl

def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1

def replacement_args(in_1, in_2):
    return (in_1, in_2)

@triton.jit
def optimized_multiply_kernel(
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    in_1_size,
    in_2_shape_0,
    in_2_shape_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row of the output
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual column indices (handle last block)
    actual_col_idx = col_idx
    col_mask = actual_col_idx < in_2_shape_1
    
    # Broadcast in_1 to match in_2 shape: [in_1_size, 1] * [in_2_shape_0, in_2_shape_1]
    # Each element in in_1 is broadcast across all columns for its row
    if row_idx < in_1_size:
        # Load the corresponding element from in_1
        in_1_val = tl.load(in_1_ptr + row_idx)
        
        # Load elements from in_2 for this row
        in_2_offsets = row_idx * in_2_shape_1 + actual_col_idx
        in_2_vals = tl.load(in_2_ptr + in_2_offsets, mask=col_mask, other=0.0)
        
        # Perform multiplication (broadcasting)
        out_vals = in_1_val * in_2_vals
        
        # Store results
        out_offsets = row_idx * in_2_shape_1 + actual_col_idx
        tl.store(out_ptr + out_offsets, out_vals, mask=col_mask)

@torch.fx.wrap
def optimized_multiply(in_1, in_2):
    # Get shapes
    in_2_shape = in_2.shape
    in_1_size = in_1.size(0)
    
    # Block size for columns
    BLOCK_SIZE = 256
    
    # Create output tensor
    out = torch.empty_like(in_2)
    
    # Launch kernel
    grid = (in_2_shape[0],)
    
    optimized_multiply_kernel[grid](
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        in_1_size=in_1_size,
        in_2_shape_0=in_2_shape[0],
        in_2_shape_1=in_2_shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_multiply