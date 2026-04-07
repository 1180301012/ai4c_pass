import torch
import triton
import triton.language as tl

def pattern(in_0, tmp_1):
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    return tmp_3

def replacement_args(in_0, tmp_1):
    return (in_0, tmp_1)

@triton.jit
def optimized_expand_kernel(
    in_0_ptr,
    tmp_1_ptr,
    out_ptr,
    in_0_size,
    tmp_1_shape_0,
    tmp_1_shape_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    
    # Calculate 2D indices for the output
    out_offsets = row_idx * tmp_1_shape_1 + col_idx
    mask = col_idx < tmp_1_shape_1
    
    # The key insight: expand_as with view(-1, 1) means we broadcast each element of in_0
    # across a row of size tmp_1_shape_1
    if row_idx < in_0_size:
        # Load the corresponding element from in_0
        in_0_val = tl.load(in_0_ptr + row_idx)
        
        # Broadcast this value across all columns for this row
        # Create a tensor with the same value repeated across columns
        broadcast_vals = tl.full((BLOCK_SIZE,), in_0_val, dtype=tl.float32)
        
        # Store the broadcast values
        tl.store(out_ptr + out_offsets, broadcast_vals, mask=mask)

@torch.fx.wrap
def optimized_expand(in_0, tmp_1):
    # Get the shape of tmp_1 for expansion
    tmp_1_shape = tmp_1.shape
    
    # Calculate total elements needed
    total_elements = tmp_1_shape[0] * tmp_1_shape[1]
    
    # Block size for the computation
    BLOCK_SIZE = 128
    
    # Create output tensor
    out = torch.empty_like(tmp_1)
    
    # Launch kernel
    grid = (tmp_1_shape[0],)
    
    optimized_expand_kernel[grid](
        in_0_ptr=in_0,
        tmp_1_ptr=tmp_1,
        out_ptr=out,
        in_0_size=in_0.size(0),
        tmp_1_shape_0=tmp_1_shape[0],
        tmp_1_shape_1=tmp_1_shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_expand