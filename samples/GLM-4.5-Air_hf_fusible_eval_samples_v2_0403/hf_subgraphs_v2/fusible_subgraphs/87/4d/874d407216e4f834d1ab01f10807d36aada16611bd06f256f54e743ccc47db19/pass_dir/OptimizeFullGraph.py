import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, shape_0, shape_1):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_0 = None
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_2 = None
    tmp_4 = tmp_1.new_zeros((shape_0, shape_1))
    return (tmp_3, tmp_4, tmp_1)

def replacement_args(in_0, in_1, in_2, shape_0, shape_1):
    return (in_0, in_1, in_2, shape_0, shape_1)

@triton.jit
def optimized_full_graph_kernel(
    in_0_ptr,
    in_1_ptr, 
    in_2_ptr,
    out_0_ptr,
    out_1_ptr,
    out_2_ptr,
    in_0_size,
    in_1_size,
    in_2_shape_0,
    in_2_shape_1,
    output_shape_0,
    output_shape_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row of the computation
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    
    # Mask for column bounds
    col_mask = col_idx < in_2_shape_1
    
    # 1. Compute element-wise multiplication (tmp_1 = in_1.view(-1, 1) * in_2)
    if row_idx < in_1_size:
        # Load the corresponding element from in_1 (broadcasted)
        in_1_val = tl.load(in_1_ptr + row_idx)
        
        # Load elements from in_2 for this row
        in_2_offsets = row_idx * in_2_shape_1 + col_idx
        in_2_vals = tl.load(in_2_ptr + in_2_offsets, mask=col_mask, other=0.0)
        
        # Perform multiplication (broadcasting)
        tmp_1_vals = in_1_val * in_2_vals
        
        # Store tmp_1 result (element-wise multiplication)
        tmp_1_offsets = row_idx * in_2_shape_1 + col_idx
        tl.store(out_2_ptr + tmp_1_offsets, tmp_1_vals, mask=col_mask)
    
    # 2. Compute expanded tensor (tmp_3 = in_0.view(-1, 1).expand_as(tmp_1))
    if row_idx < in_0_size:
        # Load the corresponding element from in_0
        in_0_val = tl.load(in_0_ptr + row_idx)
        
        # Broadcast this value across all columns for this row
        broadcast_vals = tl.full((BLOCK_SIZE,), in_0_val, dtype=tl.float32)
        
        # Store the broadcast values for tmp_3
        out_0_offsets = row_idx * in_2_shape_1 + col_idx
        tl.store(out_0_ptr + out_0_offsets, broadcast_vals, mask=col_mask)
    
    # 3. Compute zeros tensor (tmp_4)
    # For zeros tensor, we only need to handle the rows if we're in the valid range
    if row_idx < output_shape_0:
        # Calculate column indices
        col_offsets = col_idx
        col_mask_zeros = col_offsets < output_shape_1
        
        # Store zeros
        zeros_offsets = row_idx * output_shape_1 + col_offsets
        tl.store(out_1_ptr + zeros_offsets, 0.0, mask=col_mask_zeros)

@torch.fx.wrap  
def optimized_full_graph(in_0, in_1, in_2, shape_0, shape_1):
    # Get tensor shapes and properties
    in_2_shape = in_2.shape
    
    # Create output tensors with correct dtypes
    tmp_3 = torch.empty_like(in_2)  # expand_as result
    tmp_4 = torch.zeros((shape_0, shape_1), dtype=in_1.dtype, device=in_1.device)  # zeros tensor
    tmp_1 = torch.empty_like(in_2)  # multiplication result
    
    # Choose appropriate block size
    BLOCK_SIZE = 256
    
    # Calculate grid dimensions
    grid = (max(in_0.size(0), in_1.size(0), in_2_shape[0], shape_0),)
    
    # Launch the optimized kernel
    optimized_full_graph_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_0_ptr=tmp_3,
        out_1_ptr=tmp_4,
        out_2_ptr=tmp_1,
        in_0_size=in_0.size(0),
        in_1_size=in_1.size(0),
        in_2_shape_0=in_2_shape[0],
        in_2_shape_1=in_2_shape[1],
        output_shape_0=shape_0,
        output_shape_1=shape_1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (tmp_3, tmp_4, tmp_1)

def replacement_func():
    return optimized_full_graph