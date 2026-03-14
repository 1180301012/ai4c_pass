import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, slice_dim1, expand_dim0):
    """
    Pattern: Complete forward function optimization
    Matches the entire computation from all observed graphs:
    1. tmp_2 = tmp_1[slice(None, None, None), slice(None, slice_dim1, None)]
    2. tmp_3 = tmp_2.expand(expand_dim0, slice_dim1)  
    3. tmp_4 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
    4. return (tmp_3, tmp_4)
    """
    # Step 1: Slice from second input
    tmp_2 = in_1[slice(None, None, None), slice(None, slice_dim1, None)]
    
    # Step 2: Expand the sliced tensor
    tmp_3 = tmp_2.expand(expand_dim0, slice_dim1)
    
    # Step 3: Add None dimensions to first input
    tmp_4 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    
    # Return both results as in the original function
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1, slice_dim1, expand_dim0):
    return (in_0, in_1, slice_dim1, expand_dim0)

@triton.jit
def complete_forward_kernel(
    in0_ptr,
    in1_ptr,
    out3_ptr,  
    out4_ptr,
    in0_batch,
    in0_seq,
    in1_batch,
    in1_seq,
    out3_batch,
    out3_seq,
    out4_batch,
    out4_seq,
    slice_dim1: tl.constexpr,
    expand_dim0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Complete optimized kernel for the entire forward function"""
    pid = tl.program_id(0)
    
    # Process either result 3 or result 4 based on program ID
    total_elements_out3 = out3_batch * out3_seq
    total_elements_out4 = out4_batch * out4_seq
    
    if pid < triton.cdiv(total_elements_out3, BLOCK_SIZE):
        # Process output 3 (expand result)
        block_offset = pid * BLOCK_SIZE
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements_out3
        
        if tl.any(mask):
            # Convert linear offset to 2D indices for output [expand_dim0, slice_dim1]
            row = offsets // slice_dim1
            col = offsets % slice_dim1
            
            # Valid row check
            row_mask = (row < expand_dim0) & mask
            
            if tl.any(row_mask):
                # Always read first row of input [1,514]
                input_row = 0
                input_col = col[row_mask]
                input_idx = input_row * in1_batch + input_col
                
                output_idx = offsets[row_mask]
                
                input_val = tl.load(in1_ptr + input_idx, other=0)
                tl.store(out3_ptr + output_idx, input_val)
    
    elif pid < triton.cdiv(total_elements_out3 + total_elements_out4, BLOCK_SIZE):
        # Process output 4 (None dimensions result)
        pid_out4 = pid - triton.cdiv(total_elements_out3, BLOCK_SIZE)
        block_offset = pid_out4 * BLOCK_SIZE
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements_out4
        
        if tl.any(mask):
            # Convert linear offset to 2D indices for original input [batch, seq]
            batch_idx = offsets // out4_seq
            seq_idx = offsets % out4_seq
            
            # Valid batch index check  
            batch_mask = (batch_idx < in0_batch) & mask
            
            if tl.any(batch_mask):
                # Input indices (no None dims)
                input_idx = batch_idx[batch_mask] * in0_seq + seq_idx[batch_mask]
                
                # Output indices (with None dims): [batch, 1, 1, seq]
                output_idx = batch_idx[batch_mask] * (1 * 1 * out4_seq) + seq_idx[batch_mask]
                
                input_val = tl.load(in0_ptr + input_idx, other=0)
                tl.store(out4_ptr + output_idx, input_val)

@torch.fx.wrap
def complete_forward_op(in_0, in_1, slice_dim1, expand_dim0):
    """Complete optimized implementation of the forward function"""
    # Get input shapes
    in0_shape = in_0.shape  # [batch, seq]
    in1_shape = in_1.shape  # [batch, seq]
    
    # Output shapes
    out3_shape = (expand_dim0, slice_dim1)  # Expanded result
    out4_shape = (in0_shape[0], 1, 1, in0_shape[1])  # None dimensions result
    
    # Create output tensors
    out3 = torch.empty(out3_shape, dtype=in_1.dtype, device=in_1.device)
    out4 = torch.empty(out4_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Total elements for both outputs
    total_elements_out3 = out3_shape[0] * out3_shape[1]
    total_elements_out4 = out4_shape[0] * out4_shape[3]  # seq_len is at dim 3
    
    # Block size tuning
    BLOCK_SIZE = 1024
    
    # Calculate grid size (programs for both outputs)
    grid_out3 = triton.cdiv(total_elements_out3, BLOCK_SIZE)
    grid_out4 = triton.cdiv(total_elements_out4, BLOCK_SIZE)
    total_grid = grid_out3 + grid_out4
    
    # Launch comprehensive kernel
    complete_forward_kernel[(
        total_grid,
    )](
        in_0,
        in_1,
        out3,
        out4,
        in0_shape[0], in0_shape[1],  # in_0 dimensions
        in1_shape[0], in1_shape[1],  # in_1 dimensions
        out3_shape[0], out3_shape[1],  # out_3 dimensions
        out4_shape[0], out4_shape[3],  # out_4 dimensions (seq_len)
        slice_dim1,
        expand_dim0,
        BLOCK_SIZE
    )
    
    return (out3, out4)

def replacement_func():
    return complete_forward_op