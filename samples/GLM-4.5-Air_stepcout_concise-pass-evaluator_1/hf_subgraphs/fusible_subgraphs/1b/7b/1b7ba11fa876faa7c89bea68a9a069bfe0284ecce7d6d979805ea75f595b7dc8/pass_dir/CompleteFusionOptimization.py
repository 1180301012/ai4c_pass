import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Complete pattern matching for the entire computation:
    tmp_0 = in_0
    tmp_1 = in_1  
    tmp_2 = tmp_1[:, :N]  # Slice from second input
    tmp_3 = tmp_2.expand(M, N)  # Expand sliced tensor
    tmp_4 = tmp_0[:, None, None, :]  # Reshape first input
    return (tmp_3, tmp_4)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    
    # Common slice size pattern - slice to first dimension of in_0
    N = in_0.shape[1]  # slice to this size
    
    tmp_2 = tmp_1[slice(None, None, None), slice(None, N, None)]
    M = in_0.shape[0]  # expand to this size
    tmp_3 = tmp_2.expand(M, N)
    
    tmp_4 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def complete_fusion_kernel(
    input1_ptr,      # tmp_1 (token type ids)
    input0_ptr,      # tmp_0 (attention mask) 
    output1_ptr,     # tmp_3 (expanded tensor)
    output0_ptr,     # tmp_4 (reshaped tensor)
    input0_dim0,
    input0_dim1,
    input1_dim0,
    input1_dim1,
    slice_size,
    expand_dim0,
    expand_dim1,
    BLOCK_SIZE: tl.constexpr,
):
    """Complete fusion kernel that processes both outputs in one kernel launch"""
    pid = tl.program_id(0)
    
    # Process the expanded tensor (output1)
    if pid < expand_dim0:
        row_idx = pid
        col_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < expand_dim1
        
        # Determine values for expanded tensor
        if row_idx < input1_dim0 and col_offsets < min(slice_size, input1_dim1):
            # Copy from input1 where valid
            input_row_idx = row_idx
            input_col = col_offsets
            value1 = tl.load(input1_ptr + input_row_idx * input1_dim1 + input_col, mask=mask)
        else:
            # Broadcast from input1
            value1 = tl.load(input1_ptr + (row_idx % input1_dim0) * input1_dim1 + col_offsets, mask=mask)
        
        # Store to expanded output
        output1_idx = row_idx * expand_dim1 + col_offsets
        tl.store(output1_ptr + output1_idx, value1, mask=mask)
    
    # Process the reshaped tensor (output0) - interleave with output1
    pid_offset = expand_dim0
    if pid >= pid_offset:
        adjusted_pid = pid - pid_offset
        if adjusted_pid < input0_dim0:
            row_idx = adjusted_pid
            col_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < input0_dim1
            
            # Load input0 data
            input_idx = row_idx * input0_dim1 + col_offsets
            value0 = tl.load(input0_ptr + input_idx, mask=mask, other=0)
            
            # Store to reshaped output (accounting for None dimensions)
            # Output shape: (input0_dim0, 1, 1, input0_dim1)
            # Mapping: (row_idx, 0, 0, col_idx) -> linear index: row_idx * input0_dim1 + col_idx
            output0_idx = row_idx * input0_dim1 + col_offsets
            tl.store(output0_ptr + output0_idx, value0, mask=mask)

@torch.fx.wrap
def optimized_complete_fusion(in_0, in_1):
    """Complete fusion function that processes both outputs with maximum efficiency"""
    # Get input shapes
    input0_shape = in_0.shape
    input1_shape = in_1.shape
    
    input0_dim0 = input0_shape[0]
    input0_dim1 = input0_shape[1]
    input1_dim0 = input1_shape[0]
    input1_dim1 = input1_shape[1]
    
    # Calculate dimensions
    slice_size = min(input1_dim1, input0_dim1)  # Slice to min dimension
    expand_dim0 = input0_dim0
    expand_dim1 = slice_size
    
    # Allocate output tensors
    output1_shape = (expand_dim0, expand_dim1)
    output0_shape = (input0_dim0, 1, 1, input0_dim1)
    
    output1 = torch.empty(output1_shape, dtype=in_1.dtype, device=in_1.device)
    output0 = torch.empty(output0_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Calculate launch configuration
    BLOCK_SIZE = 1024  # Optimal for most GPUs
    
    # Total programs needed for both outputs
    total_programs = expand_dim0 + input0_dim0
    
    # Check if we need to adjust for better load balancing
    if expand_dim0 > 0 and input0_dim0 > 0:
        n_programs = max((expand_dim0 + BLOCK_SIZE - 1) // BLOCK_SIZE,
                         (input0_dim0 + BLOCK_SIZE - 1) // BLOCK_SIZE)
        n_programs = max(n_programs, 1)  # At least 1 program
    elif expand_dim0 > 0:
        n_programs = (expand_dim0 + BLOCK_SIZE - 1) // BLOCK_SIZE
    elif input0_dim0 > 0:
        n_programs = (input0_dim0 + BLOCK_SIZE - 1) // BLOCK_SIZE
    else:
        n_programs = 1
    
    # Launch the fusion kernel
    complete_fusion_kernel[(n_programs,)](
        input1_ptr=in_1,
        input0_ptr=in_0,
        output1_ptr=output1,
        output0_ptr=output0,
        input0_dim0=input0_dim0,
        input0_dim1=input0_dim1,
        input1_dim0=input1_dim0,
        input1_dim1=input1_dim1,
        slice_size=slice_size,
        expand_dim0=expand_dim0,
        expand_dim1=expand_dim1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (output1, output0)

def replacement_func():
    return optimized_complete_fusion