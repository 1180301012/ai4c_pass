import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    tmp_11 = tmp_0.tensor_split(2, -1)
    tmp_12 = tmp_11[0]
    tmp_13 = tmp_11[1]
    return tmp_12, tmp_13

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def optimized_split_kernel_even(
    input_ptr,
    output1_ptr,
    output2_ptr,
    batch_size,
    other_dims,
    split_dim,
    total_split_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * other_dims * total_split_dim
    
    # Calculate position in the split dimension
    elements_per_row = other_dims * total_split_dim
    row_idx = offsets // elements_per_row
    pos_in_row = offsets % elements_per_row
    
    # Which half of the split dimension
    split_half = total_split_dim // 2
    dim_pos = pos_in_row // other_dims
    offset_in_dim = pos_in_row % other_dims
    
    # Determine which output this belongs to
    if dim_pos < split_half:
        # Belongs to first output
        output1_offset = row_idx * other_dims * split_half + (dim_pos * other_dims + offset_in_dim)
        output2_data = 0.0  # Not used for first output
        output1_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        # Store in first output
        tl.store(output1_ptr + output1_offset, output1_data, mask=mask)
    else:
        # Belongs to second output
        output2_offset = row_idx * other_dims * split_half + ((dim_pos - split_half) * other_dims + offset_in_dim)
        output1_data = 0.0  # Not used for second output  
        output2_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        # Store in second output
        tl.store(output2_ptr + output2_offset, output2_data, mask=mask)

@torch.fx.wrap
def optimized_tensor_split_even(input_tensor):
    # Get input shape
    input_shape = input_tensor.shape
    split_dim = -1  # Split on last dimension
    total_split_dim = input_shape[split_dim]
    split_half = total_split_dim // 2
    
    # Calculate other dimensions (all dimensions except split_dim)
    other_dims = 1
    for i, dim_size in enumerate(input_shape):
        if i != split_dim:
            other_dims *= dim_size
    
    batch_size = input_shape[split_dim] // 2
    
    # Create output tensors
    output1_shape = list(input_shape).copy()
    output1_shape[split_dim] = split_half
    output1 = torch.empty(output1_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    output2_shape = list(input_shape).copy()
    output2_shape[split_dim] = split_half
    output2 = torch.empty(output2_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    total_elements = input_tensor.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_split_kernel_even[(num_programs,)](
        input_ptr=input_tensor,
        output1_ptr=output1,
        output2_ptr=output2,
        batch_size=batch_size,
        other_dims=other_dims,
        split_dim=split_half,
        total_split_dim=total_split_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output2, output1  # Return in reverse order to match original pattern

def replacement_func():
    return optimized_tensor_split_even