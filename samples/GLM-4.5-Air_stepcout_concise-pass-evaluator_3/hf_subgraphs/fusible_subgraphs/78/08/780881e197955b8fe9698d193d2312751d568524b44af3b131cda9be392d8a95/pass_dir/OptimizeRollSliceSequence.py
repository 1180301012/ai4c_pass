import torch
import triton
import triton.language as tl

def pattern(in_3, in_2):
    """Optimize the roll + slice + reshape sequence"""
    # This pattern is for the first graph: 133->133, 96 channels, slice to 128x128
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 133, 133, 96)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 16384, 96)
    tmp_8 = in_2 + tmp_7
    return tmp_8

def replacement_args(in_3, in_2):
    return (in_3, in_2)

@triton.jit
def optimized_roll_kernel(
    input_ptr,
    output_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that combines roll, slice, and reshape operations"""
    pid = tl.program_id(0)
    total_elements = output_height * output_width * channels
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate output indices in output space
    spatial_idx = offsets // channels
    channel_idx = offsets % channels
    
    # Convert to 2D spatial coordinates (output space: 0-127 for 128x128)
    h_out = spatial_idx // output_width  # [0, 127]
    w_out = spatial_idx % output_width   # [0, 127]
    
    # Convert to input coordinates (accounting for roll and slicing)
    # Input is 133x133, output is 128x128 slice starting from (3,3)
    h_in = h_out + 3  # [3, 130] 
    w_in = w_out + 3  # [3, 130]
    
    # Check if indices are within input bounds (133x133)
    h_valid = h_in < input_height
    w_valid = w_in < input_width
    valid_mask = h_valid & w_valid & mask
    
    # Calculate 1D index in input tensor
    input_index = h_in * input_width * channels + w_in * channels + channel_idx
    
    # Load from input and store to output (only valid indices)
    input_val = tl.load(input_ptr + input_index, mask=valid_mask, other=0.0)
    tl.store(output_ptr + offsets, input_val, mask=mask)

@torch.fx.wrap
def optimized_roll_slice_reshape(in_3, in_2):
    """Wrapper for the optimized kernel operation"""
    # For first case: 133x133 input, slice to 128x128, 96 channels
    input_shape = in_3.shape
    
    if len(input_shape) == 6:
        # Handle 6D input: [1, 19, 7, 19, 7, 96] -> reshape to [-1, 133, 133, 96]
        batch, d1, d2, d3, d4, channels = input_shape
        # We need to calculate the proper dimensions after view(-1, 133, 133, 96)
        # The actual input to kernel should be the reshaped tensor
        reshaped = in_3.view(-1, 133, 133, channels)
        input_tensor = reshaped.reshape(-1, 133 * 133 * channels)
        input_height, input_width = 133, 133
    else:
        input_tensor = in_3.reshape(-1)  # Flatten the input
        input_height, input_width = 133, 133  # Fallback for 4D input
    
    # Output parameters for 128x128 slice
    output_height, output_width = 128, 128
    channels = 96
    batch = 1
    output_spatial = output_height * output_width
    
    # Create output tensor
    output_shape = (batch, output_spatial, channels)
    result = torch.empty(output_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel
    n_elements = output_spatial * channels
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    optimized_roll_kernel[grid_size](
        input_tensor,
        result,
        input_height,
        input_width,
        output_height,
        output_width,
        channels,
        BLOCK_SIZE
    )
    
    # Add the in_2 tensor
    return in_2 + result

def replacement_func():
    return optimized_roll_slice_reshape