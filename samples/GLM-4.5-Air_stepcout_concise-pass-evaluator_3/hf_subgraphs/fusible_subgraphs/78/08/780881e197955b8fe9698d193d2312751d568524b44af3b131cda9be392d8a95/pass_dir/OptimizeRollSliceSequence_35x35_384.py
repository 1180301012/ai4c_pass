import torch
import triton
import triton.language as tl

def pattern(in_3, in_2):
    """Optimize the roll + slice + reshape sequence for 35x35 spatial dimensions"""
    # This pattern is for the third graph: 35->35, 384 channels, slice to 32x32
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    tmp_8 = in_2 + tmp_7
    return tmp_8

def replacement_args(in_3, in_2):
    return (in_3, in_2)

@triton.jit
def optimized_roll_kernel_35x35_384(
    input_ptr,
    output_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for 32x32 output with 384 channels"""
    pid = tl.program_id(0)
    total_elements = output_height * output_width * channels
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate output indices in output space
    spatial_idx = offsets // channels
    channel_idx = offsets % channels
    
    # Convert to 2D spatial coordinates (output space: 0-31 for 32x32)
    h_out = spatial_idx // output_width  # [0, 31]
    w_out = spatial_idx % output_width   # [0, 31]
    
    # Convert to input coordinates (accounting for roll and slicing)
    # Input is 35x35, output is 32x32 slice starting from (3,3)
    h_in = h_out + 3  # [3, 34] 
    w_in = w_out + 3  # [3, 34]
    
    # Check if indices are within input bounds (35x35)
    h_valid = h_in < input_height
    w_valid = w_in < input_width
    valid_mask = h_valid & w_valid & mask
    
    # Calculate 1D index in input tensor
    input_index = h_in * input_width * channels + w_in * channels + channel_idx
    
    # Load from input and store to output (only valid indices)
    input_val = tl.load(input_ptr + input_index, mask=valid_mask, other=0.0)
    tl.store(output_ptr + offsets, input_val, mask=mask)

@torch.fx.wrap
def optimized_roll_slice_reshape_35x35_384(in_3, in_2):
    """Wrapper for the optimized kernel for 35x35 case"""
    # Handle 6D input: [1, 5, 7, 5, 7, 384] -> reshape to [-1, 35, 35, 384]
    input_shape = in_3.shape
    if len(input_shape) == 6:
        batch, d1, d2, d3, d4, channels = input_shape
        reshaped = in_3.view(-1, 35, 35, channels)
        input_tensor = reshaped.reshape(-1)
        input_height, input_width = 35, 35
    else:
        input_tensor = in_3.reshape(-1)  # Flatten the input
        input_height, input_width = 35, 35  # Fallback for 4D input
    
    # Output parameters for 32x32 slice
    output_height, output_width = 32, 32
    channels = 384
    batch = 1
    output_spatial = output_height * output_width
    
    # Create output tensor
    output_shape = (batch, output_spatial, channels)
    result = torch.empty(output_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel
    n_elements = output_spatial * channels
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    optimized_roll_kernel_35x35_384[grid_size](
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
    return optimized_roll_slice_reshape_35x35_384