import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    """Match conv2d operation simpler version"""
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def fused_conv_view_kernel(
    in_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
):
    """Simple 1x1 convolution kernel that flattens spatial dimensions"""
    # Each program handles one (batch, output_channel, spatial_position) combination
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)  # Output channel index
    spatial_id = tl.program_id(2)  # Spatial position (flattened)
    
    # Check bounds - only proceed if within valid range
    if channel_id >= out_channels or spatial_id >= (height * width):
        return
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + channel_id, mask=channel_id < out_channels, other=0.0)
    
    # Compute 1x1 convolution: sum over input channels
    output_val = bias
    for k in range(in_channels):
        # Calculate indices for weight and input
        weight_idx = k * out_channels + channel_id
        input_idx = (batch_id * in_channels + k) * height * width + spatial_id
        
        # Load weight and input values with bounds checking
        weight_val = tl.load(weight_ptr + weight_idx, mask=weight_idx < (in_channels * out_channels), other=0.0).to(tl.float32)
        input_val = tl.load(in_ptr + input_idx, mask=input_idx < (batch_size * in_channels * height * width), other=0.0).to(tl.float32)
        
        # Add to the output
        output_val += input_val * weight_val
    
    # Store the result at the correct flattened position
    output_idx = batch_id * out_channels * (height * width) + channel_id * (height * width) + spatial_id
    tl.store(out_ptr + output_idx, output_val, mask=True)

@torch.fx.wrap
def fused_conv_view(in_3, in_1, in_0):
    # Extract input tensor dimensions
    batch_size, in_channels, height, width = in_3.shape
    out_channels = in_0.shape[0]
    
    # Output shape: [batch_size, out_channels, height * width]
    output_size = batch_size * out_channels * (height * width)
    
    # Configure kernel launch parameters - each program handles one (batch, channel, spatial) combination
    num_batch = batch_size
    num_channels = 64  # Process multiple channels per program to reduce grid size
    num_spatial = 512  # Process multiple spatial positions per program
    
    # Allocate output tensor
    out_shape = (batch_size, out_channels, height * width)
    output = torch.empty(out_shape, dtype=torch.float32, device=in_3.device)
    
    # Launch kernel
    fused_conv_view_kernel[(num_batch, num_channels, num_spatial)](
        in_3,
        in_1,
        in_0,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
    )
    
    return output

def replacement_func():
    return fused_conv_view