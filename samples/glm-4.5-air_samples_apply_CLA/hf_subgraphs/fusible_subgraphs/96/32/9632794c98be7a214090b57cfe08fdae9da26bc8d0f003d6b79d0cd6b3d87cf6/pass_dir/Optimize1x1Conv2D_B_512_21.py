import torch
import triton
import triton.language as tl
import math

# Pattern matching function for the 1x1 conv2d operation
def pattern(in_0, in_1, in_2):
    bias = in_0
    weight = in_1  
    input_tensor = in_2
    output = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return output

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized 1x1 convolution kernel using Triton
@triton.jit
def optimized_conv1x1_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_K: tl.constexpr
):
    # Get program ids
    pid_m = tl.program_id(0)  # batch dimension
    pid_c = tl.program_id(1)  # output channel dimension  
    pid_n = tl.program_id(2) # spatial dimension (flattened)
    
    # Early return for out-of-bounds program ids
    if pid_m >= batch_size:
        return
    if pid_c >= out_channels:
        return
    if pid_n >= height * width:
        return
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Initialize accumulator with bias
    accum = bias_val
    
    # Loop over input channels
    offsets = tl.arange(0, BLOCK_SIZE_K)
    mask = offsets < in_channels
    
    # Compute input tensor offsets: [batch, ic, h, w] -> flattened
    input_offsets = (pid_m * height * width * in_channels + 
                    offsets * height * width + pid_n)
    
    # Compute weight tensor offsets: [out_c, ic, 1, 1] -> flattened  
    weight_offsets = (pid_c * in_channels + offsets)
    
    # Load input and weight values
    input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    weight_vals = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    
    # Compute dot product and accumulate
    dots = input_vals * weight_vals
    accum += tl.sum(dots)
    
    # Store the result
    output_offset = (pid_m * out_channels * height * width + 
                    pid_c * height * width + pid_n)
    tl.store(output_ptr + output_offset, accum)

@torch.fx.wrap
def optimized_1x1_conv2d(bias, weight, input_tensor):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Output tensor
    output = torch.empty((batch_size, out_channels, height, width), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # Block size for input channels - vectorize the channel computation
    BLOCK_SIZE_K = min(512, in_channels)  # Process up to 512 channels at once
    
    # Grid dimensions - 3D grid: [batch, output_channel, spatial_element]
    # Each program handles one spatial location for one batch and output channel
    grid_x = batch_size
    grid_y = out_channels
    grid_z = height * width  # One program per spatial location
    
    # Launch kernel with 3D grid
    optimized_conv1x1_kernel[(grid_x, grid_y, grid_z)](
        input_tensor,
        weight,
        bias,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE_K
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_1x1_conv2d