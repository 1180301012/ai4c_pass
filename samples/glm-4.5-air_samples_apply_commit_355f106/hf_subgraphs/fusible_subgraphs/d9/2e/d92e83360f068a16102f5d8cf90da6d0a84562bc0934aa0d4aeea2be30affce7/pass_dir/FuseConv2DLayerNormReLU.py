import torch
import triton
import triton.language as tl
import math

def pattern(x, weight):
    # For this fused pattern, we only need the input tensor and weights
    # The bias and layer norm parameters will be handled in replacement_args
    return x  # We'll do the full computation in replacement_func

def replacement_args(x, weight, bias, ln_weight, ln_bias):
    # Extract the shape of the layer norm weight, which indicates the number of channels
    # normalized_shape is (channels, 1, 1) for our case
    ln_shape = ln_weight.shape
    num_channels = ln_shape[0]
    return (x, weight, bias, ln_weight, ln_bias, num_channels)

@triton.jit
def fused_conv_ln_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, ln_weight_ptr, ln_bias_ptr, out_ptr,
    batch_size, in_channels, num_channels, height, width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Each program handles one spatial location in one batch and one output channel
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    hw_id = tl.program_id(2)
    
    if batch_id >= batch_size or channel_id >= num_channels or hw_id >= height * width:
        return
    
    # Layer normalization parameters for this channel
    ln_weight = tl.load(ln_weight_ptr + channel_id)
    ln_bias = tl.load(ln_bias_ptr + channel_id)
    
    # Conv2D computation for this (batch, channel, spatial) location
    conv_sum = 0.0
    
    # Input spatial coordinates
    hw_y = hw_id // width
    hw_x = hw_id % width
    
    # Iterate over input channels and spatial extent of kernel (1x1)
    for in_c in range(in_channels):
        for kh in range(1):  # kernel height = 1
            for kw in range(1):  # kernel width = 1
                # Input coordinate
                in_y = hw_y + kh
                in_x = hw_x + kw
                
                if in_y < height and in_x < width:
                    # Load input value
                    x_offset = (batch_id * in_channels + in_c) * height * width + hw_id
                    x_val = tl.load(x_ptr + x_offset)
                    
                    # Load weight value
                    w_offset = channel_id * in_channels + in_c
                    weight_val = tl.load(weight_ptr + w_offset)
                    
                    conv_sum += x_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + channel_id)
    conv_result = conv_sum + bias_val
    
    # Apply layer normalization weight and bias
    ln_result = conv_result * ln_weight + ln_bias
    
    # Apply ReLU activation
    relu_result = tl.maximum(ln_result, 0.0)
    
    # Store result
    out_offset = (batch_id * num_channels + channel_id) * height * width + hw_id
    tl.store(out_ptr + out_offset, relu_result)

@torch.fx.wrap
def fused_conv_ln_relu(x, weight, bias, ln_weight, ln_bias, num_channels):
    # Get input tensor shapes
    batch_size, in_channels, height, width = x.shape
    
    # Initialize output tensor
    out = torch.empty((batch_size, num_channels, height, width), device=x.device, dtype=x.dtype)
    
    # Block sizes for GPU utilization
    BLOCK_SIZE_M = 64  # Number of channels per block
    BLOCK_SIZE_N = 32   # Number of spatial locations per block
    
    # Calculate grid size - 3D grid: (batch, channels, spatial_locations)
    grid_m = triton.cdiv(batch_size, 1)  # Each program handles one batch
    grid_n = triton.cdiv(num_channels, BLOCK_SIZE_M)  # Blocks of channels
    grid_k = triton.cdiv(height * width, BLOCK_SIZE_N)  # Blocks of spatial locations
    
    # Launch kernel
    fused_conv_ln_relu_kernel[grid_m, grid_n, grid_k](
        x, weight, bias, ln_weight, ln_bias, out,
        batch_size, in_channels, num_channels, height, width,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return fused_conv_ln_relu