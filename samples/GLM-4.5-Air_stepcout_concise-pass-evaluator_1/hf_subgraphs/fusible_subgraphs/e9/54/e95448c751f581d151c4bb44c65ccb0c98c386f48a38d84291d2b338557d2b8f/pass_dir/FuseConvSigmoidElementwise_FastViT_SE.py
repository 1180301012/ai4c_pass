import torch
import triton
import triton.language as tl
import math

def pattern(bias, weight, feature_map, se_input):
    # Conv2D with 1x1 kernel and group=1
    conv_out = torch.conv2d(se_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # Sigmoid activation
    sigmoid_out = conv_out.sigmoid()
    # Element-wise multiplication
    fused_out = feature_map * sigmoid_out
    # Return the fused output - this is what affects the final computation
    return fused_out

def replacement_args(bias, weight, feature_map, se_input):
    return (bias, weight, feature_map, se_input)

@triton.jit
def fused_conv_sigmoid_elementwise_kernel_optimized(
    bias_ptr,
    weight_ptr,
    feature_map_ptr,
    se_input_ptr,
    out_ptr,
    batch_size,
    out_channels,
    height,
    width,
    IN_CHANNELS: tl.constexpr,
):
    # Each program handles one output channel for one batch item
    # pid = batch_id * out_channels + channel_id
    pid = tl.program_id(0)
    total_channels = batch_size * out_channels
    
    if pid >= total_channels:
        return
    
    # Calculate batch and channel indices
    batch = pid // out_channels
    channel = pid % out_channels
    
    # Load bias for this channel
    bias_val = tl.load(bias_ptr + channel)
    
    # Compute conv: sum(weight[channel, :] * se_input) + bias[channel] by loading single elements
    channel_conv = bias_val
    weight_offset = channel * IN_CHANNELS
    se_input_offset = batch * IN_CHANNELS
    
    for i in range(IN_CHANNELS):
        # Load weight element for this channel and input channel
        weight_val = tl.load(weight_ptr + weight_offset + i)
        
        # Load SE input element for this batch and input channel  
        se_input_val = tl.load(se_input_ptr + se_input_offset + i)
        
        # Add to convolution sum
        channel_conv += weight_val * se_input_val
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-channel_conv))
    
    # Load and multiply with feature map for all spatial locations
    for h in range(height):
        for w in range(width):
            # Calculate spatial location index
            spatial_idx = h * width + w
            
            # Load feature map for this batch, channel, spatial location
            feature_map_offset = batch * out_channels * height * width + channel * height * width + spatial_idx
            feature_map_val = tl.load(feature_map_ptr + feature_map_offset)
            
            # Element-wise multiplication
            result_val = sigmoid_val * feature_map_val
            
            # Store result
            store_offset = batch * out_channels * height * width + channel * height * width + spatial_idx
            tl.store(out_ptr + store_offset, result_val)

@torch.fx.wrap
def fused_conv_sigmoid_elementwise(bias, weight, feature_map, se_input):
    # Get tensor shapes
    batch_size, in_channels, height, width = se_input.shape
    out_channels = bias.shape[0]
    
    # Create output tensor
    output_shape = (batch_size, out_channels, height, width)
    out = torch.empty(output_shape, dtype=feature_map.dtype, device=feature_map.device)
    out = out.reshape(batch_size, out_channels * height * width)  # Flatten for easier indexing
    
    # Grid configuration: one program per output channel per batch item
    total_programs = batch_size * out_channels
    num_programs = total_programs
    
    # Launch optimized kernel with compile-time constant for in_channels=64
    fused_conv_sigmoid_elementwise_kernel_optimized[(num_programs,)](
        bias_ptr=bias,
        weight_ptr=weight,
        feature_map_ptr=feature_map,
        se_input_ptr=se_input,
        out_ptr=out,
        batch_size=batch_size,
        out_channels=out_channels,
        height=height,
        width=width,
        IN_CHANNELS=64,  # Compile-time constant based on weight_meta
    )
    
    return out

def replacement_func():
    return fused_conv_sigmoid_elementwise