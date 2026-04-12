import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    """Match 1x1 Conv2D with stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1"""
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def conv2d_1x1_kernel(
    input_ptr, 
    weight_ptr, 
    bias_ptr, 
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """Optimized 1x1 convolution kernel"""
    # Program IDs
    pid_batch_out = tl.program_id(0)  # Combined batch and out_channel
    pid_h = tl.program_id(1)          # Height position
    pid_w = tl.program_id(2)          # Width position
    
    # Decode batch and out_channel
    out_channels_per_batch = out_channels
    batch_idx = pid_batch_out // out_channels_per_batch
    out_channel_idx = pid_batch_out % out_channels_per_batch
    
    # Only process valid elements
    if batch_idx >= batch_size or out_channel_idx >= out_channels or pid_h >= height or pid_w >= width:
        return
    
    # Initialize accumulator
    acc = 0.0
    
    # Process each input channel (for 1x1 conv, spatial pos is the same)
    for k in range(in_channels):
        # Input offset: [batch, channel, h, w]
        input_offset = (batch_idx * in_channels + k) * height * width + pid_h * width + pid_w
        
        # Weight offset: [out_channel, in_channel, 1, 1]
        weight_offset = (out_channel_idx * in_channels + k) * 1 * 1 + 0
        
        input_val = tl.load(input_ptr + input_offset)
        weight_val = tl.load(weight_ptr + weight_offset)
        acc += input_val * weight_val
    
    # Add bias
    acc += tl.load(bias_ptr + out_channel_idx)
    
    # Store result at [batch, out_channel, h, w]
    output_offset = (batch_idx * out_channels + out_channel_idx) * height * width + pid_h * width + pid_w
    tl.store(output_ptr + output_offset, acc)

@torch.fx.wrap
def optimized_conv2d_1x1(input, weight, bias):
    """Optimized Conv2D wrapper using Triton"""
    batch_size, in_channels, height, width = input.shape
    out_channels = weight.shape[0]
    
    # Output shape
    output = torch.empty((batch_size, out_channels, height, width), dtype=input.dtype, device=input.device)
    
    # 3D grid: [batch * out_channels, height, width]
    # Each program handles one spatial position for one batch-out_channel combination
    grid = (batch_size * out_channels, height, width)
    
    # Launch kernel with 3D grid
    conv2d_1x1_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_H=1,  # Not used in this version
        BLOCK_SIZE_W=1   # Not used in this version
    )
    
    return output

def replacement_func():
    return optimized_conv2d_1x1