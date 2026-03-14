import torch
import triton
import triton.language as tl
import math

def pattern(x, y):
    # Start with a simple conv2d pattern to test matching
    # x = input tensor, y = weight tensor
    out = torch.conv2d(x, y, None, (1, 1), (0, 0), (1, 1), 1)
    return out

def replacement_args(x, y):
    return (x, y)

@triton.jit
def conv_add_interpolate_kernel(
    input_ptr,           # [batch, in_channels, input_h, input_w]
    residual_ptr,        # [batch, out_channels, input_h, input_w] 
    weight_ptr,          # [out_channels, in_channels, 1, 1]
    output_ptr,          # [batch, out_channels, output_h, output_w]
    batch_size,
    in_channels,
    out_channels,
    input_h, input_w,
    output_h, output_w,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles one output location (flattened spatial dimensions)
    pid_b = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # out_channel
    pid_spatial = tl.program_id(2)  # flattened spatial position (y * width + x)
    
    # Convert flattened spatial index to y, x coordinates
    pid_x = pid_spatial % output_w
    pid_y = pid_spatial // output_w
    
    # Bounds checking - Triton doesn't support chained boolean operators
    if pid_b >= batch_size:
        return
    if pid_c >= out_channels:
        return
    if pid_y >= output_h:
        return
    if pid_x >= output_w:
        return
    
    # For 1x1 convolution, input and output spatial coordinates are the same
    src_y = pid_y
    src_x = pid_x
    
    # Calculate input offset for this spatial position
    input_offset = (pid_b * in_channels + pid_c) * input_h * input_w + src_y * input_w + src_x
    
    # Compute 1x1 convolution: sum over input channels
    conv_sum = 0.0
    for k in range(0, in_channels, BLOCK_SIZE_C):
        channel_mask = (k + tl.arange(0, BLOCK_SIZE_C)) < in_channels
        
        # Load weights for output channel pid_c, input channels k:k+BLOCK_SIZE_C
        weight_idx = pid_c * in_channels + k + tl.arange(0, BLOCK_SIZE_C)
        weights = tl.load(weight_ptr + weight_idx, mask=channel_mask, other=0.0)
        
        # Load input features for this spatial position across channels k:k+BLOCK_SIZE_C
        input_idx = input_offset + tl.arange(0, BLOCK_SIZE_C) * (input_h * input_w)
        inputs = tl.load(input_ptr + input_idx, mask=channel_mask, other=0.0)
        
        # Convolution: dot product of weights and inputs
        conv_sum += tl.sum(weights * inputs)
    
    # Just use the convolution result (no residual for simple conv2d pattern)
    result = conv_sum
    
    # Store to output
    output_offset = (pid_b * out_channels + pid_c) * output_h * output_w + pid_y * output_w + pid_x
    tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def fused_conv_add_interpolate(x, y):
    # Simple convolution wrapper using Triton kernel
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels = y.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, input_height, input_width), 
                        dtype=x.dtype, device=x.device)
    
    # Launch kernel with 3D grid (flattened spatial dimensions)
    spatial_size = input_height * input_width
    grid = (batch_size, out_channels, spatial_size)
    
    conv_add_interpolate_kernel[grid](
        input_ptr=x,
        residual_ptr=None,  # Not used in simple conv2d pattern
        weight_ptr=y,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_h=input_height,
        input_w=input_width,
        output_h=input_height,
        output_w=input_width,
        BLOCK_SIZE_C=32,
    )
    
    return output

def replacement_func():
    return fused_conv_add_interpolate