import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_pool_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    height,
    width,
    out_channels,
):
    # Simplified execution - each program handles one output pixel
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    spatial_idx = tl.program_id(2)
    
    # Output spatial dimensions
    out_height = height // 2
    out_width = width // 2
    
    # Bounds checking
    if (batch_idx >= batch_size or channel_idx >= out_channels or spatial_idx >= out_height * out_width):
        return
    
    # Decode spatial position
    out_h = spatial_idx // out_width
    out_w = spatial_idx % out_width
    
    # Compute the 2x2 pooling region
    h_start = out_h * 2
    w_start = out_w * 2
    
    # Simple accumulation for fused conv + pooling
    accumulator = 0.0
    valid_pixels = 0
    
    # Iterate over the 2x2 pooling region
    for dy in range(2):
        for dx in range(2):
            h = h_start + dy
            w = w_start + dx
            
            if h < height and w < width:
                # Compute convolution at this position
                conv_sum = 0.0
                for c in range(in_channels):
                    in_val = tl.load(input_ptr + batch_idx * (in_channels * height * width) + c * (height * width) + h * width + w,
                                   other=0.0)
                    weight_val = tl.load(weight_ptr + channel_idx * in_channels + c,
                                       other=0.0)
                    conv_sum += in_val * weight_val
                
                accumulator += conv_sum
                valid_pixels += 1
    
    # Store result (average pooling)
    if valid_pixels > 0:
        result = accumulator / valid_pixels
        output_offset = (batch_idx * out_channels + channel_idx) * (out_height * out_width) + spatial_idx
        tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def fused_conv_pool_2d(weight, input):
    # Get tensor shapes
    batch_size, in_channels, height, width = input.shape
    out_channels, _, weight_height, weight_width = weight.shape
    
    # Output shape after pooling (halved spatial dimensions)
    output_height = height // 2
    output_width = width // 2
    
    # Reshape weight to 2D: [out_channels, in_channels x 1 x 1]
    weight_2d = weight.reshape(out_channels, in_channels)
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_height, output_width), 
                        device=input.device, dtype=input.dtype)
    
    # Flatten input for easier kernel access: [batch_size, in_channels, height, width]
    flattened_input = input
    
    # Calculate grid dimensions for 3D grid
    grid_size_x = batch_size
    grid_size_y = out_channels
    grid_size_z = output_height * output_width
    
    # Launch kernel with 3D grid
    fused_conv_pool_kernel[grid_size_x, grid_size_y, grid_size_z](
        flattened_input,
        weight_2d,
        output,
        batch_size,
        in_channels,
        height,
        width,
        out_channels
    )
    
    return output

def pattern(weight, input):
    # Match conv2d followed by avg_pool2d
    conv_out = torch.conv2d(input, weight, None, (1, 1), (0, 0), (1, 1), 1)
    pool_out = torch.nn.functional.avg_pool2d(conv_out, 2, 2, 0, False, True, None)
    return pool_out

def replacement_args(weight, input):
    return (weight, input)

def replacement_func():
    return fused_conv_pool_2d