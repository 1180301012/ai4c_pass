import torch
import triton
import triton.language as tl
import math

@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    kernel_h,
    kernel_w,
    padding_h,
    padding_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    # Get program IDs - using 2D grid for spatial parallelism
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # Calculate output coordinates
    out_h = pid_x
    out_c = pid_y
    
    # Calculate input coordinates with padding
    in_c = out_c
    in_h = out_h * stride_h
    in_w = out_w = 0  # Will be calculated per thread
    
    # Initialize output accumulator
    output_val = 0.0 if bias_ptr is None else tl.load(bias_ptr + out_c)
    
    # Iterate over kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input coordinates with padding and dilation
            in_h_padded = in_h + (kh * dilation_h) - padding_h
            in_w_padded = in_w + (kw * dilation_w) - padding_w
            
            # Check bounds
            if (0 <= in_h_padded < height and 0 <= in_w_padded < width):
                # Calculate linear output element index
                out_idx = (out_c * height + out_h) * width + out_w
                
                # Calculate linear input element index for this channel and spatial location
                in_idx = (in_c * height + in_h_padded) * width + in_w_padded
                
                # Load input and weight
                input_val = tl.load(input_ptr + in_idx)
                weight_val = tl.load(weight_ptr + (out_c * kernel_h + kh) * kernel_w + kw)
                
                # Accumulate
                output_val += input_val * weight_val
    
    # Store result if within bounds
    if out_c < channels and out_h < height:
        out_idx = (out_c * height + out_h) * width
        tl.store(output_ptr + out_idx, output_val)

@triton.jit  
def optimized_depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    kernel_h,
    kernel_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Simplified 1D kernel for depthwise convolution
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    
    # Calculate total elements in output
    total_elements = batch_size * channels * height * width
    output_idx = pid * block_size + tl.arange(0, block_size)
    mask = output_idx < total_elements
    
    # Convert linear index to coordinates
    batch_idx = output_idx // (channels * height * width)
    remainder = output_idx % (channels * height * width)
    channel_idx = remainder // (height * width)
    spatial_idx = remainder % (height * width)
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Load bias if available
    bias_val = 0.0
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + channel_idx)
    
    # Initialize accumulator
    output_val = bias_val
    
    # Depthwise convolution computation
    # Each output element is computed by convolving its corresponding input 
    # channel with the dedicated kernel channel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input coordinates with padding (7x7 kernel with padding=3 gives valid convolution for same output size)
            in_h = h_idx + kh - 3  # padding = 3
            in_w = w_idx + kw - 3  # padding = 3
            
            # Check bounds
            if (0 <= in_h < height and 0 <= in_w < width):
                # Calculate input index
                input_idx = (batch_idx * channels + channel_idx) * height * width + in_h * width + in_w
                weight_idx = channel_idx * kernel_h * kernel_w + kh * kernel_w + kw
                
                # Load and accumulate
                input_val = tl.load(input_ptr + input_idx, mask=True, other=0.0)
                weight_val = tl.load(weight_ptr + weight_idx)
                output_val += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + output_idx, output_val, mask=mask)

@torch.fx.wrap
def depthwise_conv2d_optimized(input_tensor, weight_tensor, bias_tensor):
    # Get tensor shapes
    batch_size, channels, height, width = input_tensor.shape
    kernel_channels, kernel_h, kernel_w = weight_tensor.shape[0], weight_tensor.shape[2], weight_tensor.shape[3]
    
    # Verify depthwise convolution setup
    assert kernel_channels == channels, "Depthwise convolution requires weight channels == input channels"
    assert weight_tensor.shape[1] == 1, "Weight should be [channels, 1, kernel_h, kernel_w] for depthwise"
    
    # Prepare output
    output = torch.empty((batch_size, channels, height, width), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Calculate grid configuration
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_programs,)
    
    # Launch kernel
    optimized_depthwise_conv2d_kernel[grid](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        batch_size,
        channels,
        height,
        width,
        kernel_h,
        kernel_w,
        BLOCK_SIZE
    )
    
    return output

def pattern(input_tensor, weight_tensor, bias_tensor):
    # Match conv2d operation using functional API
    # tmp_6 = torch.conv2d(in_6, tmp_5, tmp_4, (1, 1), (3, 3), (1, 1), 192)
    # Using functional conv2d with explicit parameters
    result = torch.nn.functional.conv2d(input_tensor, weight_tensor, bias_tensor, 
                                       stride=(1, 1), padding=(3, 3), dilation=(1, 1), 
                                       groups=input_tensor.shape[1])
    return result

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

def replacement_func():
    return depthwise_conv2d_optimized