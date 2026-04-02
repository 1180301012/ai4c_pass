import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Optimized pattern: Just Conv2D for better kernel performance"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for optimized 1x1 Conv2D"""
    return (in_0, in_1, in_2)

# Highly optimized Triton kernel for 1x1 Conv2D
@triton.jit
def optimized_conv1x1_kernel(
    bias_ptr, weight_ptr, input_ptr, output_ptr,
    batch_size, input_channels, output_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    """Highly optimized 1x1 Conv2D kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * output_channels * height * width)
    
    # Efficient tile-based computation
    x = offsets // output_channels
    y = offsets % output_channels
    
    # Calculate spatial position
    spatial = x // (output_channels)
    idx_b = spatial // (height * width)
    idx_h = (spatial % (height * width)) // width  
    idx_w = spatial % width
    
    # Load input: [batch, input_channels, height, width]
    input_val = tl.load(input_ptr + idx_b * input_channels * height * width + x * input_channels + idx_h * width + idx_w,
                       mask=(idx_b < batch_size) & (idx_h < height) & (idx_w < width), other=0.0)
    
    # Load weights: [output_channels, input_channels, 1, 1] 
    # For 1x1 conv: we can leverage the fact that we're mapping input->output channels directly
    if input_channels == output_channels:
        # Square matrix case: diagonal elements only for identity transformation
        weight_val = tl.load(weight_ptr + x * input_channels + y,
                           mask=(x < input_channels) & (y < input_channels), other=0.0)
    else:
        # General case: full matrix multiplication
        weight_val = tl.load(weight_ptr + y * input_channels + (x % input_channels),
                           mask=(y < output_channels) & ((x % input_channels) < input_channels), other=0.0)
    
    # Load bias
    bias_val = tl.load(bias_ptr + y, mask=(y < output_channels), other=0.0)
    
    # Compute 1x1 convolution
    output_val = bias_val + input_val * weight_val
    tl.store(output_ptr + offsets, output_val, mask=mask)

@torch.fx.wrap
def optimized_conv1x1(bias, weight, input_tensor):
    """Highly optimized wrapper for 1x1 Conv2D"""
    batch_size, input_channels, height, width = input_tensor.shape
    output_channels = weight.shape[0]
    
    output = torch.empty((batch_size, output_channels, height, width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    total_elements = batch_size * output_channels * height * width
    
    # Tune block size based on total elements for optimal occupancy
    if total_elements < 1024:
        BLOCK_SIZE = 64
    elif total_elements < 10000:
        BLOCK_SIZE = 256  
    elif total_elements < 50000:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
        
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_conv1x1_kernel[(num_programs,)](
        bias_ptr=bias, weight_ptr=weight, input_ptr=input_tensor, output_ptr=output,
        batch_size=batch_size, input_channels=input_channels, output_channels=output_channels,
        height=height, width=width, BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Returns optimized 1x1 Conv2D function"""
    return optimized_conv1x1