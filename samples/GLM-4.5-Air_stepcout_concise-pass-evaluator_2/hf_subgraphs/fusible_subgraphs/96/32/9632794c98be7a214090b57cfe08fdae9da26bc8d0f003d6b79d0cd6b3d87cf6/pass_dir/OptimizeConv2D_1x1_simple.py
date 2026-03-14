import torch
import triton
import triton.language as tl

# Pattern matching function for 1x1 convolution
def pattern(in_2, in_1, in_0):
    """Match 1x1 convolution with exact parameters from the model"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

# Argument extraction function
def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

# Simplified 1x1 convolution kernel
@triton.jit
def simple_1x1_conv_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Decode batch, output_channel, h, w from linear index
    linear_idx = pid
    if linear_idx >= batch_size * out_channels * height * width:
        return
        
    # Calculate coordinates
    b = linear_idx // (out_channels * height * width)
    remainder = linear_idx % (out_channels * height * width)
    c_out = remainder // (height * width)
    remainder2 = remainder % (height * width)
    h = remainder2 // width
    w = remainder2 % width
    
    # 1x1 convolution: sum over input channels
    # result = bias[c_out] + sum_{c_in} (input[b, c_in, h, w] * weight[c_out, c_in])
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + c_out)
    
    # Compute actual 1x1 convolution: bias[c_out] + sum_{c_in} (input[b, c_in, h, w] * weight[c_out, c_in])
    result = bias_val
    
    # Loop through input channels manually
    for c_in in range(in_channels):
        # Compute input tensor index: (b, c_in, h, w)
        input_idx = (b * in_channels + c_in) * height * width + h * width + w
        input_val = tl.load(input_ptr + input_idx)
        
        # Compute weight tensor index: (c_out, c_in)
        weight_idx = c_out * in_channels + c_in
        weight_val = tl.load(weight_ptr + weight_idx)
        
        # Add to result
        result += input_val * weight_val
    
    # Store the result
    tl.store(output_ptr + linear_idx, result)

@torch.fx.wrap  
def simple_conv2d(input_tensor, weight_tensor, bias_tensor):
    """Simple 1x1 convolution using Triton"""
    # Get tensor dimensions
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, height, width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch simple kernel with proper parameters
    total_elements = batch_size * out_channels * height * width
    num_programs = (total_elements + 255 - 1) // 256
    grid = (num_programs,)
    
    simple_1x1_conv_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width
    )
    
    return output

# Replacement function
def replacement_func():
    return simple_conv2d