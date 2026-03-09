import torch
import triton
import triton.language as tl

# Pattern matching: Conv2D operation (matching the actual computation)
def pattern(input_tensor, weight_tensor, bias_tensor):
    # Conv2D operation matching the actual computation
    # This matches: torch.conv2d(input, weight, bias, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1)
    result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    return result

# Argument extraction
def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

# Optimized 1x1 Conv2D kernel using Triton
@triton.jit
def conv2d_1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Program indices
    pid = tl.program_id(0)
    batch_idx = pid // out_channels
    channel_idx = pid % out_channels
    
    if batch_idx >= batch_size or channel_idx >= out_channels:
        return
    
    # Output offset
    output_offset = batch_idx * out_channels * height * width + channel_idx * height * width
    
    # Initialize with bias
    result = 0.0
    if bias_ptr is not None:
        result = tl.load(bias_ptr + channel_idx)
    
    # Compute 1x1 convolution (sum over all spatial positions and input channels)
    for c_in in range(in_channels):
        weight_val = tl.load(weight_ptr + channel_idx * in_channels + c_in)
        spatial_sum = 0.0
        for h in range(height):
            for w in range(width):
                input_offset = (batch_idx * in_channels * height * width + 
                              c_in * height * width + h * width + w)
                input_val = tl.load(input_ptr + input_offset)
                spatial_sum += input_val
        result += weight_val * spatial_sum
    
    # Store result
    tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def optimized_conv2d_1x1(input_tensor, weight_tensor, bias_tensor):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    BLOCK_SIZE = 256
    total_elements = batch_size * out_channels
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty([batch_size, out_channels, height, width], 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    conv2d_1x1_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_conv2d_1x1