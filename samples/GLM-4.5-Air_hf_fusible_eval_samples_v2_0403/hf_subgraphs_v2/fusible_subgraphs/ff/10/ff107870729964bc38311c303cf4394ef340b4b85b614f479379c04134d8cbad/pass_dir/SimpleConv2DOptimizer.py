import torch
import triton
import triton.language as tl

def pattern(a, b, c):
    """Pattern: simple conv2d operation"""
    return torch.conv2d(a, b, c, (1, 1), (0, 0), (1, 1), 1)

def replacement_args(a, b, c):
    return (a, b, c)

@triton.jit
def simple_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch, input_channels, height, width, output_channels,
    kernel_size: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each thread processes one output position for all channels
    channel = pid % output_channels
    h = (pid // output_channels) // width
    w = (pid // output_channels) % width
    
    # 1x1 convolution is straightforward
    input_idx = (channel * height * width) + (h * width) + w
    weight_idx = channel
    bias_idx = channel
    
    input_val = tl.load(input_ptr + input_idx)
    weight_val = tl.load(weight_ptr + weight_idx)
    bias_val = tl.load(bias_ptr + bias_idx)
    
    result = input_val * weight_val + bias_val
    output_idx = (channel * height * width) + (h * width) + w
    
    tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def simple_conv2d(input, weight, bias):
    batch, input_channels, height, width = input.shape
    output_channels = weight.shape[0]
    
    output = torch.empty((batch, output_channels, height, width), 
                        dtype=input.dtype, device=input.device)
    
    input_flat = input.reshape(-1)
    weight_flat = weight.reshape(-1)
    bias_flat = bias.reshape(-1)
    output_flat = output.reshape(-1)
    
    grid_size = batch * output_channels * height * width
    simple_conv2d_kernel[(grid_size,)](
        input_flat, weight_flat, bias_flat, output_flat,
        batch, input_channels, height, width, output_channels, 1
    )
    
    return output

def replacement_func():
    return simple_conv2d