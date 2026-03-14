import torch
import triton
import triton.language as tl

def pattern(in_9, in_4):
    # Simplified pattern: just conv2d + view
    tmp_4 = torch.conv2d(input=in_9, weight=in_4, groups=512)
    tmp_5 = tmp_4.view(1, 512, 64, 64)
    return tmp_5

def replacement_args(in_9, in_4):
    return (in_9, in_4)

@triton.jit
def depthwise_conv_kernel(
    x_ptr,  # input [1, 512, 70, 70] - contiguous memory layout
    weight_ptr,  # weight [512, 1, 7, 7]
    out_ptr,  # output [1, 512, 64, 64]
    stride: tl.constexpr,
    padding: tl.constexpr,
    groups: tl.constexpr,
    batch: tl.constexpr,
    channels: tl.constexpr,
    in_height: tl.constexpr,
    in_width: tl.constexpr,
    kernel_size: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
):
    # Each program handles one output position for one channel group
    gid = tl.program_id(0)
    channel_group = gid // (out_height * out_width)
    spatial_idx = gid % (out_height * out_width)
    
    if channel_group >= groups or spatial_idx >= out_height * out_width:
        return
    
    # Calculate spatial coordinates
    h_out = spatial_idx // out_width
    w_out = spatial_idx % out_width
    
    # Input coordinates with padding
    h_in = h_out * stride - padding
    w_in = w_out * stride - padding
    
    # Output pointer for this channel group
    out_base = out_ptr + (channel_group * out_height + h_out) * out_width + w_out
    
    # Input base pointer for this channel group (each group processes one channel)
    in_base = x_ptr + (channel_group * in_height + h_in) * in_width + w_in
    
    # Weight pointer for this group (7x7 kernel)
    weight_base = weight_ptr + channel_group * kernel_size * kernel_size
    
    # Apply depthwise convolution using proper strides
    acc = 0.0
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input coordinates with stride
            ih = h_in + kh * stride
            iw = w_in + kw * stride
            
            # Calculate linear offset using proper stride layout
            input_offset = channel_group * (in_height * in_width) + ih * in_width + iw
            weight_offset = channel_group * (kernel_size * kernel_size) + kh * kernel_size + kw
            
            # Load input and weight with boundary checking
            if ih < in_height and iw < in_width:
                input_val = tl.load(x_ptr + input_offset)
                weight_val = tl.load(weight_ptr + weight_offset)
                acc += input_val * weight_val
    
    # Store result
    tl.store(out_base, acc)

@triton.jit
def batch_norm_relu_kernel(
    x_ptr,  # input [1, 512, 64, 64]
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,  # output [1, 512, 64, 64]
    eps: tl.constexpr,
    momentum: tl.constexpr,
    batch: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    # Each program handles one output position
    gid = tl.program_id(0)
    if gid >= batch * channels * height * width:
        return
    
    # Calculate coordinates
    channel = gid // (height * width)
    spatial_idx = gid % (height * width)
    h = spatial_idx // width
    w = spatial_idx % width
    
    # Input pointer to current element
    x_val = tl.load(x_ptr + gid)
    
    # Load batch norm parameters
    mean = tl.load(running_mean_ptr + channel)
    var = tl.load(running_var_ptr + channel)
    weight = tl.load(weight_ptr + channel)
    bias = tl.load(bias_ptr + channel)
    
    # BatchNorm + ReLU: y = relu((x - mean) / sqrt(var + eps) * weight + bias)
    norm = (x_val - mean) / tl.sqrt(var + eps)
    batch_norm_out = norm * weight + bias
    relu_out = tl.maximum(batch_norm_out, 0.0)
    
    # Store result
    tl.store(out_ptr + gid, relu_out)

@torch.fx.wrap
def simple_conv_view(x, weight):
    # Use Triton kernel for depthwise convolution + view
    # Input shapes from meta
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, k_channels, kernel_h, kernel_w = weight.shape
    
    # Assert depthwise conv setup
    assert in_channels == out_channels == 512 and k_channels == 1 and kernel_h == kernel_w == 7
    assert batch_size == 1 and in_height == 70 and in_width == 70
    
    # Conv parameters
    groups = 512
    stride = 1
    padding = 3  # To get 64x64 output: (70 + 6 - 7) // 1 + 1 = 64
    out_height = out_width = 64
    kernel_size = 7
    
    # Create output tensor
    conv_out = torch.empty((batch_size, out_channels, out_height, out_width), dtype=torch.float32, device=x.device)
    
    # Launch depthwise convolution kernel
    total_elements_conv = batch_size * groups * out_height * out_width
    block_size_conv = 1024
    num_programs_conv = (total_elements_conv + block_size_conv - 1) // block_size_conv
    
    depthwise_conv_kernel[(num_programs_conv,)](
        x_ptr=x,
        weight_ptr=weight,
        out_ptr=conv_out,
        stride=stride,
        padding=padding,
        groups=groups,
        batch=batch_size,
        channels=out_channels,
        in_height=in_height,
        in_width=in_width,
        kernel_size=kernel_size,
        out_height=out_height,
        out_width=out_width,
    )
    
    # Apply view operation
    result = conv_out.view(1, 512, 64, 64)
    
    return result

def replacement_func():
    return simple_conv_view