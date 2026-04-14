import torch
import triton
import triton.language as tl

# Pattern matching for conv2d operation (exact match to model)
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_2, tmp_0, None, (1, 1), (32, 0), (1, 1), 4)
    tmp_0 = None
    in_1 += tmp_1
    tmp_2 = in_1
    tmp_1 = None
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Simple and effective implementation for conv2d + add fusion
@torch.fx.wrap
def fused_conv2d_add_simple(in_0, in_1, in_2):
    """
    Simple and effective fused conv2d + add implementation
    """
    # Extract dimensions
    input_shape = in_2.shape  # [batch, channels, height, width]
    weight_shape = in_0.shape  # [out_channels, channels/groups, kernel_h, kernel_w]
    
    batch_size, in_channels, in_height, in_width = input_shape
    out_channels, weight_channels_per_group, kernel_h, kernel_w = weight_shape
    
    # Convolution parameters (fixed for our pattern)
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 32, 0
    dilation_h, dilation_w = 1, 1
    groups = 4
    
    # Compute output dimensions
    out_height = in_height + 2 * pad_h - kernel_h + 1
    out_width = in_width + 2 * pad_w - kernel_w + 1
    
    # Use built-in conv2d (optimized by PyTorch)
    conv_output = torch.nn.functional.conv2d(
        in_2, in_0, 
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        groups=groups
    )
    
    # Add to in_1 
    result = in_1 + conv_output
    return result

# Replacement function (returns function reference)
def replacement_func():
    return fused_conv2d_add_simple