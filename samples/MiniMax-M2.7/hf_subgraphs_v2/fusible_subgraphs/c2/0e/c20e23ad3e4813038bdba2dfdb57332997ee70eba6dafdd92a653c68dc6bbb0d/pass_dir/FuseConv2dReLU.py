import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv2d_relu_kernel(
    # Pointers
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Stride parameters
    in_batch_stride, in_channel_stride, in_height_stride, in_width_stride,
    wt_batch_stride, wt_channel_stride, wt_height_stride, wt_width_stride,
    out_batch_stride, out_channel_stride, out_height_stride, out_width_stride,
    # Sizes
    in_batch, in_channels, in_height, in_width,
    out_channels, out_height, out_width,
    # Conv params
    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, 
    dilation_h, dilation_w, group,
    # Meta
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv2D + ReLU kernel.
    - input: (batch, in_channels, height, width) 
    - weight: (out_channels, in_channels//groups, kernel_h, kernel_w)
    - bias: (out_channels,)
    - output: (batch, out_channels, out_height, out_width)
    """
    # Get position
    pid = tl.program_id(0)
    batch = pid // (out_channels * out_height * out_width)
    channel = (pid // (out_height * out_width)) % out_channels
    out_y = (pid // out_width) % out_height
    out_x = pid % out_width
    
    # Compute input position for this output
    in_y_start = out_y * stride_h - padding_h
    in_x_start = out_x * stride_w - padding_w
    
    # Accumulator for convolution
    res = 0.0
    
    # Load bias
    bias_val = tl.load(bias_ptr + channel)
    
    # Compute convolution for this output position
    # Channels per group
    channels_per_group = in_channels // group
    in_group_start = 0  # Simplified, assuming group=1 for this pattern
    
    # Loop over kernel
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            in_y = in_y_start + ky * dilation_h
            in_x = in_x_start + kx * dilation_w
            
            # Check bounds
            if in_y >= 0 and in_y < in_height and in_x >= 0 and in_x < in_width:
                for ic in range(channels_per_group):
                    # Input offset
                    in_off = (batch * in_batch_stride + 
                             ic * in_channel_stride + 
                             in_y * in_height_stride + 
                             in_x * in_width_stride)
                    # Weight offset
                    wt_off = (channel * wt_batch_stride + 
                             ic * wt_channel_stride + 
                             ky * wt_height_stride + 
                             kx * wt_width_stride)
                    
                    val = tl.load(input_ptr + in_off)
                    w = tl.load(weight_ptr + wt_off)
                    res += val * w
    
    # Add bias
    res = res + bias_val
    
    # Apply ReLU: max(0, x)
    res = tl.where(res > 0, res, 0.0)
    
    # Store output
    out_off = (batch * out_batch_stride + 
              channel * out_channel_stride + 
              out_y * out_height_stride + 
              out_x * out_width_stride)
    tl.store(output_ptr + out_off, res)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the Conv2D + ReLU pattern.
    Conv2D: input(in_3), weight(in_1), bias(in_0), stride=(2,2), padding=(1,1), dilation=(1,1), groups=1
    ReLU: inplace activation
    """
    # Note: torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
    conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(conv2d, inplace=True)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the fused kernel."""
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def fused_conv2d_relu(in_0, in_1, in_2, in_3):
    """
    Fused Conv2D + ReLU implementation.
    in_0: bias (128,)
    in_1: weight (128, 256, 3, 3)
    in_2: residual input (1, 128, 24, 24) - used later, not needed here
    in_3: input (1, 256, 48, 48)
    """
    # Get shapes
    batch, in_ch, in_h, in_w = in_3.shape
    out_ch = in_1.shape[0]  # 128
    kernel_h, kernel_w = in_1.shape[2], in_1.shape[3]
    stride_h, stride_w = 2, 2
    padding_h, padding_w = 1, 1
    dilation_h, dilation_w = 1, 1
    
    # Compute output size
    out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Allocate output
    output = torch.empty((batch, out_ch, out_h, out_w), dtype=in_3.dtype, device=in_3.device)
    
    # Flatten tensors for ptr arithmetic
    in_3_flat = in_3.flatten()
    in_1_flat = in_1.flatten()
    in_0_flat = in_0.flatten()
    output_flat = output.flatten()
    
    # Strides (as flattened indices)
    # Input: (batch, in_ch, in_h, in_w)
    in_batch_stride = in_ch * in_h * in_w
    in_channel_stride = in_h * in_w
    in_height_stride = in_w
    in_width_stride = 1
    
    # Weight: (out_ch, in_ch//groups, k_h, k_w)
    wt_batch_stride = (in_ch // 1) * kernel_h * kernel_w
    wt_channel_stride = kernel_h * kernel_w
    wt_height_stride = kernel_w
    wt_width_stride = 1
    
    # Output: (batch, out_ch, out_h, out_w)
    out_batch_stride = out_ch * out_h * out_w
    out_channel_stride = out_h * out_w
    out_height_stride = out_w
    out_width_stride = 1
    
    # Total output elements
    total_elements = batch * out_ch * out_h * out_w
    
    BLOCK_SIZE = 128
    
    fused_conv2d_relu_kernel[(total_elements,)](
        in_3_flat, in_1_flat, in_0_flat, output_flat,
        in_batch_stride, in_channel_stride, in_height_stride, in_width_stride,
        wt_batch_stride, wt_channel_stride, wt_height_stride, wt_width_stride,
        out_batch_stride, out_channel_stride, out_height_stride, out_width_stride,
        batch, in_ch, in_h, in_w,
        out_ch, out_h, out_w,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, 1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_conv2d_relu