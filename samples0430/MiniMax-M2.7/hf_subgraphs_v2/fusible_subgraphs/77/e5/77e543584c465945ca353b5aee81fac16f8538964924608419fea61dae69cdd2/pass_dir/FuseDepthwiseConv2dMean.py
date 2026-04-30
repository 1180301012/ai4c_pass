import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv2d_mean_kernel(
    input_ptr, weight_ptr, output_ptr, mean_ptr,
    batch_stride_in, channel_stride_in, height_stride_in, width_stride_in,
    batch_stride_w, channel_stride_w, height_stride_w, width_stride_w,
    batch_stride_out, channel_stride_out, height_stride_out, width_stride_out,
    batch_stride_mean, channel_stride_mean, height_stride_mean, width_stride_mean,
    batch_size, channels, in_height, in_width,
    out_height, out_width, kernel_h, kernel_w,
    stride_h, stride_w, padding_h, padding_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused depthwise conv2d + spatial mean kernel.
    
    Args:
        input_ptr: input tensor [B, C, H, W]
        weight_ptr: weight tensor [C, 1, 3, 3]
        output_ptr: conv2d output [B, C, H', W']
        mean_ptr: spatial mean output [B, C, 1, 1]
    """
    pid = tl.program_id(0)
    num_blocks = batch_size * channels
    block_idx = pid
    
    if block_idx >= num_blocks:
        return
    
    batch_idx = block_idx // channels
    channel_idx = block_idx % channels
    
    # Compute output height and width for this batch/channel
    output_height = out_height
    output_width = out_width
    
    # Compute mean accumulator
    mean_acc = tl.zeros([1], dtype=tl.float32)
    mean_count = tl.zeros([1], dtype=tl.float32)
    
    # Process each output spatial location
    for out_h_idx in range(output_height):
        for out_w_idx in range(output_width):
            conv_sum = tl.zeros([1], dtype=tl.float32)
            
            # Compute convolution
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    in_h = out_h_idx * stride_h + kh - padding_h
                    in_w = out_w_idx * stride_w + kw - padding_w
                    
                    # Boundary check
                    if 0 <= in_h < in_height and 0 <= in_w < in_width:
                        in_offset = (
                            batch_idx * batch_stride_in +
                            channel_idx * channel_stride_in +
                            in_h * height_stride_in +
                            in_w * width_stride_in
                        )
                        w_offset = (
                            channel_idx * batch_stride_w +  # channels dimension
                            0 * channel_stride_w +  # groups=1, so input_channel=0
                            kh * height_stride_w +
                            kw * width_stride_w
                        )
                        
                        inp_val = tl.load(input_ptr + in_offset)
                        w_val = tl.load(weight_ptr + w_offset)
                        conv_sum += inp_val * w_val
            
            # Store conv output
            out_offset = (
                batch_idx * batch_stride_out +
                channel_idx * channel_stride_out +
                out_h_idx * height_stride_out +
                out_w_idx * width_stride_out
            )
            tl.store(output_ptr + out_offset, conv_sum, mask=None)
            
            # Accumulate for mean
            mean_acc += conv_sum
            mean_count += 1.0
    
    # Compute mean
    mean_val = mean_acc / mean_count
    
    # Store mean output [B, C, 1, 1]
    mean_offset = (
        batch_idx * batch_stride_mean +
        channel_idx * channel_stride_mean +
        0 * height_stride_mean +
        0 * width_stride_mean
    )
    tl.store(mean_ptr + mean_offset, mean_val)


@torch.fx.wrap
def fused_conv2d_mean_wrapper(input_tensor, weight_tensor, stride, padding, dilation, groups):
    """
    Wrapper for the fused conv2d + mean kernel.
    
    Args:
        input_tensor: [B, C_in, H, W] 
        weight_tensor: [C_out, 1, 3, 3] for depthwise conv
        stride: tuple (stride_h, stride_w)
        padding: tuple (pad_h, pad_w)
        dilation: tuple (dil_h, dil_w) - typically (1, 1)
        groups: number of groups - should equal C_out for depthwise
    
    Returns:
        (conv_output, mean_output)
    """
    B, C_in, H, W = input_tensor.shape
    C_out, _, KH, KW = weight_tensor.shape
    
    # Compute output spatial dimensions
    out_h = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    out_w = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1
    
    # Create output tensors
    output = torch.empty(B, C_out, out_h, out_w, dtype=input_tensor.dtype, device=input_tensor.device)
    mean_output = torch.empty(B, C_out, 1, 1, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Get strides
    input_stride = input_tensor.stride()
    weight_stride = weight_tensor.stride()
    output_stride = output.stride()
    mean_stride = mean_output.stride()
    
    # Launch kernel
    num_programs = B * C_out
    BLOCK_SIZE = 64
    
    fused_conv2d_mean_kernel[(num_programs,)](
        input_tensor, weight_tensor, output, mean_output,
        input_stride[0], input_stride[1], input_stride[2], input_stride[3],
        weight_stride[0], weight_stride[1], weight_stride[2], weight_stride[3],
        output_stride[0], output_stride[1], output_stride[2], output_stride[3],
        mean_stride[0], mean_stride[1], mean_stride[2], mean_stride[3],
        B, C_out, H, W, out_h, out_w, KH, KW,
        stride[0], stride[1], padding[0], padding[1],
        BLOCK_SIZE
    )
    
    return output, mean_output


def pattern(in_0, in_1):
    """
    Match the pattern: conv2d followed by mean over spatial dimensions.
    
    The weight tensor in_0 has shape [C, 1, 3, 3] (depthwise conv).
    The input tensor in_1 has shape [B, C, H, W].
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 384)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return conv2d, tmp_2


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the fused kernel.
    
    The pattern always uses stride=(1,1), padding=(1,1), dilation=(1,1).
    Groups is determined by the weight tensor's first dimension.
    """
    # From the pattern: stride=(1,1), padding=(1,1), dilation=(1,1)
    # The groups parameter for conv2d corresponds to weight.shape[0] / 1 = weight.shape[0]
    groups = in_0.shape[0]
    return (in_0, in_1, (1, 1), (1, 1), (1, 1), groups)


def replacement_func():
    """
    Return the fused conv2d + mean wrapper function.
    """
    return fused_conv2d_mean_wrapper