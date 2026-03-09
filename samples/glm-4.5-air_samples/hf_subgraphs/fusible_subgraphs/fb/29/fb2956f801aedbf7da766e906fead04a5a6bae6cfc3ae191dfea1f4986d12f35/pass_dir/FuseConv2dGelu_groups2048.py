import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels: tl.constexpr, out_channels: tl.constexpr,
    height: tl.constexpr, width: tl.constexpr,
    kernel_h: tl.constexpr, kernel_w: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    dilation_h: tl.constexpr, dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv2D + GELU kernel for depthwise convolution (groups=2048).
    """
    pid = tl.program_id(0)
    batch_pid = tl.program_id(1)
    
    out_height = (height + stride_h - dilation_h * (kernel_h - 1) - 1) // stride_h
    out_width = (width + stride_w - dilation_w * (kernel_w - 1) - 1) // stride_w
    
    out_ch = pid
    batch_idx = batch_pid
    
    if out_ch >= out_channels:
        return
    
    num_outs = out_height * out_width
    
    bias_val = tl.load(bias_ptr + out_ch)
    
    for h_out in range(out_height):
        for w_out in range(out_width):
            h_in = h_out * stride_h - pad_h
            w_in = w_out * stride_w - pad_w
            
            conv_sum = 0.0
            
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    h_img = h_in + kh * dilation_h
                    w_img = w_in + kw * dilation_w
                    
                    if h_img >= 0 and h_img < height and w_img >= 0 and w_img < width:
                        inp_idx = batch_idx * in_channels * height * width + out_ch * height * width + h_img * width + w_img
                        inp_val = tl.load(input_ptr + inp_idx)
                        
                        w_idx = out_ch * in_channels * kernel_h * kernel_w + 0 * kernel_h * kernel_w + kh * kernel_w + kw
                        w_val = tl.load(weight_ptr + w_idx)
                        
                        conv_sum += inp_val * w_val
            
            conv_sum = conv_sum + bias_val
            
            x = conv_sum
            gelu = x * 0.5 * (1 + tl.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
            
            out_idx = batch_idx * out_channels * num_outs + out_ch * num_outs + h_out * out_width + w_out
            tl.store(output_ptr + out_idx, gelu)


def fused_conv2d_gelu(input, weight, bias, stride, padding, dilation, groups):
    batch, in_channels, height, width = input.shape
    out_channels = weight.shape[0]
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    
    out_height = (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_width = (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    
    output = torch.empty((batch, out_channels, out_height, out_width), 
                        device=input.device, dtype=input.dtype)
    
    grid = (out_channels, batch)
    
    conv2d_gelu_kernel[grid](
        input, weight, bias, output,
        in_channels, out_channels,
        height, width,
        kernel_h, kernel_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        BLOCK_SIZE=1024,
    )
    
    return output


@torch.fx.wrap
def fuse_conv2d_gelu_groups2048_wrapper(in_0, in_1, in_2, stride, padding, dilation, groups):
    return fused_conv2d_gelu(in_2, in_1, in_0, stride, padding, dilation, groups)


def pattern(in_0, in_1, in_2):
    """
    Match pattern: Conv2D -> GELU -> Dropout(p=0) for groups=2048.
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 2048)
    tmp_1 = tmp_0 = None
    tmp_3 = torch.nn.functional.gelu(tmp_2)
    tmp_2 = None
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    tmp_3 = None
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, (1, 1), (1, 1), (1, 1), 2048)


def replacement_func():
    return fuse_conv2d_gelu_groups2048_wrapper