import torch
import triton
import triton.language as tl


def pattern(in6, in1, in0):
    conv = torch.conv3d(in6, in1, in0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp7 = conv.flatten(2)
    tmp8 = tmp7.transpose(1, 2)
    return tmp8

def replacement_args(in6, in1, in0):
    return (in6, in1, in0)


@triton.jit
def fused_conv3d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    depth,
    height,
    width,
    out_channels,
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_depth,
    stride_height,
    stride_width,
    padding_depth,
    padding_height,
    padding_width,
    out_depth,
    out_height,
    out_width,
):
    d = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    s = d * (out_height * out_width) + h * out_width + w
    oc = tl.thread_id(0)

    input_d_start = d * stride_depth
    input_h_start = h * stride_height
    input_w_start = w * stride_width

    acc = tl.zeros((1,), dtype=tl.float32)

    for k_d in tl.arange(0, kernel_depth):
        for k_h in tl.arange(0, kernel_height):
            for k_w in tl.arange(0, kernel_width):
                for in_ch in tl.arange(0, in_channels):
                    input_val = tl.load(
                        input_ptr + 
                        0 * in_channels * depth * height * width +
                        in_ch * depth * height * width +
                        (input_d_start + k_d) * height * width +
                        (input_h_start + k_h) * width +
                        (input_w_start + k_w),
                        mask=(input_d_start + k_d < depth) & 
                              (input_h_start + k_h < height) & 
                              (input_w_start + k_w < width)
                    )
                    weight_val = tl.load(
                        weight_ptr + 
                        oc * in_channels * kernel_depth * kernel_height * kernel_width +
                        in_ch * kernel_depth * kernel_height * kernel_width +
                        k_d * kernel_height * kernel_width +
                        k_h * kernel_width +
                        k_w
                    )
                    acc = acc + input_val * weight_val

    acc = acc + tl.load(bias_ptr + oc)
    tl.store(output_ptr + s * out_channels + oc, acc)


@torch.fx.wrap
def fused_conv3d_wrapper(in6, in1, in0):
    batch_size, in_channels, depth, height, width = in6.shape
    out_channels = in1.shape[0]
    kernel_depth = in1.shape[2]
    kernel_height = in1.shape[3]
    kernel_width = in1.shape[4]
    
    stride_depth, stride_height, stride_width = 2, 16, 16
    padding_depth, padding_height, padding_width = 0, 0, 0
    out_depth = (depth - kernel_depth + 2 * padding_depth) // stride_depth + 1
    out_height = (height - kernel_height + 2 * padding_height) // stride_height + 1
    out_width = (width - kernel_width + 2 * padding_width) // stride_width + 1
    seq_len = out_depth * out_height * out_width

    output = torch.empty((batch_size, seq_len, out_channels), dtype=in6.dtype, device=in6.device)
    grid = (out_depth, out_height, out_width)

    fused_conv3d_kernel[grid](
        in6,
        in1,
        in0,
        output,
        batch_size,
        in_channels,
        depth,
        height,
        width,
        out_channels,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        out_depth,
        out_height,
        out_width
    )

    return output

def replacement_func():
    return fused_conv3d_wrapper