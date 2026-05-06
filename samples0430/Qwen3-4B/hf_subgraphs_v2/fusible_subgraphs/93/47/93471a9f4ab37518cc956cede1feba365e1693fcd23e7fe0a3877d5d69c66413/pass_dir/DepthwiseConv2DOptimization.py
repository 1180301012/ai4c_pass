import torch
import triton
import triton.language as tl

def pattern(in_9, in_4):
    """Pattern matching function for depthwise convolution with 512 groups."""
    return torch.conv2d(input=in_9, weight=in_4, groups=512)

def replacement_args(in_9, in_4):
    """Extract arguments for the replacement kernel."""
    return (in_9, in_4)

@triton.jit
def depthwise_conv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size: tl.int32,
    channels: tl.int32,
    input_h: tl.int32,
    input_w: tl.int32,
    kernel_h: tl.int32,
    kernel_w: tl.int32,
    stride_h: tl.int32,
    stride_w: tl.int32,
    padding_h: tl.int32,
    padding_w: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    output_h = (input_h + 2 * padding_h - kernel_h) // stride_h
    output_w = (input_w + 2 * padding_w - kernel_w) // stride_w
    pid = tl.program_id(0)
    h = pid % output_h
    w = pid // output_h
    
    input_h_start = max(0, h * stride_h - padding_h)
    input_w_start = max(0, w * stride_w - padding_w)
    
    output_val = tl.zeros(1, tl.float32)
    for i in range(kernel_h):
        for j in range(kernel_w):
            input_idx = input_h_start + i
            weight_idx = input_w_start + j
            input_val = tl.load(input_ptr + (input_idx * input_w + weight_idx), mask=tl.ones(1), other=0.0)
            weight_val = tl.load(weight_ptr + (input_idx * input_w + weight_idx), mask=tl.ones(1), other=0.0)
            output_val += input_val * weight_val
    
    tl.store(output_ptr + (h * output_w + w), output_val)

@torch.fx.wrap
def depthwise_conv_wrapper(input, weight):
    batch_size = 1
    channels = 512
    input_h = 70
    input_w = 70
    kernel_h = 7
    kernel_w = 7
    stride_h = 1
    stride_w = 1
    padding_h = 3
    padding_w = 3
    grid_size = 128
    output = torch.empty_like(input)
    
    depthwise_conv_kernel[grid_size](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        input_h=input_h,
        input_w=input_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        BLOCK_SIZE=128,
    )
    return output

def replacement_func():
    return depthwise_conv_wrapper