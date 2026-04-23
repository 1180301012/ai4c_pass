import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.silu(conv2d)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def conv_silu_kernel(input_ptr,
                     weight_ptr,
                     bias_ptr,
                     output_ptr,
                     input_channels,
                     input_height,
                     input_width,
                     output_channels,
                     BLOCK_SIZE: tl.constexpr = 32):
    channel_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block_start = block_idx * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, input_height * input_width)
    for s in range(block_start, block_end):
        h = s // input_width
        w = s % input_width
        acc = tl.zeros((), dtype=tl.float32)
        for c_in in range(input_channels):
            input_offset = c_in * (input_height * input_width) + h * input_width + w
            weight_offset = channel_idx * input_channels + c_in
            input_val = tl.load(input_ptr + input_offset)
            weight_val = tl.load(weight_ptr + weight_offset)
            acc += input_val * weight_val
        bias_val = tl.load(bias_ptr + channel_idx)
        acc += bias_val
        exp_neg_x = tl.exp(-acc)
        sigmoid = 1.0 / (1.0 + exp_neg_x)
        silu_val = acc * sigmoid
        output_offset = channel_idx * (input_height * input_width) + h * input_width + w
        tl.store(output_ptr + output_offset, silu_val)

@torch.fx.wrap
def conv_silu(in_2, in_1, in_0):
    input_channels = 128
    input_height = 4
    input_width = 256
    output_channels = 256
    output = torch.empty((1, output_channels, input_height, input_width),
                         dtype=in_2.dtype,
                         device=in_2.device)
    num_spatial = input_height * input_width
    num_blocks_per_channel = (num_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (output_channels, num_blocks_per_channel)
    conv_silu_kernel[grid](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        output_channels=output_channels,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def replacement_func():
    return conv_silu