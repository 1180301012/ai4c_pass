import torch
import triton
import triton.language as tl

def pattern(in_6, in_4, in_0, in_1, in_3, in_2):
    conv = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    bn = torch.nn.functional.batch_norm(conv, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return bn

def replacement_args(in_6, in_4, in_0, in_1, in_3, in_2):
    return (in_6, in_4, in_0, in_1, in_3, in_2)

@triton.jit
def conv_bn_1x1_kernel(
    input_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    H,
    W,
    BLOCK_OUT_CH: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    c_out_block = tl.program_id(3)
    
    start_c_out = c_out_block * BLOCK_OUT_CH
    c_out = start_c_out + tl.thread_id(0)
    
    if c_out >= out_channels:
        return
    
    input_offset = b * in_channels * H * W + h * W + w
    input = tl.load(input_ptr + input_offset, (in_channels,))
    
    acc = tl.zeros((), dtype=tl.float32)
    for c_in in range(in_channels):
        weight_offset = c_out * in_channels + c_in
        w_val = tl.load(weight_ptr + weight_offset)
        acc = acc + input[c_in] * w_val
    
    scale_val = tl.load(scale_ptr + c_out)
    shift_val = tl.load(shift_ptr + c_out)
    output_val = acc * scale_val + shift_val
    
    output_offset = b * out_channels * H * W + c_out * H * W + h * W + w
    tl.store(output_ptr + output_offset, output_val)

@torch.fx.wrap
def kernel_wrapper(in_6, in_4, in_0, in_1, in_3, in_2):

    
    batch_size, _, H, W = in_6.shape
    _, in_channels, _, _ = in_4.shape
    out_channels = in_4.shape[0]
    
    output = torch.empty_like(in_6)
    
    BLOCK_OUT_CH = 64
    num_blocks_out = (out_channels + BLOCK_OUT_CH - 1) // BLOCK_OUT_CH
    grid = (batch_size, H, W, num_blocks_out)
    
    conv_bn_1x1_kernel[grid](
        in_6,
        in_4,
        scale,
        shift,
        output,
        batch_size,
        in_channels,
        out_channels,
        H,
        W,
        BLOCK_OUT_CH
    )
    
    return output

def replacement_func():
    return kernel_wrapper