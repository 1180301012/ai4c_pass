"""
Fuse Conv2d + GELU + Dropout(0) with an optimized block-based Triton kernel.

Uses a 2D blocking strategy for better GPU utilization:
- Grid: (batch, channel, out_height_blocks) with 2D spatial tiling
"""

import torch
import triton
import triton.language as tl


# GELU constants
GELU_SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)
GELU_SCALING = tl.constexpr(0.044715)


@triton.jit
def depthwise_conv3x3_gelu_kernel_v2(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    # Strides
    out_batch_stride,
    out_channel_stride,
    out_h_stride,
    out_w_stride,
    in_batch_stride,
    in_channel_stride,
    in_h_stride,
    in_w_stride,
    wt_channel_stride,
    wt_h_stride,
    wt_w_stride,
    # Shapes
    batch_size,
    num_channels,
    out_height,
    out_width,
    in_height,
    in_width,
    # Grid info
    h_blocks,
    w_blocks,
    # Padding
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Depthwise conv3x3 + GELU with 2D spatial tiling.
    
    Grid: (batch * channels * h_blocks * w_blocks,)
    Each program handles a BLOCK_H x BLOCK_W tile of output.
    """
    # Flatten everything to 1D
    program_id = tl.program_id(0)
    
    # Decode position
    w_block_id = program_id % w_blocks
    tmp = program_id // w_blocks
    h_block_id = tmp % h_blocks
    tmp2 = tmp // h_blocks
    c_block_id = tmp2 % num_channels
    batch_id = tmp2 // num_channels
    
    # Block boundaries
    h_start = h_block_id * BLOCK_H
    w_start = w_block_id * BLOCK_W
    
    # Thread offsets within block
    h_offs = h_start + tl.arange(0, BLOCK_H)
    w_offs = w_start + tl.arange(0, BLOCK_W)
    h_mask = h_offs < out_height
    w_mask = w_offs < out_width
    
    # Load bias for this channel
    bias_val = tl.load(bias_ptr + c_block_id)
    
    # Accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Convolution loop
    for kh in range(3):
        for kw in range(3):
            in_h = h_offs[:, None] + kh - padding_h
            in_w = w_offs[None, :] + kw - padding_w
            
            in_h_mask = (in_h >= 0) & (in_h < in_height)
            in_w_mask = (in_w >= 0) & (in_w < in_width)
            mask = h_mask[:, None] & w_mask[None, :] & in_h_mask & in_w_mask
            
            in_idx = (batch_id * in_batch_stride + c_block_id * in_channel_stride +
                      in_h * in_h_stride + in_w * in_w_stride)
            wt_idx = c_block_id * wt_channel_stride + kh * wt_h_stride + kw * wt_w_stride
            
            inp = tl.load(input_ptr + in_idx, mask=mask, other=0.0)
            wt = tl.load(weight_ptr + wt_idx)
            acc += inp * wt
    
    # GELU activation
    acc = acc + bias_val
    inner = GELU_SQRT_2_OVER_PI * (acc + GELU_SCALING * acc * acc * acc)
    exp_val = tl.exp(2.0 * inner)
    tanh_val = (exp_val - 1.0) / (exp_val + 1.0)
    result = acc * 0.5 * (1.0 + tanh_val)
    
    # Store
    out_idx = (batch_id * out_batch_stride + c_block_id * out_channel_stride +
               h_offs[:, None] * out_h_stride + w_offs[None, :] * out_w_stride)
    out_mask = h_mask[:, None] & w_mask[None, :]
    tl.store(output_ptr + out_idx, result, mask=out_mask)


def fused_conv2d_gelu(bias, weight, input_tensor):
    """Fused depthwise conv2d 3x3 + GELU with 2D tiling."""
    batch_size, num_channels, in_height, in_width = input_tensor.shape
    out_height = in_height
    out_width = in_width
    
    output = torch.empty(
        (batch_size, num_channels, out_height, out_width),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )
    
    out_b_stride = output.stride(0)
    out_c_stride = output.stride(1)
    out_h_stride = output.stride(2)
    out_w_stride = output.stride(3)
    
    in_b_stride = input_tensor.stride(0)
    in_c_stride = input_tensor.stride(1)
    in_h_stride = input_tensor.stride(2)
    in_w_stride = input_tensor.stride(3)
    
    wt_c_stride = weight.stride(0)
    wt_h_stride = weight.stride(2)
    wt_w_stride = weight.stride(3)
    
    # Choose block sizes based on output dimensions
    if out_height >= 56:
        block_h, block_w = 8, 8
    elif out_height >= 28:
        block_h, block_w = 7, 7
    elif out_height >= 14:
        block_h, block_w = 7, 7
    else:
        block_h, block_w = min(out_height, 7), min(out_width, 7)
    
    h_blocks = (out_height + block_h - 1) // block_h
    w_blocks = (out_width + block_w - 1) // block_w
    
    # Total programs: batch * channels * h_blocks * w_blocks
    num_programs = batch_size * num_channels * h_blocks * w_blocks
    
    depthwise_conv3x3_gelu_kernel_v2[(num_programs,)](
        output,
        input_tensor,
        weight,
        bias,
        out_b_stride, out_c_stride, out_h_stride, out_w_stride,
        in_b_stride, in_c_stride, in_h_stride, in_w_stride,
        wt_c_stride, wt_h_stride, wt_w_stride,
        batch_size, num_channels, out_height, out_width, in_height, in_width,
        h_blocks, w_blocks,
        1, 1,
        BLOCK_H=block_h,
        BLOCK_W=block_w,
    )
    
    return output


@torch.fx.wrap
def fused_conv2d_gelu_wrapper(in_0, in_1, in_2):
    """Fused Conv2d + GELU wrapper. Dropout(0) is identity - skipped."""
    return fused_conv2d_gelu(in_0, in_1, in_2)


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: conv2d -> gelu -> dropout(0.0)
    
    The dropout with p=0.0 is a no-op (identity function), so we fuse:
    conv2d + gelu + dropout(0) -> fused_conv2d_gelu
    
    Pattern matches:
    - torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
    - torch.nn.functional.gelu(conv_output)
    - torch.nn.functional.dropout(gelu_output, 0.0, training, inplace)
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 128)
    tmp_3 = torch.nn.functional.gelu(conv2d)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for the fused kernel.
    
    Args:
        in_0: bias tensor
        in_1: weight tensor  
        in_2: input tensor
        
    Returns:
        Tuple of (bias, weight, input) for the fused kernel
    """
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Return the fused conv2d + gelu function.
    """
    return fused_conv2d_gelu_wrapper