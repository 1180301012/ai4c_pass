import torch
import triton
import triton.language as tl

def pattern(in_3, in_4, in_2):
    conv1d = torch.conv1d(in_3, in_4, in_2, (2,), (15,), (1,), 16)
    gelu = torch.nn.functional.gelu(conv1d)
    return gelu

def replacement_args(in_3, in_4, in_2):
    return (in_3, in_4, in_2)

@triton.jit
def conv_gelu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    in_seq_len, out_seq_len, in_channels, out_channels, groups, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Basic convolution + GELU kernel
    # Fused computation for memory coalescing
    
    # Calculate current output position range
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_seq_len

    # Iterate over output channels
    for out_ch in tl.arange(0, out_channels):
        # Group index handling
        group_idx = out_ch // (out_channels // groups)
        in_ch_start = group_idx * (in_channels // groups)
        in_ch_end = in_ch_start + (in_channels // groups)

        # Accumulate convolution result
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for in_ch in tl.arange(in_ch_start, in_ch_end):
            for k in tl.arange(0, weight.shape[2]):
                input_pos = offsets * stride - padding + k
                valid = (input_pos >= 0) & (input_pos < in_seq_len)
                x_val = tl.load(x_ptr + in_ch * in_seq_len + input_pos, mask=valid, other=0.0)
                w_val = tl.load(weight_ptr + out_ch * (in_channels // groups) * weight.shape[2] + (in_ch - in_ch_start) * weight.shape[2] + k)
                acc += x_val * w_val

        # Apply bias and GELU
        bias_val = tl.load(bias_ptr + out_ch)
        out_val = acc + bias_val
        gelu_val = out_val * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (out_val + 0.044715 * out_val * out_val)))
        tl.store(out_ptr + out_ch * out_seq_len + offsets, gelu_val, mask=mask)

@torch.fx.wrap
def fused_conv_gelu(x, weight, bias):
    # Get input dimensions
    batch, in_channels, in_seq_len = x.shape
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    groups = 16

    # Calculate output dimensions
    output_seq_len = (in_seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Initialize output
    out = torch.empty((batch, out_channels, output_seq_len), dtype=x.dtype, device=x.device)

    # Setup Triton kernel grid
    BLOCK_SIZE = 32
    num_blocks = (output_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    conv_gelu_kernel[(num_blocks,)](
        x, weight, bias,
        out,
        in_seq_len, output_seq_len, in_channels, out_channels, groups, stride, padding, dilation,
        BLOCK_SIZE
    )

    return out

def replacement_func():
    return fused_conv_gelu