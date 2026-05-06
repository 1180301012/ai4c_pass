import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    silu = torch.nn.functional.silu(conv2d, inplace=False)
    dropout = torch.nn.functional.dropout(silu, 0.0, False, False)
    return dropout
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv_silu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    block_size,
):
    pid = tl.program_id(0)
    start_idx = pid * block_size
    for out_chan in range(block_size):
        out_chan_idx = start_idx + out_chan
        if out_chan_idx >= out_channels:
            continue
        
        total = 0.0
        for in_chan in range(in_channels):
            input_val = tl.load(input_ptr + (0 * in_channels * height * width + in_chan), tl.float32)
            weight_val = tl.load(weight_ptr + (out_chan_idx * in_channels + in_chan), tl.float32)
            total += input_val * weight_val
        total += tl.load(bias_ptr + out_chan_idx, tl.float32)
        
        x = total
        silu_val = x * (1.0 / (1.0 + tl.exp(-x)))
        tl.store(output_ptr + out_chan_idx, silu_val)

@torch.fx.wrap
def kernel_wrapper(input, weight, bias):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    height = input.shape[2]
    width = input.shape[3]
    out_channels = weight.shape[0]
    
    output = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    num_programs = (out_channels + 1024 - 1) // 1024
    
    fused_conv_silu_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        block_size=1024,
    )
    
    return output
def replacement_func():
    return kernel_wrapper