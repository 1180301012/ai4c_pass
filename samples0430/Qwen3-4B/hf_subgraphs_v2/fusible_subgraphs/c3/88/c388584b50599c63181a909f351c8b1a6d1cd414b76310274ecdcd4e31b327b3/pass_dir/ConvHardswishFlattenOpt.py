import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.int32,
    in_channels: tl.int32,
    out_channels: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    if i >= batch_size or j >= out_channels:
        return
    
    total = tl.zeros(tl.bfloat16, dtype=tl.bfloat16)
    for k in range(in_channels):
        input_k = tl.load(input_ptr + (i * in_channels + k))
        weight_k = tl.load(weight_ptr + (j * in_channels + k))
        total += input_k * weight_k
    
    bias_val = tl.load(bias_ptr + j)
    total += bias_val
    total = total * (1 + total.clamp(-6, 6) / 6.0)
    
    tl.store(output_ptr + (i * out_channels + j), total)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    batch_size = in_2.shape[0]
    in_channels = in_2.shape[1]
    out_channels = in_0.shape[0]
    output = torch.empty(batch_size, out_channels, dtype=in_0.dtype, device=in_0.device)
    
    grid = (batch_size, out_channels)
    optimized_kernel[grid](
        input_ptr=in_2.contiguous().data_ptr(),
        weight_ptr=in_1.contiguous().data_ptr(),
        bias_ptr=in_0.contiguous().data_ptr(),
        output_ptr=output.contiguous().data_ptr(),
        batch_size=tl.int32(batch_size),
        in_channels=tl.int32(in_channels),
        out_channels=tl.int32(out_channels),
        BLOCK_SIZE=128,
    )
    return output

def replacement_func():
    return kernel_wrapper