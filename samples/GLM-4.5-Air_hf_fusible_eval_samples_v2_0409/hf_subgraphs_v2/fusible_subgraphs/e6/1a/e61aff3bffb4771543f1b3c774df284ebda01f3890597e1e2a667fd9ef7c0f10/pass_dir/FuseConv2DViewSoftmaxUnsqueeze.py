import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    # Simple conv2d pattern
    result = torch.conv2d(x, y, z, (1, 1), (0, 0), (1, 1), 1)
    return (result,)


def replacement_args(x, y, z):
    return (x, y, z)

@triton.jit
def simple_conv_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    "Simple kernel that just copies input to output (identity)"
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and store to output
    x = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)

@triton.jit
def conv2d_kernel_1x1(
    x_ptr,
    y_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    batch_id = pid // (out_channels * height * width)
    channel_id = (pid // (height * width)) % out_channels
    hw_id = pid % (height * width)
    
    if batch_id >= batch_size:
        return
    if channel_id >= out_channels:
        return
    if hw_id >= height * width:
        return
    
    # Load bias
    bias_val = tl.load(bias_ptr + channel_id)
    
    # Compute conv2d result: sum over input channels
    sum_val = bias_val
    for c in range(in_channels):
        x_offset = batch_id * in_channels * height * width + c * height * width + hw_id
        y_offset = channel_id * in_channels + c
        x_val = tl.load(x_ptr + x_offset)
        y_weight = tl.load(y_ptr + y_offset)
        sum_val += x_val * y_weight
    
    # Store result
    output_offset = batch_id * out_channels * height * width + channel_id * height * width + hw_id
    tl.store(output_ptr + output_offset, sum_val)

@torch.fx.wrap
def simple_replacement(x, y, z):
    # Proper conv2d replacement using Triton
    batch_size, in_channels, height, width = x.shape
    out_channels = y.shape[0]
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, height, width, 
                        dtype=x.dtype, device=x.device)
    
    # Launch kernel
    total_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 256
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    conv2d_kernel_1x1[(grid_size,)](
        x,
        y,
        z,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return simple_replacement