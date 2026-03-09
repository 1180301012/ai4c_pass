import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    tmp_2 = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.sigmoid()
    return tmp_3

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def simple_conv_sigmoid_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread computes one output element
    pid = tl.program_id(0)
    
    # Calculate output position: batch * out_channels + out_channel, spatial_pos
    total_elements = batch_size * out_channels * input_height * input_width
    if pid >= total_elements:
        return
    
    # Extract indices
    out_channel = pid % out_channels
    remaining = pid // out_channels
    batch_idx = remaining // (input_height * input_width)
    spatial_pos = remaining % (input_height * input_width)
    
    # Load bias
    bias_val = tl.load(bias_ptr + out_channel)
    acc = bias_val
    
    # Compute convolution sum over input channels
    for ic in range(in_channels):
        weight_val = tl.load(weight_ptr + out_channel * in_channels + ic)
        input_base = (batch_idx * in_channels + ic) * input_height * input_width
        x_addr = x_ptr + input_base + spatial_pos
        x_val = tl.load(x_addr)
        acc += weight_val * x_val
    
    # Store result with sigmoid activation
    output_base = (batch_idx * out_channels + out_channel) * input_height * input_width
    out_addr = out_ptr + output_base + spatial_pos
    tl.store(out_addr, tl.sigmoid(acc))

@torch.fx.wrap
def simple_conv2d_sigmoid_impl(x, weight, bias):
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, _, _ = weight.shape
    
    # Use smaller block size for better parallelization
    BLOCK_SIZE = 1024
    total_elements = batch_size * out_channels * input_height * input_width
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((batch_size, out_channels, input_height, input_width), dtype=torch.float32, device=x.device)
    
    simple_conv_sigmoid_kernel[(grid,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return simple_conv2d_sigmoid_impl