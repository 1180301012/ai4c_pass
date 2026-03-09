import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = tmp_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    return (tmp_2, tmp_1)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def conv2d_1x1_fused_kernel(
    weight_ptr,
    input_ptr,
    full_output_ptr,
    sliced_output_ptr,
    input_batch, input_channels, input_height, input_width,
    output_channels,
    batch_idx: tl.constexpr,
    out_idx: tl.constexpr,
    channel_idx: tl.constexpr,
    height_idx: tl.constexpr,
    width_idx: tl.constexpr
):
    # Each program handles one output element
    n = tl.program_id(0)
    c = tl.program_id(1)
    b = tl.program_id(2)
    h = tl.program_id(3)
    w = tl.program_id(4)
    
    # Check bounds
    if n >= output_channels or b >= input_batch or h >= input_height or w >= input_width:
        return
    
    # Perform 1x1 convolution: out[b, n, h, w] = sum_c(in[b, c, h, w] * weight[n, c, 0, 0])
    acc = 0.0
    for k in range(input_channels):
        # Load input element
        input_val = tl.load(input_ptr + b * input_channels * input_height * input_width + 
                           k * input_height * input_width + h * input_width + w)
        
        # Load weight element
        weight_val = tl.load(weight_ptr + n * input_channels + k)
        
        acc += input_val * weight_val
    
    # Store both outputs
    output_full_idx = b * output_channels * input_height * input_width + n * input_height * input_width + h * input_width + w
    tl.store(full_output_ptr + output_full_idx, acc)
    
    output_sliced_idx = b * output_channels * input_height * input_width + n * input_height * input_width + h * input_width + w
    tl.store(sliced_output_ptr + output_sliced_idx, acc)

@torch.fx.wrap
def fused_conv2d_1x1(in_0, in_1):
    # Get input and output shapes
    batch_size, in_channels, in_height, in_width = in_1.shape
    out_channels = in_0.shape[0]  # weight shape is [out_channels, in_channels, 1, 1]
    
    # Create output tensors
    full_output = torch.empty((batch_size, out_channels, in_height, in_width), dtype=in_1.dtype, device=in_1.device)
    sliced_output = torch.empty((batch_size, out_channels, in_height, in_width), dtype=in_1.dtype, device=in_1.device)
    
    # Create grid dimensions: one program per output element
    grid = (
        out_channels,           # output channels (n)
        in_channels,            # input channels (c) - needed for the channel loop
        batch_size,             # batch (b)
        in_height,              # height (h)
        in_width                # width (w)
    )
    
    # Launch kernel
    conv2d_1x1_fused_kernel[grid](
        in_0, in_1, full_output, sliced_output,
        batch_size, in_channels, in_height, in_width,
        out_channels,
    )
    
    return sliced_output, full_output

def replacement_func():
    return fused_conv2d_1x1