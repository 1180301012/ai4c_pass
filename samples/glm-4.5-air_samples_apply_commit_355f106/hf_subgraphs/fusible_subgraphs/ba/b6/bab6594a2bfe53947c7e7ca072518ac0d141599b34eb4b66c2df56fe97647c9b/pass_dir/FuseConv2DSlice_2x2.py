import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (2, 2), (0, 0), (1, 1), 1)
    tmp_2 = tmp_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    return (tmp_2, tmp_1)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def conv2d_2x2_fused_kernel(
    weight_ptr,
    input_ptr,
    full_output_ptr,
    sliced_output_ptr,
    input_batch, input_channels, input_height, input_width,
    output_channels,
    stride_h: tl.constexpr, stride_w: tl.constexpr
):
    # Each program handles one output element
    n = tl.program_id(0)  # output channel
    c = tl.program_id(1)  # input channel  
    b = tl.program_id(2)  # batch
    h = tl.program_id(3)  # output height
    w = tl.program_id(4)  # output width
    
    # Check bounds
    if n >= output_channels or b >= input_batch or c >= input_channels:
        return
    
    # Calculate output dimensions for 2x2 stride 2
    output_height = (input_height + 2 * 0 - 1 * (2 - 1) - 1) // 2 + 1  # output_height/2
    output_width = (input_width + 2 * 0 - 1 * (2 - 1) - 1) // 2 + 1   # output_width/2
    
    if h >= output_height or w >= output_width:
        return
    
    # For 2x2 convolution with stride 2
    # out[b, n, h, w] = sum_c(in[b, c, 2*h, 2*w] * weight[n, c, 0, 0] +
    #                       in[b, c, 2*h+1, 2*w] * weight[n, c, 0, 1] +
    #                       in[b, c, 2*h, 2*w+1] * weight[n, c, 1, 0] +
    #                       in[b, c, 2*h+1, 2*w+1] * weight[n, c, 1, 1])
    acc = 0.0
    
    # Load weight matrix (2x2)
    for kh in range(2):
        for kw in range(2):
            weight_idx = n * input_channels * 4 + c * 4 + kh * 2 + kw
            weight_val = tl.load(weight_ptr + weight_idx)
            weight_val = weight_val  # For 1x1 weights, this will be simplified
            
            # Calculate input coordinates (strided)
            input_h = 2 * h + kh
            input_w = 2 * w + kw
            
            # Skip if out of bounds (padding is 0)
            if input_h < input_height and input_w < input_width:
                input_idx = b * input_channels * input_height * input_width + \
                           c * input_height * input_width + input_h * input_width + input_w
                input_val = tl.load(input_ptr + input_idx)
                acc += input_val * weight_val
    
    # Store both outputs at output position
    output_height = (input_height + stride_h - 1) // stride_h
    output_width = (input_width + stride_w - 1) // stride_w
    
    output_idx = b * output_channels * output_height * output_width + \
                 n * output_height * output_width + h * output_width + w
    
    tl.store(full_output_ptr + output_idx, acc)
    tl.store(sliced_output_ptr + output_idx, acc)

@torch.fx.wrap
def fused_conv2d_2x2(in_0, in_1):
    # Get input and output shapes
    batch_size, in_channels, in_height, in_width = in_1.shape
    out_channels = in_0.shape[0]  # weight shape is [out_channels, in_channels, 2, 2]
    
    # Calculate output dimensions for stride 2
    out_height = (in_height + 2 * 0 - 1 * (2 - 1) - 1) // 2 + 1
    out_width = (in_width + 2 * 0 - 1 * (2 - 1) - 1) // 2 + 1
    
    # Create output tensors
    full_output = torch.empty((batch_size, out_channels, out_height, out_width), dtype=in_1.dtype, device=in_1.device)
    sliced_output = torch.empty((batch_size, out_channels, out_height, out_width), dtype=in_1.dtype, device=in_1.device)
    
    # Create grid dimensions: one program per output element
    grid = (
        out_channels,           # output channels (n)
        in_channels,            # input channels (c) 
        batch_size,             # batch (b)
        out_height,             # output height (h)
        out_width               # output width (w)
    )
    
    # Launch kernel
    conv2d_2x2_fused_kernel[grid](
        in_0, in_1, full_output, sliced_output,
        batch_size, in_channels, in_height, in_width,
        out_channels,
        2, 2  # stride_h, stride_w
    )
    
    return sliced_output, full_output

def replacement_func():
    return fused_conv2d_2x2