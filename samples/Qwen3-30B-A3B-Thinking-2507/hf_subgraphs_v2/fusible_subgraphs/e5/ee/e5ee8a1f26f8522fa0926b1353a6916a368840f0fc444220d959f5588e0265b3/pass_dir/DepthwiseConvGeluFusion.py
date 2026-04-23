import torch
import triton
import triton.language as tl

@triton.jit
def depthwise_conv_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_batch, in_channels, in_height, in_width,
    out_channels, kernel_size,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    # Compute output indices
    idx = tl.program_id(0)
    batch_idx = idx // in_channels
    out_channel_idx = idx % in_channels
    out_height_idx = tl.program_id(1)
    out_width_idx = tl.program_id(2)

    # Calculate input region boundaries (with padding)
    in_height_start = out_height_idx * 1 - 1
    in_width_start = out_width_idx * 1 - 1

    # Initialize accumulator for convolution
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Calculate output buffer offset
    out_offset = (batch_idx * out_channels + out_channel_idx) * in_height * in_width + \
                 out_height_idx * in_width + out_width_idx

    # Load bias value (once per output channel)
    bias_val = tl.load(bias_ptr + out_channel_idx)

    # Iterate over kernel region
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            in_h = in_height_start + kh
            in_w = in_width_start + kw
            # Check input boundaries
            if in_h < 0 or in_h >= in_height or in_w < 0 or in_w >= in_width:
                continue
            
            # Calculate input element offset
            input_offset = (batch_idx * in_channels + out_channel_idx) * in_height * in_width + \
                           in_h * in_width + in_w
            input_val = tl.load(input_ptr + input_offset)
            
            # Calculate weight offset (depthwise convolution)
            weight_offset = (out_channel_idx * kernel_size + kh) * kernel_size + kw
            weight_val = tl.load(weight_ptr + weight_offset)
            
            # Accumulate convolution result
            acc += input_val * weight_val

    # Add bias
    acc += bias_val

    # Apply GELU activation (fast polynomial approximation)
    x = acc
    x3 = x * x * x
    gelu_val = 0.5 * x * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x3)))

    # Store result
    tl.store(output_ptr + out_offset, gelu_val)

@torch.fx.wrap
def depthwise_conv_gelu(input, weight, bias):
    batch, channels, in_height, in_width = input.shape
    _, _, kernel_h, kernel_w = weight.shape
    out_height = in_height
    out_width = in_width

    # Create output tensor
    output = torch.empty_like(input)

    # Configure kernel block sizes for optimal GPU utilization
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 8

    # Grid dimensions (batch x channels x height x width)
    grid = (batch * channels, out_height, out_width)

    # Launch kernel
    depthwise_conv_gelu_kernel[grid](
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        output.data_ptr(),
        in_batch=batch,
        in_channels=channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=channels,
        kernel_size=kernel_h,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C
    )

    return output

def pattern(in_2, in_1, in_0):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 128)
    tmp_3 = torch.nn.functional.gelu(conv2d)
    return tmp_3

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

def replacement_func():
    return depthwise_conv_gelu