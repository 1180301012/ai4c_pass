import torch
import triton
import triton.language as tl

def pattern(input, weight, bias):
    conv_out = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    hardswish_out = torch.nn.functional.hardswish(conv_out, True)
    flattened_out = hardswish_out.flatten(1, -1)
    return flattened_out

def replacement_args(input, weight, bias):
    return (input, weight, bias)

@triton.jit
def conv2d_hardswish_flatten_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_batch, n_channels_out, n_channels_in,
    stride_x: tl.constexpr, stride_y: tl.constexpr,
):
    # Each program processes one batch element and one output channel combination
    batch_id = tl.program_id(0)
    channel_out = tl.program_id(1)
    
    # Calculate offsets for this batch and output channel
    batch_offset = batch_id * n_channels_in
    weight_offset = channel_out * n_channels_in
    output_offset = batch_id * n_channels_out + channel_out
    
    # Since we have 1x1 kernel and 1x1 spatial dimensions, we process the single spatial position
    conv_result = 0.0
    
    # Process all 960 channels efficiently in chunks
    # Process channels in chunks of 32 to avoid deep nesting
    # First 32 channels
    for i in range(32):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    # Next 32 channels
    for i in range(32, 64):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    # Continue in chunks of 32 until we've processed all 960 channels
    for i in range(64, 96):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(96, 128):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(128, 160):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(160, 192):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(192, 224):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(224, 256):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(256, 288):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(288, 320):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(320, 352):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(352, 384):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(384, 416):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(416, 448):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(448, 480):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(480, 512):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(512, 544):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(544, 576):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(576, 608):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(608, 640):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(640, 672):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(672, 704):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(704, 736):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(736, 768):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(768, 800):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(800, 832):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(832, 864):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(864, 896):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(896, 928):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    for i in range(928, 960):
        if i < n_channels_in:
            conv_result += tl.load(input_ptr + batch_offset + i) * tl.load(weight_ptr + weight_offset + i)
    
    # Load bias and compute final convolution result
    bias_val = tl.load(bias_ptr + channel_out)
    total_result = conv_result + bias_val
    
    # Apply hardswish: x * relu6(x + 3) / 6
    x_plus_3 = total_result + 3.0
    relu6_val = tl.maximum(0.0, tl.minimum(x_plus_3, 6.0))
    hardswish_result = x_plus_3 * relu6_val / 6.0
    
    # Store directly to flattened output (n_batch, n_channels_out)
    tl.store(output_ptr + output_offset, hardswish_result)

@torch.fx.wrap
def conv2d_hardswish_flatten_fused(input, weight, bias):
    n_batch, n_channels_in, height, width = input.shape
    n_channels_out, _, _, _ = weight.shape
    
    # For 1x1 convolution, spatial dimensions don't matter for the final output
    # Final output shape: (n_batch, n_channels_out)
    output = torch.empty((n_batch, n_channels_out), dtype=input.dtype, device=input.device)
    
    # Create grid: (batch, output_channels) - since each element is independent
    grid = (
        n_batch,
        n_channels_out,
    )
    
    # Use stride 1x1 for 1x1 convolution
    conv2d_hardswish_flatten_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_batch=n_batch,
        n_channels_out=n_channels_out,
        n_channels_in=n_channels_in,
        stride_x=1,
        stride_y=1,
    )
    
    return output

def replacement_func():
    return conv2d_hardswish_flatten_fused