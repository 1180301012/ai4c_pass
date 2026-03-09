import torch
import triton
import triton.language as tl

def pattern(conv_input, weight, bias):
    # Match the exact computation structure from the original forward function
    tmp_0 = bias
    tmp_1 = weight
    tmp_2 = torch.conv2d(conv_input, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.permute(0, 2, 3, 1)
    tmp_2 = None
    channels = weight.shape[0]
    batch_size = conv_input.shape[0]
    tmp_4 = tmp_3.reshape(batch_size, -1, channels)
    tmp_3 = None
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    tmp_4 = None
    return tmp_5

def replacement_args(conv_input, weight, bias):
    # Extract arguments needed for the replacement
    channels = weight.shape[0]
    batch_size = conv_input.shape[0]
    height = conv_input.shape[2]
    width = conv_input.shape[3]
    return (conv_input, weight, bias, channels, batch_size, height, width)

@triton.jit
def fused_kernel(
    conv_input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    channels,
    batch_size,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    spatial_size = height * width
    n_elements = batch_size * spatial_size * channels
    
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate indices for the flattened output layout [batch, spatial_size, channels]
    batch_idx = offsets // (spatial_size * channels)
    spatial_idx = (offsets % (spatial_size * channels)) // channels
    channel_idx = offsets % channels
    
    # Convert back to original 4D coordinates for input loading
    input_h = spatial_idx // width
    input_w = spatial_idx % width
    input_c = channel_idx
    
    # Calculate input position: [batch, C_in, H, W]
    input_pos = batch_idx * (weight.shape[1] * height * width) + input_c * (height * width) + input_h * width + input_w
    
    # Load input data
    x = tl.load(conv_input_ptr + input_pos, mask=mask, other=0.0)
    
    # Load weight and bias for this channel (1x1 conv)
    weight_val = tl.load(weight_ptr + channel_idx, mask=channel_idx < channels, other=0.0)
    bias_val = tl.load(bias_ptr + channel_idx, mask=channel_idx < channels, other=0.0)
    
    # Apply 1x1 convolution and add bias
    conv_output = x * weight_val + bias_val
    
    # Apply sigmoid
    output = 1.0 / (1.0 + tl.exp(-conv_output))
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def fused_conv_sigmoid(conv_input, weight, bias, channels, batch_size, height, width):
    spatial_size = height * width
    total_elements = batch_size * spatial_size * channels
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty((batch_size, spatial_size, channels), dtype=torch.float32, device=conv_input.device)
    
    fused_kernel[(num_programs,)](
        conv_input_ptr=conv_input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        channels=channels,
        batch_size=batch_size,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_conv_sigmoid