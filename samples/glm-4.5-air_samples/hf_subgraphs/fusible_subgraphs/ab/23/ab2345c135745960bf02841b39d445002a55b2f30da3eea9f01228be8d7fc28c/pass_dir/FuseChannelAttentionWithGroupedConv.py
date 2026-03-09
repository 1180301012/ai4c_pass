import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Grouped Conv2D: in_3 [1,32,1,1] with weights in_1 [96,8,1,1] and bias in_0 [96], groups=4
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    # Sigmoid activation
    tmp_3 = torch.sigmoid(tmp_2)
    # Reshape (no-op in this case since shape is already [1,96,1,1])
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    # Broadcast multiplication with spatial feature map
    tmp_5 = in_2 * tmp_4
    # Make contiguous (for safety, though our kernel will handle this)
    tmp_6 = tmp_5.contiguous()
    # Return the final output
    return (tmp_6,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_attention_kernel(
    bias_ptr,                    # [96] bias tensor
    weight_ptr,                 # [96,8,1,1] weight tensor  
    input_spatial_ptr,         # [1,96,128,128] spatial input feature map
    global_pooling_ptr,        # [1,32,1,1] global average pooling input
    output_ptr,                # [1,96,128,128] output
    
    # Metadata
    input_spatial_h: tl.constexpr,  # 128
    input_spatial_w: tl.constexpr,  # 128
    input_channels: tl.constexpr,   # 32
    output_channels: tl.constexpr,  # 96
    groups: tl.constexpr,           # 4
    group_size: tl.constexpr,        # 24 (actual channels per group)
    group_input_channels: tl.constexpr,  # 8 (input_channels // groups)
    
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program indices: (spatial_blocks, channels)
    pid = tl.program_id(0)  # Linearized spatial position 
    pid_c = tl.program_id(1)  # channel index (0-95)
    
    # Check if channel is valid (0-95 for our case)
    if pid_c >= output_channels:
        return
    
    # Calculate spatial coordinates from linear index
    h = pid // input_spatial_w
    w = pid % input_spatial_w
    
    # Check if spatial position is valid
    if h >= input_spatial_h or w >= input_spatial_w:
        return
    
    # Calculate group info for this channel
    group_id = pid_c // group_size  # 0-3
    
    if group_id >= groups:
        return
    
    # Group-specific input channel offset (8 channels per group)
    group_input_start = group_id * group_input_channels
    
    # Load the specific input values for this group (8 channels) - same for ALL channels in this group
    input_offsets = group_input_start + tl.arange(0, group_input_channels)
    group_inputs = tl.load(global_pooling_ptr + input_offsets)
    
    # Load bias for this specific channel
    bias = tl.load(bias_ptr + pid_c)
    
    # Load weights for this output channel [8]
    weight_start = pid_c * input_channels + group_input_start
    weight_offsets = weight_start + tl.arange(0, group_input_channels)
    weights = tl.load(weight_ptr + weight_offsets)
    
    # Compute grouped convolution for this channel: sum over input channels  
    conv_output = tl.sum(weights * group_inputs)
    
    # Add bias and apply sigmoid
    attn = 1.0 / (1.0 + tl.exp(-(conv_output + bias)))
    
    # Store output for this specific spatial location and channel
    output_offset = h * input_spatial_w * output_channels + w * output_channels + pid_c
    tl.store(output_ptr + output_offset, attn)

@torch.fx.wrap
def fused_attention_kernel_wrapper(bias, weight, input_spatial, global_pooling):
    # Get tensor shapes and metadata
    input_spatial_shape = input_spatial.shape  # [1,96,128,128]
    global_pooling_shape = global_pooling.shape  # [1,32,1,1]
    
    input_spatial_h = input_spatial_shape[3]
    input_spatial_w = input_spatial_shape[2]
    input_channels = global_pooling_shape[1]
    output_channels = input_spatial_shape[1]
    groups = 4  # from original conv2d groups parameter
    group_size = output_channels // groups  # 24
    group_input_channels = input_channels // groups  # 8
    
    # Calculate launch grid - simplified 2D grid: (spatial_locations, channels)
    grid_spatial = input_spatial_h * input_spatial_w  # All spatial positions (16384)
    grid_channels = output_channels  # 96 total channels
    
    # Create output tensor
    output = torch.empty_like(input_spatial)
    
    # Launch kernel with 2D grid: (spatial_locations, channels)
    fused_attention_kernel[(
        grid_spatial,
        grid_channels
    )](
        bias,
        weight,
        input_spatial,
        global_pooling,
        output,
        input_spatial_h,
        input_spatial_w,
        input_channels,
        output_channels,
        groups,
        group_size,
        group_input_channels,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )
    
    return output

def replacement_func():
    return fused_attention_kernel_wrapper