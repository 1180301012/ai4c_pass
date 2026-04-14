import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, hardtanh_input):
    # Match the exact computation pattern from the model
    conv2d = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(hardtanh_input, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return (tmp_4,)

def replacement_args(conv_input, conv_weight, conv_bias, hardtanh_input):
    return (conv_input, conv_weight, conv_bias, hardtanh_input)

@triton.jit
def fused_conv_hardtanh_mul_kernel(
    input_ptr, weight_ptr, bias_ptr, hardtanh_input_ptr,
    output_ptr,
    batch_size, in_channels, out_channels, 
    height, width, in_channels_height_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which block we process
    pid = tl.program_id(0)
    
    # Calculate offset range for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total elements and create bounds mask
    total_elements = batch_size * out_channels * height * width
    mask = offsets < total_elements
    
    # Calculate 4D indices for all elements in the block (vectorized)
    batch_idx = offsets // (out_channels * height * width)
    remaining = offsets % (out_channels * height * width)
    out_channel_idx = remaining // (height * width)
    spatial_idx = remaining % (height * width)
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Initialize output array
    conv_output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Vectorized bias loading and 1x1 convolution computation
    for c_in in range(in_channels):
        # Load bias values for all output channels in this block
        bias_vals = tl.load(bias_ptr + out_channel_idx, mask=mask, other=0.0)
        
        # Load input values for this input channel across all spatial positions
        input_positions = batch_idx * in_channels_height_width + c_in * height * width + h_idx * width + w_idx
        input_vals = tl.load(input_ptr + input_positions, mask=mask, other=0.0)
        
        # Load weight values for this input channel across all output channels
        weight_positions = out_channel_idx * in_channels + c_in
        weight_vals = tl.load(weight_ptr + weight_positions, mask=mask, other=0.0)
        
        # Compute weighted input contribution (vectorized across all elements in block)
        weighted_input = input_vals * weight_vals
        conv_output += weighted_input
    
    # Add bias values
    bias_vals = tl.load(bias_ptr + out_channel_idx, mask=mask, other=0.0)
    conv_output += bias_vals
    
    # Load hardtanh input values (vectorized)
    hardtanh_positions = batch_idx * out_channels * height * width + out_channel_idx * height * width + h_idx * width + w_idx
    hardtanh_vals = tl.load(hardtanh_input_ptr + hardtanh_positions, mask=mask, other=0.0)
    
    # Apply hardtanh activation: max(0, min(6, x)) - vectorized
    hardtanh_clamped = tl.where(hardtanh_vals < 0, 0.0, 
                              tl.where(hardtanh_vals > 6.0, 6.0, hardtanh_vals))
    
    # Final multiplication - multiply conv_output with hardtanh_output
    final_output = conv_output * hardtanh_clamped
    
    # Store results
    tl.store(output_ptr + offsets, final_output, mask=mask)

@torch.fx.wrap
def fused_conv_hardtanh_mul(conv_input, conv_weight, conv_bias, hardtanh_input):
    # Get tensor dimensions
    batch_size, in_channels, height, width = conv_input.shape
    out_channels = conv_bias.shape[0]
    
    # Create output tensor
    output_shape = (batch_size, out_channels, height, width)
    output = torch.empty(output_shape, dtype=conv_input.dtype, device=conv_input.device)
    
    # Calculate total elements and grid size
    total_elements = batch_size * out_channels * height * width
    in_channels_height_width = in_channels * height * width
    
    # Launch Triton kernel with proper grid settings
    # Use blocks of 256 elements for better GPU utilization
    BLOCK_SIZE = 256
    # Calculate grid size (ceiling division)
    import math
    grid_size = math.ceil(total_elements / BLOCK_SIZE)
    
    fused_conv_hardtanh_mul_kernel[(grid_size,)](
        input_ptr=conv_input,
        weight_ptr=conv_weight,
        bias_ptr=conv_bias,
        hardtanh_input_ptr=hardtanh_input,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        in_channels_height_width=in_channels_height_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_conv_hardtanh_mul