import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_float32_kernel(
    input_ptr, weight_ptr, bias_ptr, in_2_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate linear program index
    pid = tl.program_id(0)
    total_elements = batch_size * out_channels * in_height * in_width
    
    # Create mask to handle boundary conditions
    mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < total_elements
    
    # Calculate batch, channel, height, width from linear index
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    batch_idx = idx // (out_channels * in_height * in_width) 
    remaining = idx % (out_channels * in_height * in_width)
    channel_idx = remaining // (in_height * in_width)
    hw_idx = remaining % (in_height * in_width)
    height_idx = hw_idx // in_width
    width_idx = hw_idx % in_width
    
    # Validate indices
    batch_mask = batch_idx < batch_size
    channel_mask = channel_idx < out_channels  # Use out_channels for safety
    height_mask = height_idx < in_height
    width_mask = width_idx < in_width
    
    final_mask = mask & batch_mask & channel_mask & height_mask & width_mask
    
    # Calculate input positions
    input_pos = batch_idx * (in_channels * in_height * in_width) + height_idx * (in_width) + width_idx
    weight_pos = channel_idx * (in_channels * 1 * 1)  # For 1x1 conv
    bias_pos = channel_idx
    
    # Load input, weight, and bias
    input_val = tl.load(input_ptr + input_pos, mask=final_mask, other=0.0)
    weight_val = tl.load(weight_ptr + weight_pos, mask=final_mask, other=0.0)
    bias_val = tl.load(bias_ptr + bias_pos, mask=final_mask, other=0.0)
    
    # Calculate 1x1 conv result using proper indexing
    conv_result = input_val * weight_val + bias_val
    
    # Apply fused activation for the float32 pattern: (conv + 3.0) / 6.0 then clamp to [0.0, 1.0]
    norm_val = (conv_result + 3.0) / 6.0
    clamped_val = tl.clamp(norm_val, 0.0, 1.0)
    
    # Load corresponding value from in_2 
    in_2_pos = batch_idx * (out_channels * in_height * in_width) + channel_idx * (in_height * in_width) + height_idx * in_width + width_idx
    in_2_val = tl.load(in_2_ptr + in_2_pos, mask=final_mask, other=0.0)
    
    # Final multiplication
    output_val = clamped_val * in_2_val
    
    # Store output
    output_pos = batch_idx * (out_channels * in_height * in_width) + channel_idx * (in_height * in_width) + height_idx * in_width + width_idx
    tl.store(output_ptr + output_pos, output_val, mask=final_mask)

@torch.fx.wrap
def fused_conv_float32(input_3, weight, bias, in_2):
    # Get input shapes
    batch_size, in_channels, in_height, in_width = input_3.shape
    out_channels, _, _, _ = weight.shape
    
    # Output shape (since padding=0, kernel=1x1, stride=1x1)
    out_height, out_width = in_height, in_width
    
    # Determine optimal block size based on tensor sizes
    total_elements = batch_size * out_channels * in_height * in_width
    BLOCK_SIZE = 1024  # Fixed block size for better performance
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), 
                        dtype=input_3.dtype, device=input_3.device)
    
    # Launch kernel with simplified signature
    fused_conv_float32_kernel[(grid_size,)](
        input_3, weight, bias, in_2, output,
        batch_size, in_channels, in_height, in_width, out_channels,
        BLOCK_SIZE,
    )
    
    return output

def pattern(conv2d_input, weight, bias, in_2):
    # Match the float32 pattern from model.py (this one uses different constants)
    conv_result = torch.conv2d(conv2d_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv_result + 3.0  # Note: different constant here
    tmp_4 = tmp_3 / 6.0       # Note: different divisor here
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    
    # Must return all observable outputs as in the original model
    return tmp_6

def replacement_args(conv2d_input, weight, bias, in_2):
    return (conv2d_input, weight, bias, in_2)

def replacement_func():
    return fused_conv_float32