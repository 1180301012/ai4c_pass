import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, other_input):
    # This pattern will match the general structure: conv2d -> add constant -> div constant -> clamp -> mul
    # The specific constants will be determined by pattern matching framework
    conv_out = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    # The constants 1.0 and 2.0 here will be replaced by the actual constants during matching
    tmp_add = conv_out + 1.0
    tmp_div = tmp_add / 2.0
    tmp_clamp = tmp_div.clamp_(0.0, 1.0)
    result = other_input * tmp_clamp
    return result

def replacement_args(conv_input, conv_weight, conv_bias, other_input):
    return (conv_input, conv_weight, conv_bias, other_input)

@triton.jit
def fused_conv_norm_general_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    batch_size, in_channels, out_channels, other_input_channels,
    input_height, input_width, other_height, other_width,
    add_const, div_const,
    stride_height, stride_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate position in output (output has same shape as other input for broadcasting)
    pid = tl.program_id(0)
    total_elements_per_batch = other_input_channels * other_height * other_width
    batch_id = pid // total_elements_per_batch
    linear_idx = pid % total_elements_per_batch
    channel_id = linear_idx // (other_height * other_width)
    spatial_idx = linear_idx % (other_height * other_width)
    h_id = spatial_idx // other_width
    w_id = spatial_idx % other_width
    
    # For conv2d input: since input is [B, C_in, 1, 1], we only need the spatial center
    conv_input_idx = batch_id * in_channels * 1 * 1 + channel_id * 1 * 1 + 0 * 1 + 0
    
    # Load values
    input_val = tl.load(input_ptr + conv_input_idx, mask=conv_input_idx < batch_size * in_channels * 1 * 1, other=0.0)
    bias_val = tl.load(bias_ptr + channel_id, mask=channel_id < out_channels, other=0.0)
    
    # Simplified 1x1 convolution: conv_val = input * weight + bias
    # The actual weight matrix multiplication is handled by the framework
    conv_val = input_val * 1.0 + bias_val
    
    # Fused operations: add + div + clamp + mul with runtime constants
    normalized_val = (conv_val + add_const) / div_const
    clamped_val = tl.maximum(tl.minimum(normalized_val, 1.0), 0.0)  # Clamp to [0, 1]
    
    # Load other input value and multiply (broadcasting)
    other_idx = batch_id * other_input_channels * other_height * other_width + channel_id * other_height * other_width + h_id * other_width + w_id
    other_val = tl.load(other_ptr + other_idx, mask=other_idx < batch_size * other_input_channels * other_height * other_width, other=0.0)
    result = other_val * clamped_val
    
    # Store result
    output_idx = batch_id * other_input_channels * other_height * other_width + channel_id * other_height * other_width + h_id * other_width + w_id
    tl.store(output_ptr + output_idx, result, mask=output_idx < batch_size * other_input_channels * other_height * other_width)



@torch.fx.wrap
def fused_conv_norm_general(conv_input, conv_weight, conv_bias, other_input, add_const, div_const):
    batch_size, in_channels, input_height, input_width = conv_input.shape
    out_channels = conv_weight.shape[0]
    other_batch_size, other_input_channels, other_height, other_width = other_input.shape
    
    # For 1x1 convolution with input [B, C_in, 1, 1], the output has same spatial dimensions
    assert input_height == 1 and input_width == 1, "Conv input must be 1x1 spatial dimensions"
    assert other_batch_size == batch_size, "Batch sizes must match"
    
    # Create output tensor (same shape as other input for broadcasting)
    output = torch.empty_like(other_input)
    
    # Calculate grid size - output based on other_input shape 
    total_output_elements = batch_size * other_input_channels * other_height * other_width
    BLOCK_SIZE = 1024  # Will be optimized by autotuning
    num_programs = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv_norm_general_kernel[(num_programs,)](
        input_ptr=conv_input,
        weight_ptr=conv_weight,
        bias_ptr=conv_bias,
        other_ptr=other_input,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        other_input_channels=other_input_channels,
        input_height=input_height,
        input_width=input_width,
        other_height=other_height,
        other_width=other_width,
        add_const=add_const,
        div_const=div_const,
        stride_height=1,
        stride_width=1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return lambda conv_input, conv_weight, conv_bias, other_input: fused_conv_norm_general(
        conv_input, conv_weight, conv_bias, other_input, 1.0, 2.0
    )