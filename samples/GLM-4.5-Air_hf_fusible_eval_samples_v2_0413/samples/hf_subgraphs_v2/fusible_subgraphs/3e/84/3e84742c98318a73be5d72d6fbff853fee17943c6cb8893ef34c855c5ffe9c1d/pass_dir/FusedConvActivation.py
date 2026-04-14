import torch
import triton
import triton.language as tl

def pattern(conv_input, weight, bias, multiply_input):
    # This pattern matches: conv2d + add + div + clamp + multiply
    conv_result = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv_result + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    result = multiply_input * tmp_5
    return result

def replacement_args(conv_input, weight, bias, multiply_input):
    return (conv_input, weight, bias, multiply_input)

@triton.jit
def fused_conv_activation_kernel(
    conv_input_ptr, weight_ptr, bias_ptr, multiply_input_ptr, output_ptr,
    batch, total_elements_out,
    CONV_IN_CHANNELS: tl.constexpr, CONV_OUT_CHANNELS: tl.constexpr,
    MULTI_CHANNELS: tl.constexpr, MULTI_HEIGHT: tl.constexpr, MULTI_WIDTH: tl.constexpr
):
    # Program ID for each output element
    pid = tl.program_id(0)
    
    # Calculate which output element this program handles
    if pid >= total_elements_out:
        return
    
    # Convert linear index to batch, channel, height, width coordinates
    total_spatial_elements = MULTI_HEIGHT * MULTI_WIDTH
    total_channel_elements = total_spatial_elements * CONV_OUT_CHANNELS
    
    batch_idx = pid // total_channel_elements
    remainder = pid % total_channel_elements
    channel_out_idx = remainder // total_spatial_elements
    spatial_idx = remainder % total_spatial_elements
    h_idx = spatial_idx // MULTI_WIDTH
    w_idx = spatial_idx % MULTI_WIDTH
    
    # Load bias for current output channel
    bias_val = tl.load(bias_ptr + channel_out_idx).to(tl.float32)
    
    # Compute 1x1 convolution for this output channel
    # The conv operation: output[batch_idx, channel_out_idx, 0, 0] = sum(conv_input[batch_idx, i] * weight[channel_out_idx, i] for i in range(CONV_IN_CHANNELS)) + bias[channel_out_idx]
    conv_result = bias_val
    
    for c_in in range(CONV_IN_CHANNELS):
        # Load weight for current output and input channel: [CONV_OUT_CHANNELS, CONV_IN_CHANNELS]
        weight_offset = channel_out_idx * CONV_IN_CHANNELS + c_in
        weight_val = tl.load(weight_ptr + weight_offset).to(tl.float32)
        
        # Load conv input value: [batch, CONV_IN_CHANNELS, 1, 1] - all spatial positions are the same for 1x1 conv
        conv_input_offset = batch_idx * CONV_IN_CHANNELS + c_in
        conv_input_val = tl.load(conv_input_ptr + conv_input_offset).to(tl.float32)
        
        # Accumulate: weight * conv_input
        conv_result += weight_val * conv_input_val
    
    # Apply fused activation: (conv_result + 1.0) / 2.0 = conv_result * 0.5 + 0.5
    # This activation produces a single value per channel that gets broadcasted spatially
    fused_activation = conv_result * 0.5 + 0.5
    
    # Clamp to [0, 1]
    clamped_result = tl.maximum(tl.minimum(fused_activation, 1.0), 0.0)
    
    # Load multiply input for current spatial position
    multiply_input_offset = batch_idx * MULTI_CHANNELS * MULTI_HEIGHT * MULTI_WIDTH + \
                           channel_out_idx * MULTI_HEIGHT * MULTI_WIDTH + \
                           h_idx * MULTI_WIDTH + w_idx
    multiply_input_val = tl.load(multiply_input_ptr + multiply_input_offset).to(tl.float32)
    
    # Apply final multiplication with input features
    # The clamped_result is broadcasted to all spatial positions
    final_result = clamped_result * multiply_input_val
    
    # Store result at current spatial position
    output_offset = batch_idx * CONV_OUT_CHANNELS * MULTI_HEIGHT * MULTI_WIDTH + \
                   channel_out_idx * MULTI_HEIGHT * MULTI_WIDTH + h_idx * MULTI_WIDTH + w_idx
    tl.store(output_ptr + output_offset, final_result.to(tl.float32))

@torch.fx.wrap
def fused_conv_activation_impl(conv_input, weight, bias, multiply_input):
    # Handle different data types
    dtype = conv_input.dtype
    
    # Debug: print actual tensor shapes
    print(f"DEBUG conv_input.shape: {conv_input.shape}")
    print(f"DEBUG weight.shape: {weight.shape}")
    print(f"DEBUG bias.shape: {bias.shape}")
    print(f"DEBUG multiply_input.shape: {multiply_input.shape}")
    
    # Convert to float32 for computation if needed
    if dtype == torch.float16:
        conv_input_f = conv_input.float()
        weight_f = weight.float()
        bias_f = bias.float()
        multiply_input_f = multiply_input.float()
    elif dtype == torch.bfloat16:
        conv_input_f = conv_input.float()
        weight_f = weight.float()
        bias_f = bias.float()
        multiply_input_f = multiply_input.float()
    else:
        conv_input_f = conv_input
        weight_f = weight
        bias_f = bias
        multiply_input_f = multiply_input
    
    # Get input shapes
    batch, conv_in_channels, conv_height, conv_width = conv_input_f.shape
    conv_out_channels, weight_in_channels, _, _ = weight_f.shape
    _, multi_channels, multi_height, multi_width = multiply_input_f.shape
    
    print(f"DEBUG conv_input_f.shape: {conv_input_f.shape}")
    print(f"DEBUG weight_f.shape: {weight_f.shape}")
    print(f"DEBUG bias_f.shape: {bias_f.shape}")
    print(f"DEBUG multiply_input_f.shape: {multiply_input_f.shape}")
    print(f"DEBUG conv_in_channels: {conv_in_channels}")
    print(f"DEBUG conv_out_channels: {conv_out_channels}")
    print(f"DEBUG multi_channels: {multi_channels}")
    
    # Calculate output shape (should match multiply_input shape)
    out_channels = conv_out_channels
    
    # Create output tensor  
    output = torch.empty((batch, out_channels, multi_height, multi_width), dtype=dtype, device=conv_input.device)
    print(f"DEBUG output.shape: {output.shape}")
    
    # Calculate total number of output elements
    total_elements_out = batch * out_channels * multi_height * multi_width
    
    # Grid configuration - one program per output element
    num_programs = total_elements_out
    
    fused_conv_activation_kernel[(
        num_programs,
    )](
        conv_input_ptr=conv_input_f,
        weight_ptr=weight_f,
        bias_ptr=bias_f,
        multiply_input_ptr=multiply_input_f,
        output_ptr=output,
        batch=batch,
        total_elements_out=total_elements_out,
        CONV_IN_CHANNELS=conv_in_channels,
        CONV_OUT_CHANNELS=conv_out_channels,
        MULTI_CHANNELS=multi_channels,
        MULTI_HEIGHT=multi_height,
        MULTI_WIDTH=multi_width,
    )
    
    return output

def replacement_func():
    return fused_conv_activation_impl