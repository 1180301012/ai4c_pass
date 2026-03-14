import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_1x1_flatten_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, channels_in, channels_out, height, width,
    stride_input_b, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_co, stride_weight_ci,
    stride_output_b, stride_output_co, stride_output_hw,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized 1x1 convolution + flatten kernel.
    
    Conv2d with kernel_size=1, stride=1, padding=0 is equivalent to:
    output[b, c_out, h, w] = sum_c_in(input[b, c_in, h, w] * weight[c_out, c_in]) + bias[c_out]
    
    Flatten(dim=2) produces: output[b, c_out, h*w]
    """
    # Get position in the flattened output: [batch, channel_out, h*w]
    pid = tl.program_id(0)
    
    # Calculate total output elements
    total_elements = batch_size * channels_out * height * width
    
    # Check if this program should compute anything
    # Each program computes one output element
    if pid >= total_elements:
        return
    
    # Calculate output indices - we process (b, c_out, h, w)
    # Use modulo arithmetic to decode the flattened pid
    w_pos = pid % width
    tmp = pid // width
    h_pos = tmp % height
    tmp = tmp // height
    c_out_pos = tmp % channels_out
    b_pos = tmp // channels_out
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + c_out_pos)
    
    # Compute convolution: sum over input channels
    # output[b, c_out, h, w] = sum_c_in(input[b, c_in, h, w] * weight[c_out, c_in]) + bias[c_out]
    accumulator = 0.0
    
    # Process input channels in blocks
    for ci in range(0, channels_in, BLOCK_SIZE):
        ci_range = ci + tl.arange(0, BLOCK_SIZE)
        ci_mask = ci_range < channels_in
        
        # Load weight: [c_out, ci]
        weight_offsets = c_out_pos * stride_weight_co + ci_range * stride_weight_ci
        weight = tl.load(weight_ptr + weight_offsets, mask=ci_mask, other=0.0)
        
        # Load input: [b, ci, h, w]
        input_offsets = (b_pos * stride_input_b + 
                        ci_range * stride_input_c + 
                        h_pos * stride_input_h + 
                        w_pos * stride_input_w)
        input_val = tl.load(input_ptr + input_offsets, mask=ci_mask, other=0.0)
        
        accumulator += tl.sum(input_val * weight, axis=0)
    
    # Add bias
    result = accumulator + bias
    
    # Store output: [b, c_out, h*w] (flattened)
    # Flatten: h*w goes from 0 to height*width-1
    hw_flat = h_pos * width + w_pos
    output_idx = b_pos * stride_output_b + c_out_pos * stride_output_co + hw_flat * stride_output_hw
    tl.store(output_ptr + output_idx, result)


# This is the FX-wrapped version that handles symbolic tracing properly
@torch.fx.wrap
def conv2d_1x1_flatten_kernel_wrapper(bias, weight, input):
    """
    Wrapper function for the fused 1x1 conv + flatten kernel.
    
    Args:
        bias: [channels_out]
        weight: [channels_out, channels_in, 1, 1] 
        input: [batch, channels_in, height, width]
    
    Returns:
        output: [batch, channels_out, height*width] (flattened from dim 2)
    """
    # Extract dimensions from actual tensor shapes
    batch_size = input.shape[0]
    channels_in = input.shape[1]
    height = input.shape[2]
    width = input.shape[3]
    channels_out = weight.shape[0]
    
    # Prepare contiguous tensors for efficient access
    input = input.contiguous()
    weight = weight.reshape(channels_out, channels_in).contiguous()
    bias = bias.contiguous()
    
    # Output shape after flatten(dim=2): [batch, channels_out, height*width]
    output = torch.empty((batch_size, channels_out, height * width), 
                         dtype=torch.float32, device=input.device)
    
    # Calculate strides
    stride_input_b = input.stride(0)
    stride_input_c = input.stride(1)
    stride_input_h = input.stride(2)
    stride_input_w = input.stride(3)
    
    stride_weight_co = weight.stride(0)
    stride_weight_ci = weight.stride(1)
    
    stride_output_b = output.stride(0)
    stride_output_co = output.stride(1)
    stride_output_hw = output.stride(2)
    
    # Grid configuration
    total_elements = batch_size * channels_out * height * width
    BLOCK_SIZE = 64
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv2d_1x1_flatten_kernel[(num_programs,)](
        input, weight, bias, output,
        batch_size, channels_in, channels_out, height, width,
        stride_input_b, stride_input_c, stride_input_h, stride_input_w,
        stride_weight_co, stride_weight_ci,
        stride_output_b, stride_output_co, stride_output_hw,
        BLOCK_SIZE
    )
    
    return output


# Pattern matching function
def pattern(in_0, in_1, in_2):
    """
    Match the Conv2d + Flatten pattern.
    Conv2d with 1x1 kernel, stride=1, padding=0, dilation=1 followed by flatten(dim=2)
    """
    tmp_0 = in_0  # bias
    tmp_1 = in_1  # weight
    # conv2d(input, weight, bias, stride, padding, dilation, groups)
    # stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(tmp_2, 2)
    return tmp_3


# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for the replacement function.
    in_0: bias, in_1: weight, in_2: input
    """
    return (in_0, in_1, in_2)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return conv2d_1x1_flatten_kernel_wrapper