import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, height, width, in_channels, out_channels,
    stride_h: tl.constexpr, stride_w: tl.constexpr, 
    padding_h: tl.constexpr, padding_w: tl.constexpr,
    dilation_h: tl.constexpr, dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial location
    program_id = tl.program_id(0)
    
    # Extract coordinates from program ID
    spatial_locations = height * width
    batch_idx = program_id // spatial_locations
    spatial_idx = program_id % spatial_locations
    h = spatial_idx // width
    w = spatial_idx % width
    
    # Clamp to valid ranges
    batch_idx = tl.minimum(batch_idx, batch_size - 1)
    h = tl.minimum(h, height - 1)
    w = tl.minimum(w, width - 1)
    
    # Process output channels in blocks
    for out_c in range(0, out_channels, BLOCK_SIZE):
        out_channel_offsets = out_c + tl.arange(0, BLOCK_SIZE)
        out_channel_mask = out_channel_offsets < out_channels
        
        # Initialize convolution result for this block
        conv_result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Compute dot product over input channels for output channels in this block
        for in_c in range(0, in_channels):
            in_channel_mask = in_c < in_channels
            
            # Load input value for this input channel
            input_idx = batch_idx * height * width * in_channels + h * width * in_channels + w * in_channels + in_c
            input_vals = tl.load(input_ptr + input_idx)
            
            # Load weights for this input channel and output channels in the block
            weight_idx = out_channel_offsets * in_channels + in_c
            weight_vals = tl.load(weight_ptr + weight_idx, mask=out_channel_mask, other=0.0)
            
            # Multiply and accumulate
            conv_result += input_vals * weight_vals
        
        # Load bias for these output channels
        bias_vals = tl.load(bias_ptr + out_channel_offsets, mask=out_channel_mask, other=0.0)
        
        # Add bias
        final_result = conv_result + bias_vals
        
        # Apply sigmoid element-wise
        sigmoid_result = 1.0 / (1.0 + tl.exp(-final_result))
        
        # Store output with permuted dimensions: [batch, height, width, channels]
        output_idx = (batch_idx * height * width * out_channels + 
                     h * width * out_channels + 
                     w * out_channels + 
                     out_channel_offsets)
        tl.store(output_ptr + output_idx, sigmoid_result, mask=out_channel_mask)


@torch.fx.wrap
def fused_conv_sigmoid_optimized(bias, weight, input_tensor):
    # Based on the pattern and typical conv2d usage:
    # bias should be 1D: [out_channels]
    # weight should be 4D: [out_channels, in_channels, 1, 1]
    # input_tensor should be 4D: [batch_size, in_channels, height, width]
    
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Create output tensor with shape [batch_size, height, width, out_channels]
    output = torch.empty((batch_size, height, width, out_channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid configuration: one program per spatial location (each handles multiple channels)
    spatial_locations = height * width
    grid_size = batch_size * spatial_locations
    
    # Adaptive block size selection based on tensor characteristics
    # Balance memory usage vs computational efficiency
    total_output_elements = out_channels
    
    # Choose optimal block size based on output channel count and tensor size
    if total_output_elements <= 8:
        optimal_block_size = 8
    elif total_output_elements <= 16:
        optimal_block_size = 16
    elif total_output_elements <= 32:
        optimal_block_size = total_output_elements  # Use exact size for small outputs
    else:
        optimal_block_size = 32  # Default for larger outputs
    
    # Launch Triton kernel - grid should be a tuple
    fused_conv_sigmoid_kernel[(grid_size,)](
        input_tensor,
        weight.view(-1),  # Flatten weight to [out_channels * in_channels]
        bias,
        output,
        batch_size, height, width, in_channels, out_channels,
        1, 1, 0, 0,  # stride_h, stride_w, padding_h, padding_w
        1, 1,        # dilation_h, dilation_w
        1,           # groups
        optimal_block_size,
    )
    
    return output

# Pattern matching function - must match the exact dataflow from model.py
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    # Note: The reshape operation is handled by the output shape in our optimized kernel
    # We return the permuted tensor which will be reshaped by the caller
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Replacement function - returns the optimized kernel function reference
def replacement_func():
    return fused_conv_sigmoid_optimized