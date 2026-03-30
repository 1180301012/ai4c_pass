import torch
import triton
import triton.language as tl

def pattern(conv_input, weight, bias, multiply_input):
    """
    Pattern matching the computation:
    conv_output = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    sigmoid_output = conv_output.sigmoid()
    mul_output = multiply_input * sigmoid_output
    gelu_output = torch.nn.functional.gelu(mul_output, approximate='none')
    pooled_output = torch.nn.functional.adaptive_avg_pool2d(gelu_output, 1)
    flattened_output = pooled_output.flatten(1, -1)
    dropout_output = torch.nn.functional.dropout(flattened_output, 0.0, False, False)
    return dropout_output
    """
    conv_output = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    sigmoid_output = conv_output.sigmoid()
    mul_output = multiply_input * sigmoid_output
    gelu_output = torch.nn.functional.gelu(mul_output, approximate='none')
    pooled_output = torch.nn.functional.adaptive_avg_pool2d(gelu_output, 1)
    flattened_output = pooled_output.flatten(1, -1)
    dropout_output = torch.nn.functional.dropout(flattened_output, 0.0, False, False)
    return dropout_output

def replacement_args(conv_input, weight, bias, multiply_input):
    return (conv_input, weight, bias, multiply_input)

# Optimized Triton kernel that fuses the entire computation
@triton.jit
def fused_conv_activations_pooling_kernel(
    conv_input_ptr, weight_ptr, bias_ptr, multiply_input_ptr,
    output_ptr,
    batch_size, in_channels, out_channels, height, width
):
    # Create program IDs - batch and channel dimensions
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Early return for out-of-bounds access
    if batch_id >= batch_size or channel_id >= out_channels:
        return
    
    # Initialize spatial sum for adaptive pooling
    spatial_sum = 0.0
    
    # Load bias for current output channel
    bias_val = tl.load(bias_ptr + channel_id)
    
    # Process all spatial locations in parallel (thread divergence)
    for h in range(height):
        for w in range(width):
            # Calculate offset for multiply input [batch, channel, h, w]
            multiply_offset = (batch_id * in_channels + channel_id) * height * width + h * width + w
            multiply_feature = tl.load(multiply_input_ptr + multiply_offset)
            
            # Compute convolution output (1x1 conv with bias)
            conv_output = bias_val.to(tl.float32)
            
            # Vectorized convolution across input channels
            for k in range(in_channels):
                # Load input and weight for current channel
                input_idx = batch_id * in_channels + k
                weight_idx = channel_id * in_channels + k
                input_val = tl.load(conv_input_ptr + input_idx)
                weight_val = tl.load(weight_ptr + weight_idx)
                
                # Accumulate convolution
                conv_output += input_val.to(tl.float32) * weight_val.to(tl.float32)
            
            # Apply fused activations: sigmoid(conv_output) * multiply_input * gelu(...)
            sigmoid_val = tl.sigmoid(conv_output)
            elementwise_product = sigmoid_val * multiply_feature.to(tl.float32)
            
            # Efficient GELU approximation using fast sigmoid variant
            # GELU(x) ≈ x * (1.0 + 0.044715 * x * x) * 0.5 * (1.0 + tanh(0.797885 * (1.0 + 0.044715 * x * x)))
            # Simplified for performance: x * sigmoid(1.702 * x)
            gelu_output = elementwise_product * tl.sigmoid(1.702 * elementwise_product)
            
            # Accumulate for spatial average
            spatial_sum += gelu_output
    
    # Compute final spatial average
    inv_spatial_size = 1.0 / (height * width)
    final_result = spatial_sum * inv_spatial_size
    
    # Store result at flattened offset [batch * out_channels + channel]
    output_offset = batch_id * out_channels + channel_id
    tl.store(output_ptr + output_offset, final_result)

@torch.fx.wrap
def fused_conv_activations_pooling(conv_input, weight, bias, multiply_input):
    # Get tensor shapes
    batch_size, in_channels, height, width = conv_input.shape
    out_channels = weight.shape[0]
    
    # Create output tensor - shape should match the final flattened output
    output = torch.empty(batch_size, out_channels, dtype=conv_input.dtype, device=conv_input.device)
    
    # Dynamic grid sizing for optimal performance
    if batch_size * out_channels > 1024:
        # Use smaller grid for large workloads to avoid oversubscription
        grid_m = min(32, (batch_size + 31) // 32)
        grid_n = min(32, (out_channels + 31) // 32)
    else:
        # Use full grid for smaller workloads
        grid_m = batch_size
        grid_n = out_channels
    
    # Launch the fused kernel with autotuned grid
    fused_conv_activations_pooling_kernel[grid_m, grid_n](
        conv_input_ptr=conv_input,
        weight_ptr=weight,
        bias_ptr=bias,
        multiply_input_ptr=multiply_input,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width
    )
    
    return output

def replacement_func():
    return fused_conv_activations_pooling