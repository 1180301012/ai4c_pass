import torch
import triton
import triton.language as tl

@triton.jit
def complete_pipeline_kernel(
    input_ptr, weight_ptr, bias_ptr, spatial_input_ptr, output_ptr,
    batch_size, in_features, channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread processes one output spatial element
    thread_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_elements = batch_size * channels * height * width
    
    # Early exit for out-of-bounds threads
    if thread_idx >= output_elements:
        return
    
    # Calculate coordinates: [batch, channel, y, x]
    linear_idx = thread_idx
    w_idx = linear_idx % width
    h_idx = (linear_idx // width) % height
    c_idx = (linear_idx // (width * height)) % channels
    b_idx = linear_idx // (width * height * channels)
    
    # Load bias for this channel (scalar for this channel)
    bias_val = tl.load(bias_ptr + c_idx)
    
    # Compute linear transformation: dot product of input features and weights
    linear_val = 0.0
    for k in range(in_features):
        # Load input and weight for this feature
        input_offset = b_idx * in_features + k
        weight_offset = c_idx * in_features + k
        
        input_val = tl.load(input_ptr + input_offset)
        weight_val = tl.load(weight_ptr + weight_offset)
        
        linear_val = linear_val + input_val * weight_val
    
    # Add bias
    linear_val = linear_val + bias_val
    
    # Apply sigmoid activation
    sigmoid_val = 1.0 / (1.0 + tl.exp(-linear_val))
    
    # For broadcasting: [batch, channels, 1, 1] -> [batch, channels, height, width]
    # Store sigmoid broadcast value for all spatial positions in channel
    spatial_val = tl.load(spatial_input_ptr + thread_idx)
    result = spatial_val * sigmoid_val
    
    # Store final result
    tl.store(output_ptr + thread_idx, result)

@torch.fx.wrap
def optimized_complete_pipeline(input_flat, weight, bias, spatial_input):
    """
    Optimized complete pipeline: Linear + Sigmoid + Broadcast Multiply
    
    Args:
        input_flat: [batch_size, in_features] - flattened input
        weight: [channels, in_features] - linear weight matrix  
        bias: [channels] - linear bias
        spatial_input: [batch_size, channels, height, width] - spatial input tensor
    
    Returns:
        [batch_size, channels, height, width] - final result
    """
    batch_size, in_features = input_flat.shape
    channels = weight.shape[0]
    _, _, height, width = spatial_input.shape
    
    # Allocate output tensor
    output = torch.empty((batch_size, channels, height, width), 
                        dtype=input_flat.dtype, device=input_flat.device)
    
    # Use optimal block size for GPU efficiency
    BLOCK_SIZE = 1024
    grid_size = (batch_size * channels * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Execute complete pipeline in single kernel call
    complete_pipeline_kernel[grid_size](
        input_ptr=input_flat,
        weight_ptr=weight,
        bias_ptr=bias,
        spatial_input_ptr=spatial_input,
        output_ptr=output,
        batch_size=batch_size,
        in_features=in_features,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def pattern(x, weight, bias, spatial_input):
    linear = torch.nn.functional.linear(x, weight, bias)
    sigmoid = torch.sigmoid(linear)
    # Match view operation with batch dimension flexibility
    reshaped = sigmoid.view(x.shape[0], 64, 1, 1)
    result = spatial_input * reshaped
    return result

def replacement_args(x, weight, bias, spatial_input):
    return (x, weight, bias, spatial_input)

def replacement_func():
    return optimized_complete_pipeline