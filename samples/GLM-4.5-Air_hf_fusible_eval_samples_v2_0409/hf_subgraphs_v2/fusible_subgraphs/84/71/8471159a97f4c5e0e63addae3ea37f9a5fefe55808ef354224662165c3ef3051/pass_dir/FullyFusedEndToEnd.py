import torch
import triton
import triton.language as tl
import math

@triton.jit
def fully_fused_kernel(
    input_ptr, weight_ptr, bias_ptr, spatial_input_ptr, output_ptr,
    batch_size, in_features, channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one output element
    linear_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = batch_size * channels * height * width
    
    if linear_idx >= total_elements:
        return
    
    # Convert linear index to NCHW coordinates
    w_idx = linear_idx % width
    h_idx = (linear_idx // width) % height
    c_idx = (linear_idx // (width * height)) % channels
    b_idx = linear_idx // (width * height * channels)
    
    # Compute linear transformation for this channel and batch
    # y = x * weight^T + bias
    
    # Start with bias
    result = tl.load(bias_ptr + c_idx)
    
    # Multiply input features with weights for this channel
    weight_base = c_idx * in_features
    for k in range(0, in_features, 32):  # Process in warps
        if k < in_features:
            idx = tl.arange(0, 32)
            mask = k + idx < in_features
            
            # Load input for this sample
            input_offset = b_idx * in_features + tl.maximum(k + idx, 0)
            x_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
            
            # Load weights for this channel
            weight_offset = weight_base + tl.maximum(k + idx, 0)
            w_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
            
            # Accumulate
            result = tl.fma(x_val, w_val, result)
    
    # Apply sigmoid activation
    sigmoid_result = 1.0 / (1.0 + tl.exp(-result))
    
    # Load spatial input value for broadcasting
    spatial_input_offset = linear_idx
    spatial_input = tl.load(spatial_input_ptr + spatial_input_offset)
    
    # Final multiplication (broadcasting)
    final_result = spatial_input * sigmoid_result
    
    # Store result
    tl.store(output_ptr + linear_idx, final_result)

@torch.fx.wrap
def fully_fused_forward(input_flat, weight, bias, spatial_input):
    """
    Fully fused operation: Linear + Sigmoid + View + Multiplication
    
    Args:
        input_flat: [batch_size, in_features] - flattened input
        weight: [channels, in_features] - linear weight matrix
        bias: [channels] - linear bias
        spatial_input: [batch_size, channels, height, width] - spatial input tensor
    
    Returns:
        [batch_size, channels, height, width] - final result after all operations
    """
    batch_size, in_features = input_flat.shape
    channels = weight.shape[0]
    _, _, height, width = spatial_input.shape
    
    # Use optimal block size for GPU occupancy
    BLOCK_SIZE = 1024
    grid_size = (batch_size * channels * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    output = torch.empty_like(spatial_input)
    
    # Launch single fused kernel
    fully_fused_kernel[grid_size](
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
    tmp_3 = torch.sigmoid(linear)
    # Match exact view operation with specific batch dimension
    tmp_4 = tmp_3.view(x.shape[0], 64, 1, 1)
    result = spatial_input * tmp_4
    return result

def replacement_args(x, weight, bias, spatial_input):
    return (x, weight, bias, spatial_input)

def replacement_func():
    return fully_fused_forward