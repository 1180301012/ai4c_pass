import torch
import triton
import triton.language as tl

def pattern(bias, weights, input_tensor):
    # Perform 1x1 conv2d with specific parameters
    conv_out = torch.conv2d(input_tensor, weights, bias, (1, 1), (0, 0), (1, 1), 1)
    # Reshape to flatten spatial dimensions
    reshaped = conv_out.view(conv_out.shape[0], 1, -1)
    # Apply softmax along dimension 2
    softmax_out = torch.nn.functional.softmax(reshaped, 2, _stacklevel=5)
    # Add final dimension
    result = softmax_out.unsqueeze(-1)
    return result

def replacement_args(bias, weights, input_tensor):
    return (bias, weights, input_tensor)

@triton.jit
def fused_conv_exp_kernel(
    input_ptr, weight_ptr, bias_ptr,
    exp_output_ptr,
    batch_size, in_channels, height, width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Each program processes a spatial location for one batch element
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    spatial_size = height * width
    mask_spatial = spatial_idx < spatial_size
    
    # Load weights (1, in_channels, 1, 1) - load once and broadcast
    weights = tl.load(weight_ptr, mask=tl.arange(0, in_channels) < in_channels)
    
    # Load bias
    bias = tl.load(bias_ptr)
    
    # Load input for this batch and spatial location
    input_base = input_ptr + batch_idx * in_channels * height * width + spatial_idx * in_channels
    input_vals = tl.load(input_base + tl.arange(0, in_channels), mask=tl.arange(0, in_channels) < in_channels)
    
    # Compute 1x1 convolution: sum(input * weights) + bias
    conv_val = tl.sum(input_vals * weights) + bias
    
    # Apply exponential for softmax
    exp_val = tl.exp(conv_val)
    
    # Store exponential result
    exp_output_base = exp_output_ptr + batch_idx * spatial_size
    tl.store(exp_output_base + spatial_idx, exp_val, mask=mask_spatial)

@triton.jit
def softmax_norm_kernel(
    exp_input_ptr,
    output_ptr,
    batch_size, spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one spatial location per batch
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    mask_spatial = spatial_idx < spatial_size
    
    # Compute sum of exponentials for this batch
    sum_exp = 0.0
    exp_input_base = exp_input_ptr + batch_idx * spatial_size
    
    for s_idx in range(0, spatial_size, BLOCK_SIZE):
        offsets = s_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        exp_vals = tl.load(exp_input_base + offsets, mask=mask, other=0.0)
        sum_exp += tl.sum(exp_vals)
    
    # Apply normalization
    exp_val = tl.load(exp_input_base + spatial_idx)
    softmax_val = exp_val / sum_exp
    
    # Store result 
    output_base = output_ptr + batch_idx * spatial_size
    tl.store(output_base + spatial_idx, softmax_val, mask=mask_spatial)

@torch.fx.wrap
def fused_conv_softmax(bias, weights, input_tensor):
    batch_size, in_channels, height, width = input_tensor.shape
    spatial_size = height * width
    
    # Intermediate buffer for exponential results [batch_size, spatial_size]
    exp_buffer = torch.empty((batch_size, spatial_size), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Output should be [batch_size, 1, spatial_size, 1] to match original pattern
    output_shape = (batch_size, 1, spatial_size, 1)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Grid configuration for first kernel: (batch_size, spatial_size)
    grid_conv = (batch_size, spatial_size)
    
    # Launch convolution + exponential kernel
    fused_conv_exp_kernel[grid_conv](
        input_tensor, weights, bias,
        exp_buffer,
        batch_size, in_channels, height, width,
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=256,  # Tunable block sizes
    )
    
    # Launch normalization kernel - one program per batch and spatial location
    grid_norm = (batch_size, spatial_size)
    
    # Reshape exp_buffer to [batch_size, spatial_size] for kernel
    softmax_norm_kernel[grid_norm](
        exp_buffer,
        output.reshape(batch_size, spatial_size),  # Remove singleton dims
        batch_size, spatial_size,
        BLOCK_SIZE=1024,
    )
    
    # Ensure proper shape [batch_size, 1, spatial_size, 1]
    return output

def replacement_func():
    return fused_conv_softmax