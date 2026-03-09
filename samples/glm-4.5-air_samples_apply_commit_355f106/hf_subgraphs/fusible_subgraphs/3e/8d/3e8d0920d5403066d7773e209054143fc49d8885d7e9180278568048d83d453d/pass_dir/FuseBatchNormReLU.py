import torch
import triton
import triton.language as tl

# Pattern matching function for BatchNorm → ReLU pattern
def pattern(x, running_mean, running_var, weight, bias):
    # Match the BatchNorm → ReLU sequence from the graphs
    tmp_7 = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8

# Argument extraction function  
def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

# Optimized Triton kernel for batch norm + ReLU fusion
@triton.jit
def batchnorm_relu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position in one channel
    linear_idx = tl.program_id(0)
    channel_idx = linear_idx // (height * width)
    spatial_idx = linear_idx % (height * width)
    h = spatial_idx // width
    w = spatial_idx % width
    mask = (h < height) & (w < width)
    
    # Load normalization parameters for this channel
    mean_val = tl.load(running_mean_ptr + channel_idx)
    var_val = tl.load(running_var_ptr + channel_idx)
    weight_val = tl.load(weight_ptr + channel_idx)
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Load input data for this channel and spatial position
    x_offset = channel_idx * height * width + h * width + w
    x_val = tl.load(x_ptr + x_offset, mask=mask)
    
    # Apply batch normalization
    eps = 1e-5
    normalized = (x_val - mean_val) / tl.sqrt(var_val + eps)
    result = normalized * weight_val + bias_val
    
    # Apply ReLU activation
    final_result = tl.maximum(result, 0.0)
    
    # Store output
    output_offset = channel_idx * height * width + h * width + w
    tl.store(output_ptr + output_offset, final_result, mask=mask)

@torch.fx.wrap
def batchnorm_relu_fused(x, running_mean, running_var, weight, bias):
    # Get tensor shapes
    n_channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    
    # Total number of elements
    total_elements = n_channels * height * width
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate grid dimensions
    num_programs = (total_elements + 1023) // 1024  # Block size 1024
    
    # Launch kernel
    batchnorm_relu_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE=1024,
    )
    
    return output

# Replacement function
def replacement_func():
    return batchnorm_relu_fused