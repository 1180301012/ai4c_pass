import torch
import triton
import triton.language as tl

def pattern(bias, weight, input_tensor):
    """Conv2D (1x1) + Flatten pattern matching"""
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    flat_out = torch.flatten(conv_out, 2)
    return flat_out

def replacement_args(bias, weight, input_tensor):
    """Extract arguments for the fused kernel"""
    return (bias, weight, input_tensor)

@triton.jit
def fused_conv2d_flatten_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused Conv2D+Flatten kernel for 1x1 convolutions"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Decode pid_m into batch and channel indices
    # pid_m = batch_id * out_channels + channel_id
    batch_id = pid_m // out_channels
    channel_idx = pid_m % out_channels
    
    # Calculate output position after flattening
    # Output shape: [batch, out_channels, flattened_width] -> flattened to [batch * out_channels * flattened_width]
    flattened_width = height * width
    output_idx = batch_id * (out_channels * flattened_width) + channel_idx * flattened_width + pid_n
    
    # Decode pid_n into spatial indices
    # For flatten(dim=2), we only have spatial dimensions flattened: [batch, out_channels, height, width] -> [batch, out_channels, height*width]
    spatial_idx = pid_n
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Load bias for current output channel
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # For 1x1 conv: output[b, c_out, h, w] = bias[c_out] + sum_c_in(weight[c_out, c_in] * input[b, c_in, h, w])
    # Calculate base memory addresses for batch and spatial position
    input_base_idx = batch_id * (in_channels * height * width) + (h_idx * width + w_idx)
    weight_base_idx = channel_idx * in_channels
    
    sum_val = bias_val
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Load weight fragment for current output channel
        k_idx = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_idx < in_channels
        weight_vals = tl.load(weight_ptr + (weight_base_idx + k_idx), mask=k_mask, other=0.0)
        
        # Load corresponding input values for this channel group with coalesced memory access
        input_k_idx = input_base_idx + k_idx
        input_vals = tl.load(input_ptr + input_k_idx, mask=k_mask, other=0.0)
        
        # Element-wise multiplication and accumulation
        sum_val += tl.sum(weight_vals * input_vals)
    
    # Store output
    tl.store(output_ptr + output_idx, sum_val)

@torch.fx.wrap
def fused_conv2d_flatten(bias, weight, input_tensor):
    """Wrapper function for fused Conv2D+Flatten operation"""
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, _, _ = weight.shape
    
    # Calculate output shape
    # flatten(dim=2) transforms [batch, out_channels, height, width] -> [batch, out_channels, height*width]
    flattened_width = height * width
    output_shape = (batch_size, out_channels, flattened_width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size for flattened output
    # Each thread handles one element in the flattened spatial dimension
    grid_m = batch_size * out_channels  # Each thread handles one batch/out_channel combination
    grid_n = flattened_width  # Each thread handles one spatial position
    
    # Choose block sizes for optimal GPU occupancy based on tensor dimensions
    # Adapt block sizes based on problem size for better performance
    if batch_size <= 32:
        BLOCK_SIZE_M = 128   # Smaller blocks for small batches
    else:
        BLOCK_SIZE_M = 256   # Larger blocks for larger batches
        
    BLOCK_SIZE_N = 32      # Fixed spatial parallelism
    BLOCK_SIZE_K = 64      # Moderate channel parallelism
    
    # Use smaller, more efficient grid configuration with better occupancy
    grid_m = batch_size * out_channels
    grid_n = (flattened_width + 31) // 32  # Round up to boundary for better efficiency
    
    # Use optimal block sizes for better GPU utilization and memory coalescing
    BLOCK_SIZE_M = 64    # Optimized for warp efficiency
    BLOCK_SIZE_N = 32    # Good spatial parallelism
    BLOCK_SIZE_K = 32    # Balanced channel computation
    
    # Launch kernel with optimized configuration
    fused_conv2d_flatten_kernel[(grid_m, grid_n)](
        bias,
        weight,
        input_tensor,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    """Return the fused kernel wrapper function"""
    return fused_conv2d_flatten