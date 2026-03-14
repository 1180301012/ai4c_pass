import torch
import triton
import triton.language as tl

# Pattern matching function for addition + layer norm fusion
def pattern(in_2, in_3, tmp_1, tmp_0):
    """Pattern matches: addition followed by layer normalization"""
    tmp_2 = in_2 + in_3
    # Use exact parameters from the original code
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (tmp_1.shape[0],), tmp_1, tmp_0, 1e-06)
    return tmp_2, tmp_3

# Argument extraction function
def replacement_args(in_2, in_3, tmp_1, tmp_0):
    return (in_2, in_3, tmp_1, tmp_0, 1e-06)

# Optimized Triton kernel for fused addition + layer norm
@triton.jit
def fused_add_layernorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    n_batch, n_height, n_channels,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel across all batches and positions
    channel_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    height_idx = tl.program_id(2)
    
    # Calculate pointer offsets for this channel
    x_channel_offset = batch_idx * n_height * n_channels + height_idx * n_channels + channel_idx
    y_channel_offset = batch_idx * n_height * n_channels + height_idx * n_channels + channel_idx
    
    # Load channel-wise scalars for normalization weights
    weight_val = tl.load(weight_ptr + channel_idx)
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Load x and y values
    x_val = tl.load(x_ptr + x_channel_offset)
    y_val = tl.load(y_ptr + y_channel_offset)
    
    # Perform addition
    added = x_val + y_val
    
    # Compute mean and variance (over the channel dimension, but we only have one value)
    # For layer_norm with normalized_shape=(channels,), we normalize per position
    mean = added
    var = tl.zeros([], tl.float32)  # variance of single value is 0
    
    # Compute standard deviation
    std = tl.sqrt(var + eps)
    
    # Apply layer norm formula: (x - mean) / std * weight + bias
    normalized = (added - mean) / std
    out = normalized * weight_val + bias_val
    
    # Store result
    out_offset = batch_idx * n_height * n_channels + height_idx * n_channels + channel_idx
    tl.store(out_ptr + out_offset, out)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_add_layernorm(x, y, weight, bias, eps):
    # Input shapes: x and y are [batch, height, channels]
    batch_size, height, channels = x.shape
    
    # Launch kernel with 3D grid: (channels, batch, height)
    grid = (channels, batch_size, height)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the kernel
    fused_add_layernorm_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_batch=batch_size,
        n_height=height,
        n_channels=channels,
        eps=eps,
        BLOCK_SIZE=1,  # Each program handles one value
    )
    
    added = x + y  # Return the addition result as well
    return added, out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_add_layernorm