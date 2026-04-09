import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps):
    """Match batch normalization operation"""
    result = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps)
    return result

# Argument extraction function
def replacement_args(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps):
    return (input_tensor, running_mean, running_var, weight, bias, training, momentum, eps)

# Optimized BatchNorm kernel using Triton
@triton.jit
def batch_norm_kernel(
    input_ptr, 
    running_mean_ptr, 
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    batch_size,
    num_channels,
    input_height,
    input_width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load input
    input_val = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Calculate channel for each element in the block
    channel = offset % (input_height * input_width)
    channel = channel // (input_height * input_width) % num_channels
    
    # Load parameters for this channel (if within bounds)
    running_mean_val = 0.0
    running_var_val = 1.0
    weight_val = 1.0
    bias_val = 0.0
    
    if tl.any(channel < num_channels):
        running_mean_val = tl.load(running_mean_ptr + channel, mask=channel < num_channels, other=0.0)
        running_var_val = tl.load(running_var_ptr + channel, mask=channel < num_channels, other=1.0)
        weight_val = tl.load(weight_ptr + channel, mask=channel < num_channels, other=1.0)
        bias_val = tl.load(bias_ptr + channel, mask=channel < num_channels, other=0.0)
    
    # Broadcast parameters and convert to same dtype
    running_mean_val = running_mean_val.to(input_val.dtype)
    running_var_val = running_var_val.to(input_val.dtype)
    weight_val = weight_val.to(input_val.dtype)
    bias_val = bias_val.to(input_val.dtype)
    
    # Batch normalization computation using vectorized operations
    normalized = (input_val - running_mean_val) * tl.rsqrt(running_var_val + eps)
    output_val = normalized * weight_val + bias_val
    
    # Store result
    tl.store(output_ptr + offset, output_val, mask=mask)

# Optimized kernel with balanced precision and performance
@triton.jit
def batch_norm_kernel_optimized(
    input_ptr, 
    running_mean_ptr, 
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    batch_size,
    num_channels,
    input_height,
    input_width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load input
    input_val = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Calculate channel index
    spatial_elements = input_height * input_width
    channel = (offset // spatial_elements) % num_channels
    
    # Load parameters for each channel with proper bounds checking
    param_mask = channel < num_channels
    running_mean_val = tl.load(running_mean_ptr + channel, mask=param_mask, other=0.0)
    running_var_val = tl.load(running_var_ptr + channel, mask=param_mask, other=1.0)
    weight_val = tl.load(weight_ptr + channel, mask=param_mask, other=1.0)
    bias_val = tl.load(bias_ptr + channel, mask=param_mask, other=0.0)
    
    # Convert to match input type carefully
    running_mean_val = tl.cast(running_mean_val, input_val.dtype)
    running_var_val = tl.cast(running_var_val, input_val.dtype)
    weight_val = tl.cast(weight_val, input_val.dtype)
    bias_val = tl.cast(bias_val, input_val.dtype)
    
    # Batch normalization computation with proven stability
    sqrt_var = tl.sqrt(running_var_val + eps)
    normalized = (input_val - running_mean_val) / sqrt_var
    output_val = normalized * weight_val + bias_val
    
    # Store result
    tl.store(output_ptr + offset, output_val, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5):
    """Optimized batch normalization using stable Triton kernel"""
    if training:
        # For training mode, we'd need to compute running stats, but since our graph always uses training=False
        # we can focus on inference optimization
        raise NotImplementedError("Training mode not implemented in this optimized kernel")
    
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    batch_size, num_channels, input_height, input_width = input_tensor.shape
    
    # Use conservative block size for stability
    BLOCK_SIZE = 1024 if n_elements < 1000000 else 2048
    num_warps = 8
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch optimized kernel
    batch_norm_kernel_optimized[grid](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        batch_size=batch_size,
        num_channels=num_channels,
        input_height=input_height,
        input_width=input_width,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference, not called directly)
def replacement_func():
    return optimized_batch_norm