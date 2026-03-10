import torch
import triton
import triton.language as tl

def pattern(running_mean, running_var, weight, bias, input_tensor, source_tensor):
    # This matches the combined pattern of batch_norm + tensor slicing
    # Slice from the source tensor (this matches slice(N, None, None) for any N)
    sliced_tensor = source_tensor[slice(None, None, None), slice(0, None, None), slice(None, None, None), slice(None, None, None)]
    # Apply batch normalization to the input tensor
    batch_norm_out = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    return batch_norm_out, sliced_tensor

def replacement_args(running_mean, running_var, weight, bias, input_tensor, source_tensor):
    return (running_mean, running_var, weight, bias, input_tensor, source_tensor)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_channels,
    batch_size,
    height,
    width,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE_CHANNELS: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Determine which channel this program handles
    channel_idx = pid * BLOCK_SIZE_CHANNELS + tl.arange(0, BLOCK_SIZE_CHANNELS)
    channel_mask = channel_idx < n_channels
    
    # Load running mean and var for this channel block
    running_mean = tl.load(running_mean_ptr + channel_idx, mask=channel_mask, other=0.0)
    running_var = tl.load(running_var_ptr + channel_idx, mask=channel_mask, other=1.0)
    
    # Load weight and bias for this channel block
    weight = tl.load(weight_ptr + channel_idx, mask=channel_mask, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_mask, other=0.0)
    
    # Process each spatial position in the channel block
    for h in range(height):
        for w in range(width):
            # Calculate linear index for this spatial position
            spatial_offset = h * width + w
            
            # Load input data for this spatial position and channel block
            input_offset = spatial_offset * n_channels + channel_idx
            input_data = tl.load(input_ptr + input_offset, mask=channel_mask, other=0.0)
            
            # Apply batch normalization
            inv_std = 1.0 / tl.sqrt(running_var + eps)
            normalized_data = (input_data - running_mean) * inv_std * weight + bias
            
            # Store output
            output_offset = spatial_offset * n_channels + channel_idx
            tl.store(output_ptr + output_offset, normalized_data, mask=channel_mask)

@torch.fx.wrap
def optimized_batch_norm(running_mean, running_var, weight, bias, input_tensor):
    # Check input tensor shape: [batch_size, n_channels, height, width]
    batch_size, n_channels, height, width = input_tensor.shape
    
    # Move parameters to GPU if they're not already there
    running_mean = running_mean.to(input_tensor.device)
    running_var = running_var.to(input_tensor.device)
    weight = weight.to(input_tensor.device)
    bias = bias.to(input_tensor.device)
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Block size for channels - should divide n_channels evenly
    BLOCK_SIZE_CHANNELS = min(64, triton.next_power_of_2(n_channels))
    
    # Number of programs needed to cover all channels
    n_programs = (n_channels + BLOCK_SIZE_CHANNELS - 1) // BLOCK_SIZE_CHANNELS
    
    # Launch kernel
    batch_norm_kernel[(n_programs,)](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_channels=n_channels,
        batch_size=batch_size,
        height=height,
        width=width,
        eps=0.001,
        momentum=0.1,
        BLOCK_SIZE_CHANNELS=BLOCK_SIZE_CHANNELS,
    )
    
    return output

def replacement_func():
    return optimized_batch_norm