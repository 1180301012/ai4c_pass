import torch
import triton
import triton.language as tl

def pattern(in_5, in_4, in_0, in_1, in_2, in_3):
    # Core computation after dropout removal
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_2, in_3, False, 0.1, 1e-05)
    return (tmp_8, tmp_5)

def replacement_args(in_5, in_4, in_0, in_1, in_2, in_3):
    return (in_5, in_4, in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel(
    in5_ptr, in4_ptr,
    mean_ptr, var_ptr,
    weight_ptr, bias_ptr,
    out_ptr, intermediate_ptr,
    batch_size, num_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute mean over spatial dimensions for each batch and channel
    # Each thread deals with one batch-channel pair
    
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    if batch_idx >= batch_size or channel_idx >= num_channels:
        return
        
    # Sum spatial locations for this batch-channel pair
    spatial_sum = 0.0
    spatial_count = 0
    
    for h in range(height):
        for w in range(width):
            # Compute linear index for this spatial location
            offset = ((batch_idx * num_channels + channel_idx) * height + h) * width + w
            input5_val = tl.load(in5_ptr + offset, mask=True, other=0.0)
            input4_val = tl.load(in4_ptr + offset, mask=True, other=0.0)
            spatial_sum += input5_val + input4_val
            spatial_count += 1
    
    # Compute mean for this batch-channel (this is the intermediate value)
    mean_val = spatial_sum / spatial_count
    
    # Load batch norm parameters for this channel
    running_mean = tl.load(mean_ptr + channel_idx)
    running_var = tl.load(var_ptr + channel_idx)
    weight_val = tl.load(weight_ptr + channel_idx)
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Apply batch normalization formula: y = weight * (x - running_mean) / sqrt(running_var + eps) + bias
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    result = weight_val * (mean_val - running_mean) * inv_std + bias_val
    
    # Store results - output and intermediate (the mean before scaling)
    output_idx = batch_idx * num_channels + channel_idx
    tl.store(out_ptr + output_idx, result)
    tl.store(intermediate_ptr + output_idx, mean_val)

@torch.fx.wrap
def fused_add_mean_bn(in_5, in_4, in_0, in_1, in_2, in_3):
    batch_size, num_channels, height, width = in_5.shape
    
    # Create output tensors
    output = torch.empty((batch_size, num_channels), dtype=in_5.dtype, device=in_5.device)
    intermediate = torch.empty((batch_size, num_channels), dtype=in_5.dtype, device=in_5.device)
    
    # Create 2D grid: (batch_size, num_channels)
    grid = (batch_size, num_channels)
    
    # Launch kernel with appropriate block size for spatial inner loops
    fused_kernel[grid](
        in5_ptr=in_5,
        in4_ptr=in_4,
        mean_ptr=in_0,
        var_ptr=in_1,
        weight_ptr=in_2,
        bias_ptr=in_3,
        out_ptr=output,
        intermediate_ptr=intermediate,
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE=1,  # We don't use BLOCK_SIZE for our 2D grid
    )
    
    return output, intermediate

def replacement_func():
    return fused_add_mean_bn