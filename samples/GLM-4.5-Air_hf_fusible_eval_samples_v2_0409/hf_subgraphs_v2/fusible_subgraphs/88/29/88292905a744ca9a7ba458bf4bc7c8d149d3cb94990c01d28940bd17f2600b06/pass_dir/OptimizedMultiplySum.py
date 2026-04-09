import torch
import triton
import triton.language as tl

# Optimized pattern focusing on the core computation
def pattern(tensor_group, input_features):
    """Optimized pattern for multiply and sum operations"""
    # The core computation: multiply and sum along dimension 1
    weighted = tensor_group * input_features
    result = torch.sum(weighted, dim=1)
    return result

# Argument extraction function
def replacement_args(tensor_group, input_features):
    return (tensor_group, input_features)

# Optimized Triton kernel using vectorized operations
@triton.jit
def optimized_multiply_sum_kernel(
    tensor_group_ptr,
    input_features_ptr, 
    output_ptr,
    batch_size,
    num_groups,
    channels,
    spatial_height,
    spatial_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element and one channel
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    # Calculate spatial size and offsets
    spatial_size = spatial_height * spatial_width
    group_stride = channels * spatial_size
    channel_offset = channel_idx * spatial_size
    output_offset = batch_idx * channels * spatial_size + channel_offset
    
    # Initialize accumulator for this batch and channel
    sum_val = 0.0
    
    # Process all groups for this batch and channel
    for group_idx in range(num_groups):
        # Calculate pointer offset for this group
        group_ptr = input_features_ptr + (
            batch_idx * num_groups + group_idx
        ) * group_stride + channel_offset
        
        # Vectorized load of spatial features
        spatial_vals = tl.load(
            group_ptr + tl.arange(0, BLOCK_SIZE),
            mask=tl.arange(0, BLOCK_SIZE) < spatial_size,
            other=0.0
        )
        
        # Load weight for this group and channel  
        weight_offset = (
            batch_idx * num_groups + group_idx
        ) * channels + channel_idx
        weight = tl.load(tensor_group_ptr + weight_offset)
        
        # Accumulate weighted sum
        sum_val += tl.sum(spatial_vals * weight)
    
    # Store final result
    tl.store(output_ptr + output_offset, sum_val)

@torch.fx.wrap
def optimized_multiply_sum_triton(tensor_group, input_features):
    batch_size = tensor_group.shape[0]
    num_groups = tensor_group.shape[1] 
    channels = tensor_group.shape[2]
    spatial_height = input_features.shape[3]
    spatial_width = input_features.shape[4]
    
    # Create output tensor
    output_shape = [batch_size, channels, spatial_height, spatial_width]
    output = torch.empty(output_shape, dtype=input_features.dtype, device=input_features.device)
    
    # Optimized grid and block size
    BLOCK_SIZE = 1024  # Vectorized spatial access
    grid_size = (batch_size * channels,)
    
    optimized_multiply_sum_kernel[grid_size](
        tensor_group,
        input_features,
        output,
        batch_size,
        num_groups,
        channels,
        spatial_height,
        spatial_width,
        BLOCK_SIZE
    )
    
    return output

# Replacement function returns the optimized Triton function
def replacement_func():
    return optimized_multiply_sum_triton