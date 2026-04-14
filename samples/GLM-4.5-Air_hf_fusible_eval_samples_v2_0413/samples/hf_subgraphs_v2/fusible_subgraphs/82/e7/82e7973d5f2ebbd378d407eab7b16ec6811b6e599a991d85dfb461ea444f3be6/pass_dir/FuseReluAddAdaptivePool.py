import torch
import triton
import triton.language as tl

def pattern(x, y):
    """End-to-end fusion: ReLU + Add + Adaptive Avg Pool2d (size 1)
    Matches the complete computation sequence: relu(x) + y -> adaptive_avg_pool2d(..., 1)
    """
    relu_result = torch.nn.functional.relu(x, inplace=False)
    add_result = relu_result + y
    final_result = torch.nn.functional.adaptive_avg_pool2d(add_result, 1)
    return final_result

def replacement_args(x, y):
    """Extract arguments for the end-to-end fused kernel"""
    return (x, y)

@triton.jit
def end_to_end_fused_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Complete fused kernel: adaptive_avg_pool2d(max(0, x) + y, 1)"""
    # Each program handles one batch channel combination for final output
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    elements_per_channel = height * width
    
    # Initialize sum for spatial mean calculation
    spatial_sum = 0.0
    
    # Process all spatial elements for this batch and channel
    for i in range(0, elements_per_channel, BLOCK_SIZE):
        offset_within_channel = i
        offsets = offset_within_channel + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_channel
        
        # Calculate global pointer offsets for x and y
        x_global_offset = batch_idx * channels * elements_per_channel + \
                         channel_idx * elements_per_channel + offsets
        y_global_offset = x_global_offset  # Assuming same spatial dimensions
        
        # Load x and y
        x_val = tl.load(x_ptr + x_global_offset, mask=mask, other=0.0)
        y_val = tl.load(y_ptr + y_global_offset, mask=mask, other=0.0)
        
        # Fused computation: max(0, x) + y
        relu_x = tl.maximum(x_val, 0.0)
        add_val = relu_x + y_val
        
        # Accumulate for spatial mean
        spatial_sum += add_val
    
    # Compute spatial mean
    spatial_mean = spatial_sum / elements_per_channel
    
    # Store final result
    out_offset = batch_idx * channels + channel_idx
    tl.store(out_ptr + out_offset, spatial_mean)

@torch.fx.wrap
def end_to_end_fused_relu_add_pool(x, y):
    """Wrapper for complete fused ReLU + Add + Adaptive Pool2d"""
    batch_size, channels, height, width = x.shape
    
    # Ensure tensors are on the same device
    if x.device != y.device:
        y = y.to(x.device)
    
    # Output shape is [batch_size, channels, 1, 1]
    output_shape = (batch_size, channels, 1, 1)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel if we have valid spatial dimensions
    if height * width > 0:
        BLOCK_SIZE = 1024
        num_programs = batch_size * channels
        
        end_to_end_fused_kernel[(num_programs,)](
            x_ptr=x,
            y_ptr=y,
            out_ptr=out.view(batch_size * channels),  # Flatten output
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out.view(batch_size, channels, 1, 1)

def replacement_func():
    """Return the end-to-end fused function"""
    return end_to_end_fused_relu_add_pool