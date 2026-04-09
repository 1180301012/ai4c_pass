import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(1, -1)  # Use batch size 1 as default
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_hardtanh_pool_mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
):
    # Each program handles ONE (batch, channel) pair
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting index for this (batch, channel) pair
    base_offset = batch_idx * (channels * height * width) + channel_idx * (height * width)
    
    # Initialize accumulators
    spatial_sum = 0.0
    valid_count = 0
    
    # Process each spatial element
    for h in range(height):
        for w in range(width):
            spatial_idx = base_offset + h * width + w
            
            # Check bounds
            if spatial_idx < batch_size * channels * height * width:
                # Load value
                val = tl.load(x_ptr + spatial_idx)
                
                # Apply hardtanh and accumulate
                clamped_val = tl.maximum(tl.minimum(val, 6.0), 0.0)
                spatial_sum += clamped_val
                valid_count += 1
    
    # Compute spatial mean
    if valid_count > 0:
        mean_val = spatial_sum / valid_count
    else:
        mean_val = 0.0
    
    # Store result at the output position
    output_idx = batch_idx * channels + channel_idx
    tl.store(out_ptr + output_idx, mean_val)


@torch.fx.wrap
def fused_forward_kernel(in_0):
    batch_size, channels, height, width = in_0.shape
    
    # Output shape: (batch_size, channels)
    out_shape = (batch_size, channels)
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Use 2D grid: (batch_size, channels)
    fused_hardtanh_pool_mean_kernel[(batch_size, channels)](
        x_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
    )
    
    return out


def replacement_func():
    return fused_forward_kernel