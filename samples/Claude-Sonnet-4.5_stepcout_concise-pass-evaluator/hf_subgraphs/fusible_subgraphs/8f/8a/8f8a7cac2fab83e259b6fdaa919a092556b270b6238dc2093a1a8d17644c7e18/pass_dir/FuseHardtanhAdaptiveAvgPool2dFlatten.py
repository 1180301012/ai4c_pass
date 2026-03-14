import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern to match:
    - hardtanh (ReLU6)
    - adaptive_avg_pool2d to (1, 1)
    
    Note: We don't include view and flatten as they're just reshaping operations
    that don't add computation. Our replacement will handle the reshaping.
    """
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_hardtanh_adaptive_pool_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for hardtanh + adaptive_avg_pool2d
    Simplified version for better performance
    """
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Base offset for this (batch, channel) slice
    base_offset = (batch_idx * channels + channel_idx) * spatial_size
    
    # Load and reduce
    sum_val = 0.0
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Load values
        vals = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        
        # Apply hardtanh: clamp to [0, 6]
        vals = tl.maximum(vals, 0.0)
        vals = tl.minimum(vals, 6.0)
        
        # Accumulate with mask
        vals = tl.where(mask, vals, 0.0)
        sum_val += tl.sum(vals)
    
    # Compute average and store
    avg = sum_val / spatial_size
    output_idx = batch_idx * channels + channel_idx
    tl.store(output_ptr + output_idx, avg)


@torch.fx.wrap
def fused_hardtanh_adaptive_pool(input_tensor):
    """
    Fused implementation of hardtanh + adaptive_avg_pool2d
    
    Input: [batch, channels, height, width]
    Output: [batch, channels, 1, 1]
    """
    batch_size, channels, height, width = input_tensor.shape
    spatial_size = height * width
    
    # Output buffer [batch_size, channels]
    temp_output = torch.empty((batch_size, channels), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Choose block size based on spatial size
    if spatial_size <= 64:
        BLOCK_SIZE = 64
    elif spatial_size <= 128:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 256
    
    # Launch kernel - one program per (batch, channel) pair
    num_programs = batch_size * channels
    
    fused_hardtanh_adaptive_pool_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=temp_output,
        batch_size=batch_size,
        channels=channels,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to [batch, channels, 1, 1] to match adaptive_avg_pool2d output
    output = temp_output.view(batch_size, channels, 1, 1)
    
    return output


def replacement_func():
    return fused_hardtanh_adaptive_pool