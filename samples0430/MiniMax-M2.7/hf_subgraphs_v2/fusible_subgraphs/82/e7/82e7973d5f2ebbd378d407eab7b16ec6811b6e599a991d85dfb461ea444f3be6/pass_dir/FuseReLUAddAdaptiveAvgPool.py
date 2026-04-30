import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern: ReLU(in_1) + in_0, followed by adaptive_avg_pool2d(input, 1)
    This is a common residual connection pattern in CNNs.
    """
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_relu_add_avgpool_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    n_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: ReLU(in_1) + in_0 -> adaptive_avg_pool2d(..., 1)
    
    The adaptive_avg_pool2d with output_size=1 computes the mean across 
    all spatial dimensions for each channel.
    """
    # Get program ID for channel (each program handles one channel)
    batch_pid = tl.program_id(0)
    channel_pid = tl.program_id(1)
    
    # Compute the starting offset for this channel
    channel_offset = channel_pid * spatial_size
    
    # Accumulator for sum
    accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process all spatial elements in blocks
    for spatial_start in range(0, spatial_size, BLOCK_SIZE):
        spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
        spatial_mask = spatial_offsets < spatial_size
        
        # Global indices
        offsets = batch_pid * n_channels * spatial_size + channel_pid * spatial_size + spatial_offsets
        
        # Load from in_0 and in_1
        in_0_val = tl.load(in_0_ptr + offsets, mask=spatial_mask, other=0.0)
        in_1_val = tl.load(in_1_ptr + offsets, mask=spatial_mask, other=0.0)
        
        # Compute ReLU(in_1) + in_0
        relu_in_1 = tl.where(in_1_val > 0, in_1_val, 0.0)
        fused_val = relu_in_1 + in_0_val
        
        # Accumulate
        accum = accum + fused_val.to(tl.float32)
    
    # Compute mean (adaptive_avg_pool2d with output_size=1)
    mean_val = accum / spatial_size
    
    # Store result
    output_offset = batch_pid * n_channels + channel_pid
    tl.store(out_ptr + output_offset, mean_val, mask=None)


@torch.fx.wrap
def fused_relu_add_avgpool(in_0, in_1):
    """
    Fused implementation of: ReLU(in_1) + in_0 followed by adaptive_avg_pool2d(..., 1)
    
    Input shapes: [batch, channels, H, W]
    Output shape: [batch, channels, 1, 1]
    """
    batch, channels, H, W = in_0.shape
    n_elements = batch * channels * H * W
    spatial_size = H * W
    
    # Allocate output tensor
    out = torch.empty((batch, channels, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Grid configuration: (batch, channels)
    # Each program handles one (batch, channel) combination
    grid = (batch, channels)
    
    # Block size - using 256 for good occupancy
    BLOCK_SIZE = 256
    
    fused_relu_add_avgpool_kernel[grid](
        in_0,
        in_1,
        out,
        n_elements,
        channels,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_relu_add_avgpool