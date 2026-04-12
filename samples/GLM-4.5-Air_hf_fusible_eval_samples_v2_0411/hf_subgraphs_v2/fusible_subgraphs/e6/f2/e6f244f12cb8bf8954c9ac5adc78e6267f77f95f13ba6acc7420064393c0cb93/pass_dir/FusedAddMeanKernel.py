import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = 0 + in_0
    tmp_0 = tmp_0 + 0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim = True)
    return (tmp_1, tmp_2)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_add_mean_single_kernel(
    in_ptr,
    out_ptr,
    mean_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one complete sample (batch, channels)
    sample_idx = pid // channels
    channel_idx = pid % channels
    
    if sample_idx >= batch_size:
        return
    
    # Load input tensor
    in_base = sample_idx * channels * height * width + channel_idx * height * width
    in_offset = in_base + tl.arange(0, BLOCK_SIZE)
    
    # Ensure we don't read out of bounds
    mask = in_offset < sample_idx * channels * height * width + (channel_idx + 1) * height * width
    in_data = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    
    # Sum all elements in spatial dimensions and compute mean
    spatial_sum = tl.sum(in_data)
    spatial_count = tl.sum(tl.where(mask, 1, 0))
    
    # Store mean (already averaged over spatial dimensions)
    mean_offset = sample_idx * channels + channel_idx
    tl.store(mean_ptr + mean_offset, spatial_sum / spatial_count)

@torch.fx.wrap
def fused_add_mean_single(in_0):
    batch_size, channels, height, width = in_0.shape
    
    # Output for sum (same as input since no meaningful addition)
    out_sum = in_0
    
    # Output for mean
    out_mean = torch.empty((batch_size, channels, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Flatten to 2D for processing: [batch * channels, height * width]
    in_flat = in_0.reshape(batch_size * channels, height * width)
    
    # Number of programs needed
    total_elements = batch_size * channels
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_mean_single_kernel[(num_programs,)](
        in_ptr=in_flat,
        out_ptr=out_sum.reshape(batch_size * channels, height * width),
        mean_ptr=out_mean.reshape(batch_size * channels),
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_sum, out_mean

def replacement_func():
    return fused_add_mean_single