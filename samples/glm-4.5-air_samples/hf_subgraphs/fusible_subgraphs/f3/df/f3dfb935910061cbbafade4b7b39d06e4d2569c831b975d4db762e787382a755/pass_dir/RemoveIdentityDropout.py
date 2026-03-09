import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, y):
    tmp_0 = y + x  # Note: order is y + x to match original in_1 + in_0
    tmp_1 = torch.nn.functional.silu(tmp_0, inplace=False)
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    tmp_3 = tmp_2.flatten(1, -1)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return (tmp_4,)

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Optimized kernel that removes identity dropout
@triton.jit
def fused_kernel(x_ptr, y_ptr, out_ptr, n_batch, n_channels, height, width, BATCH_SIZE: tl.constexpr, CHANNEL_SIZE: tl.constexpr):
    # Program ID determines which batch and channel to process
    pid = tl.program_id(0)
    
    # Split program ID into batch and channel indices
    batch_idx = pid // n_channels
    channel_idx = pid % n_channels
    
    # Compute spatial dimensions
    spatial_size = height * width
    
    # Calculate pointer offsets
    base_offset = batch_idx * n_channels * spatial_size + channel_idx * spatial_size
    
    # Load spatial data for this specific batch and channel
    spatial_offsets = base_offset + tl.arange(0, spatial_size)
    mask = spatial_offsets < ((batch_idx + 1) * n_channels * spatial_size)
    
    x_channel = tl.load(x_ptr + spatial_offsets, mask=mask, other=0.0)
    y_channel = tl.load(y_ptr + spatial_offsets, mask=mask, other=0.0)
    
    # Elementwise addition + SiLU activation
    add_result = x_channel + y_channel
    silu_result = add_result * tl.sigmoid(add_result)
    
    # Adaptive avg pool to 1x1 (compute mean across spatial dimensions)
    channel_mean = tl.sum(silu_result) / spatial_size
    
    # Store result
    output_offset = batch_idx * n_channels + channel_idx
    tl.store(out_ptr + output_offset, channel_mean)

@torch.fx.wrap
def fused_operation(x, y):
    n_batch, n_channels, height, width = x.shape
    output_shape = (n_batch, n_channels)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate grid size - one program per (batch, channel) pair
    total_programs = n_batch * n_channels
    
    fused_kernel[(total_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_batch=n_batch,
        n_channels=n_channels,
        height=height,
        width=width,
        BATCH_SIZE=1,
        CHANNEL_SIZE=1
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_operation