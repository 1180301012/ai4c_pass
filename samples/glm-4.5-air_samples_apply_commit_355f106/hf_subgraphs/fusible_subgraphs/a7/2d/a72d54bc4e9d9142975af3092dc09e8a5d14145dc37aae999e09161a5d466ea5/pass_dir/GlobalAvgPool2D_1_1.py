import torch
import triton
import triton.language as tl

# Pattern matching function - matches adaptive_avg_pool2d with (1,1) output
def pattern(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized global average pooling kernel using simple summation
@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    n_channels,
    n_batch,
    H_in,
    W_in,
    BLOCK_SIZE_CHANNELS: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one batch element
    batch_idx = pid
    if batch_idx >= n_batch:
        return
    
    # Channel offset within this program
    channel_offset = tl.arange(0, BLOCK_SIZE_CHANNELS)
    channel_mask = channel_offset < n_channels
    
    # Initialize spatial sum for each channel
    spatial_sum = tl.zeros((BLOCK_SIZE_CHANNELS,), dtype=tl.float32)
    spatial_sum = tl.where(channel_mask, spatial_sum, 0.0)
    
    # Load and sum spatial locations efficiently
    for spatial_idx in range(H_in * W_in):
        # Calculate spatial coordinates
        h = spatial_idx // W_in
        w = spatial_idx % W_in
        
        # Load data for this spatial location across all channels
        x_vals = tl.load(
            x_ptr + batch_idx * n_channels * H_in * W_in + 
            channel_offset * H_in * W_in + spatial_idx,
            mask=channel_mask,
            other=0.0
        )
        spatial_sum += x_vals
    
    # Average by dividing by number of spatial elements
    spatial_avg = spatial_sum / (H_in * W_in)
    
    # Store result for each channel
    tl.store(
        out_ptr + batch_idx * n_channels + channel_offset,
        spatial_avg,
        mask=channel_mask
    )

# Kernel wrapper
@torch.fx.wrap
def optimized_global_avg_pool(x):
    N, C, H, W = x.shape
    
    out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    out_flat = out.view(N, C)
    
    # Block size for channels
    BLOCK_SIZE_CHANNELS = 512
    
    # Number of programs needed
    n_programs = (N + C - 1) // BLOCK_SIZE_CHANNELS
    
    global_avg_pool_kernel[(n_programs,)](
        x_ptr=x,
        out_ptr=out_flat,
        n_channels=C,
        n_batch=N,
        H_in=H,
        W_in=W,
        BLOCK_SIZE_CHANNELS=BLOCK_SIZE_CHANNELS,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_global_avg_pool