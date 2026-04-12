import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    # Match adaptive_avg_pool2d + dropout + flatten pattern
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    n_channels,
    n_elements_per_channel,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    channel_idx = tl.program_id(0)
    if channel_idx >= n_channels:
        return
    
    # Calculate total elements in this channel
    total_elements = n_elements_per_channel
    
    # Load all elements for this channel
    offsets = channel_idx * n_elements_per_channel + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_channels * n_elements_per_channel
    
    # Load the channel data
    channel_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute average for this channel
    sum_val = tl.sum(channel_data)
    avg_val = sum_val / total_elements
    
    # Store the result at the corresponding position in output
    out_offset = channel_idx
    tl.store(out_ptr + out_offset, avg_val)

@torch.fx.wrap
def fused_global_avg_pool(x):
    # Input shape: [batch, channels, height, width]
    # Output shape: [batch, channels]
    batch_size, n_channels, height, width = x.shape
    n_elements_per_channel = height * width
    total_elements = batch_size * n_channels * height * width
    
    out = torch.empty((batch_size, n_channels), dtype=x.dtype, device=x.device)
    
    # Handle batch dimension - process each batch element
    for b in range(batch_size):
        # Process each channel in the batch
        num_channels = n_channels
        block_size = 1024
        
        # Launch kernel for this batch element
        global_avg_pool_kernel[(num_channels,)](
            x_ptr=x[b].data_ptr(),
            out_ptr=out[b].data_ptr(),
            n_channels=n_channels,
            n_elements_per_channel=n_elements_per_channel,
            BLOCK_SIZE=block_size,
        )
    
    return out

def replacement_func():
    return fused_global_avg_pool