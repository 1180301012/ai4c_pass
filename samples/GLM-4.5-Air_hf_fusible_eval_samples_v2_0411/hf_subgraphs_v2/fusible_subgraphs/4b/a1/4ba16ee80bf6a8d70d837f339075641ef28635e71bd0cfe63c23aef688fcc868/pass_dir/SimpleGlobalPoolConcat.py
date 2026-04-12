import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Match concatenation followed by global average pooling pattern
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def simple_global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    n_channels,
    elements_per_channel,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    channel_idx = tl.program_id(0)
    if channel_idx >= n_channels:
        return
    
    # Load the channel data
    offsets = channel_idx * elements_per_channel + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_channels * elements_per_channel
    
    channel_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute average for this channel
    sum_val = tl.sum(channel_data)
    avg_val = sum_val / elements_per_channel
    
    # Store the result
    tl.store(out_ptr + channel_idx, avg_val)

@torch.fx.wrap
def simple_global_pool_concat(in_0, in_1, in_2, in_3):
    batch_size = in_0.shape[0]
    inputs = [in_0, in_1, in_2, in_3]
    results = []
    
    for x in inputs:
        batch_size_x, n_channels, height, width = x.shape
        elements_per_channel = height * width
        
        # Process each batch element
        out = torch.empty((batch_size_x, n_channels), dtype=x.dtype, device=x.device)
        
        for b in range(batch_size_x):
            num_channels = n_channels
            block_size = min(1024, elements_per_channel)
            
            # Launch kernel for this batch element
            simple_global_avg_pool_kernel[(num_channels,)](
                x_ptr=x[b].data_ptr(),
                out_ptr=out[b].data_ptr(),
                n_channels=n_channels,
                elements_per_channel=elements_per_channel,
                BLOCK_SIZE=block_size,
            )
        
        results.append(out)
    
    # Concatenate the results along channel dimension
    return torch.cat(results, dim=1)

def replacement_func():
    return simple_global_pool_concat