import torch
import triton
import triton.language as tl

def pattern(tmp_1):
    # Pattern matches adaptive_avg_pool2d to (1,1) followed by flatten
    # Note: tmp_1 is the input to adaptive_avg_pool2d, and tmp_3 is the final output
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, (1, 1))
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(tmp_1):
    return (tmp_1,)

@triton.jit
def adaptive_pool_flatten_kernel(input_ptr, output_ptr, n_channels, spatial_size, BLOCK_SIZE: tl.constexpr):
    """Kernel that combines adaptive pooling to (1,1) and flattening"""
    pid = tl.program_id(0)
    
    # Each thread block processes one channel
    channel_idx = pid
    
    if channel_idx >= n_channels:
        return
    
    # Load all spatial values for this channel
    offsets = tl.arange(0, spatial_size)
    mask = offsets < spatial_size
    
    # Calculate global offset for this channel
    channel_offset = channel_idx * spatial_size
    spatial_ptr = input_ptr + channel_offset + offsets
    
    # Load spatial values
    spatial_values = tl.load(spatial_ptr, mask=mask, other=0.0)
    
    # Compute mean (equivalent to adaptive_avg_pool2d to (1,1))
    channel_mean = tl.sum(spatial_values) / spatial_size
    
    # Store result directly as flattened output (equivalent to flatten)
    output_offset = channel_idx
    tl.store(output_ptr + output_offset, channel_mean)

@torch.fx.wrap
def adaptive_pool_flatten(tmp_0):
    """Fuse adaptive_avg_pool2d to (1,1) and flatten operations"""
    batch_size, n_channels, height, width = tmp_0.shape
    spatial_size = height * width
    
    # Create output tensor [batch_size, n_channels] 
    output = torch.empty((batch_size, n_channels), dtype=tmp_0.dtype, device=tmp_0.device)
    
    # Process each batch item
    for b in range(batch_size):
        # Get input pointer for this batch
        input_ptr = tmp_0[b].data_ptr()
        
        # Get output pointer for this batch
        output_ptr = output[b].data_ptr()
        
        # Launch kernel
        n_programs = n_channels
        block_size = 256  # Process 256 spatial elements per thread
        
        adaptive_pool_flatten_kernel[(n_programs,)](
            input_ptr=input_ptr,
            output_ptr=output_ptr,
            n_channels=n_channels,
            spatial_size=spatial_size,
            BLOCK_SIZE=block_size
        )
    
    return output

def replacement_func():
    return adaptive_pool_flatten