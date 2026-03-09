import torch
import triton
import triton.language as tl

def pattern(x):
    flattened = x.flatten(2)
    permuted = flattened.permute(0, 2, 1)
    return permuted

def replacement_args(x):
    return (x,)

@triton.jit
def efficient_fuse_flatten_permute_kernel(
    x_ptr, out_ptr, 
    n_batch, n_channels, spatial_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    total_elements = n_batch * n_channels * spatial_size
    
    # Each thread handles multiple elements for better occupancy
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate indices for each element
    batch_ids = offsets // (n_channels * spatial_size)
    channel_ids = (offsets // spatial_size) % n_channels
    spatial_ids = offsets % spatial_size
    
    # Input: flattened shape [batch, channels, spatial_size]
    input_offset = batch_ids * (n_channels * spatial_size) + channel_ids * spatial_size + spatial_ids
    
    # Output: permuted shape [batch, spatial_size, channels] 
    output_offset = batch_ids * (spatial_size * n_channels) + spatial_ids * n_channels + channel_ids
    
    # Load and store with vectorized operations
    x_data = tl.load(x_ptr + input_offset, mask=mask)
    tl.store(out_ptr + output_offset, x_data, mask=mask)

@torch.fx.wrap
def efficient_fused_flatten_permute(x):
    batch_size, channels, height, width = x.shape
    spatial_size = height * width
    total_elements = batch_size * channels * spatial_size
    
    # Prepare output tensor with permuted dimensions: [batch, spatial_size, channels]
    out = torch.empty((batch_size, spatial_size, channels), dtype=x.dtype, device=x.device)
    
    # Optimal block size for GPU occupancy
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch 1D kernel for better efficiency
    efficient_fuse_flatten_permute_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_batch=batch_size,
        n_channels=channels,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return efficient_fused_flatten_permute