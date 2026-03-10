import torch
import triton
import triton.language as tl

# Pattern matching: hardtanh + adaptive_avg_pool2d + view + flatten
def pattern(x):
    tmp_0 = torch.nn.functional.hardtanh(x, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_0 = None
    tmp_2 = tmp_1.view(2, -1)
    tmp_1 = None
    tmp_3 = torch.flatten(tmp_2, 1)
    tmp_2 = None
    return (tmp_3,)

def replacement_args(x):
    return (x,)

@triton.jit
def spatial_mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPATIAL_TILE_SIZE: tl.constexpr,
):
    # Batch and channel program IDs
    batch_pid = tl.program_id(0)
    channel_pid = tl.program_id(1)
    
    # Compute masks
    batch_mask = batch_pid < batch_size
    channel_mask = channel_pid < channels
    mask = batch_mask & channel_mask
    
    if mask:
        # Calculate spatial offsets within this channel
        spatial_elements = height * width
        
        # Load spatial data in tiles
        spatial_sum = 0.0
        for i in range(0, spatial_elements, SPATIAL_TILE_SIZE):
            tile_mask = i + tl.arange(0, SPATIAL_TILE_SIZE) < spatial_elements
            
            # Calculate memory base offset for this spatial tile
            base_offset = batch_pid * channels * height * width + channel_pid * height * width + i
            
            # Load spatial data tile
            tile_data = tl.load(x_ptr + base_offset + tl.arange(0, SPATIAL_TILE_SIZE), 
                              mask=tile_mask,
                              other=0.0)
            
            # Add partial sum
            spatial_sum += tl.sum(tile_data)
        
        # Compute spatial mean
        spatial_mean = tl.cast(spatial_sum, x_ptr.dtype.element_ty) / spatial_elements
        
        # Store result
        out_offset = batch_pid * channels + channel_pid
        tl.store(out_ptr + out_offset, spatial_mean)

@torch.fx.wrap
def optimized_spatial_mean(x):
    batch_size, channels, height, width = x.shape
    
    # Use autotune to find optimal block sizes
    def meta_func(batch_size, channels, height, width):
        # Try different block sizes
        block_sizes_m = [1, 2, 4, 8]
        block_sizes_n = [32, 64, 128, 256]
        return max(1, (batch_size + block_sizes_m[-1] - 1) // block_sizes_m[-1]), max(1, (channels + block_sizes_n[-1] - 1) // block_sizes_n[-1])
    
    # Create output tensor
    out = torch.empty((batch_size, channels), dtype=x.dtype, device=x.device)
    
    # Determine optimal grid dimensions
    grid_m = (batch_size + 7) // 8  # Block size 8 for batch dimension
    grid_n = (channels + 127) // 128  # Block size 128 for channels
    
    # Choose spatial tile size based on spatial dimensions
    spatial_elements = height * width
    spatial_tile_size = min(256, max(64, spatial_elements // 4))  # Adaptive tile size
    
    # Launch kernel with spatial tiling
    spatial_mean_kernel[(grid_m, grid_n)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=8,
        BLOCK_SIZE_N=128,
        SPATIAL_TILE_SIZE=spatial_tile_size,
    )
    
    return out

def replacement_func():
    return optimized_spatial_mean