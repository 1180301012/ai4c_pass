import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match adaptive_avg_pool2d followed by concatenation"""
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_pool_cat_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    batch_size, in_channels_0, in_height_0, in_width_0,
    in_channels_1, target_height, target_width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Compute grid position  
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    spatial_idx = spatial_idx.to(tl.int32)
    
    # Mask for valid indices
    batch_mask = batch_idx < batch_size
    spatial_mask = spatial_idx < (target_height * target_width)
    mask = batch_mask & spatial_mask
    
    if True:
        # Convert spatial index to 2D coordinates
        h_idx = spatial_idx // target_width
        w_idx = spatial_idx % target_width
        
        # Process input_0 (adaptive pooling) and input_1 (direct copy) more efficiently
        
        # Calculate interpolation coordinates once (same for all channels)
        src_y = h_idx * in_height_0 / target_height
        src_x = w_idx * in_width_0 / target_width
        
        # Get interpolation coordinates and weights
        y0 = tl.floor(src_y).to(tl.int32)
        y1 = tl.minimum(y0 + 1, in_height_0 - 1)
        x0 = tl.floor(src_x).to(tl.int32)
        x1 = tl.minimum(x0 + 1, in_width_0 - 1)
        
        fy = src_y - y0.to(tl.float32)
        fx = src_x - x0.to(tl.float32)
        
        # Precompute bilinear weights (vectorized)
        w00 = (1 - fy) * (1 - fx)
        w01 = (1 - fy) * fx
        w10 = fy * (1 - fx)
        w11 = fy * fx
        
        # Process all channels from both inputs in a single loop
        for c_total in range(in_channels_0 + in_channels_1):
            if c_total < in_channels_0:
                # Adaptive pooling for input_0
                c_0 = c_total
                
                # Compute base index for input_0
                base_idx_0 = (batch_idx * in_channels_0 + c_0) * in_height_0 * in_width_0
                
                # Load interpolated values with bounds checking
                idx_00 = base_idx_0 + y0 * in_width_0 + x0
                idx_01 = base_idx_0 + y0 * in_width_0 + x1
                idx_10 = base_idx_0 + y1 * in_width_0 + x0
                idx_11 = base_idx_0 + y1 * in_width_0 + x1
                
                val_00 = tl.load(in_0_ptr + idx_00, mask=(y0 < in_height_0) & (x0 < in_width_0))
                val_01 = tl.load(in_0_ptr + idx_01, mask=(y0 < in_height_0) & (x1 < in_width_0))
                val_10 = tl.load(in_0_ptr + idx_10, mask=(y1 < in_height_0) & (x0 < in_width_0))
                val_11 = tl.load(in_0_ptr + idx_11, mask=(y1 < in_height_0) & (x1 < in_width_0))
                
                # Bilinear interpolation
                result = w00 * val_00 + w01 * val_01 + w10 * val_10 + w11 * val_11
            else:
                # Direct copy for input_1
                c_1 = c_total - in_channels_0
                input_idx = (batch_idx * in_channels_1 + c_1) * target_height * target_width + spatial_idx
                result = tl.load(in_1_ptr + input_idx, mask=True)
            
            # Store result
            output_idx = (batch_idx * (in_channels_0 + in_channels_1) + c_total) * target_height * target_width + spatial_idx
            tl.store(out_ptr + output_idx, result, mask=mask)

@torch.fx.wrap
def fused_pool_cat_kernel_wrapper(in_0, in_1):
    batch_size, in_channels_0, in_height_0, in_width_0 = in_0.shape
    _, in_channels_1, target_height, target_width = in_1.shape
    
    # Output shape: [batch, in_channels_0 + in_channels_1, target_height, target_width]
    out_channels = in_channels_0 + in_channels_1
    out = torch.empty((batch_size, out_channels, target_height, target_width), 
                     dtype=in_0.dtype, device=in_0.device)
    
    # Use larger block size for better GPU occupancy and memory coalescing
    BLOCK_SIZE_N = 32  # Process 32 spatial elements per thread
    
    # Calculate grid dimensions
    grid_x = batch_size
    grid_y = (target_height * target_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel with optimized configuration
    BLOCK_SIZE_M = 1  # Each thread handles one batch element
    
    fused_pool_cat_kernel[grid_x, grid_y, 1](
        in_0, in_1, out,
        batch_size, in_channels_0, in_height_0, in_width_0,
        in_channels_1, target_height, target_width,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return fused_pool_cat_kernel_wrapper