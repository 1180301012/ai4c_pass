import torch
import triton
import triton.language as tl

# Pattern for coordinate grid computation that avoids torch.arange
def pattern():
    """Pattern matches coordinate grid creation without using torch operations"""
    tmp_4 = torch.zeros(1, 196, 196, 3)
    
    # Create coordinate grids directly using tensor construction
    # Generate coordinate differences [-6.5, -5.5, ..., 6.5] representing distances
    coords = torch.tensor([-6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=torch.float32)
    
    # Create coordinate grids
    grid_i, grid_j = torch.meshgrid(coords, coords, indexing='ij')
    
    # Compute squared distance matrix
    distance_sq = grid_i ** 2 + grid_j ** 2
    
    # Create the 3-channel output
    tmp_4[0, :, :, 0] = grid_j.flatten()     # x coordinate differences
    tmp_4[0, :, :, 1] = grid_i.flatten()     # y coordinate differences  
    tmp_4[0, :, :, 2] = distance_sq.flatten() # squared distance
    
    return tmp_4

# Argument extraction function
def replacement_args():
    return ()

# Optimized Triton kernel for coordinate grid computation
@triton.jit
def coordinate_optimized_kernel(
    out_ptr,
    grid_size: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Each program handles a 2D block of the output
    h = tl.program_id(0) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)[:, None]
    w = tl.program_id(1) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)[None, :]
    
    # Bounds checking
    mask_h = h < spatial_size
    mask_w = w < spatial_size
    mask = mask_h & mask_w
    
    # Convert to coordinates in [-6.5, 6.5]
    coord_i = (w % grid_size) - (grid_size - 1) / 2.0
    coord_j = (h % grid_size) - (grid_size - 1) / 2.0
    
    # Calculate flattened output index
    base_offset = h * spatial_size * 3 + w * 3
    
    # Store coordinates and distance for all 3 channels
    tl.store(out_ptr + base_offset + 0, coord_j, mask=mask)
    tl.store(out_ptr + base_offset + 1, coord_i, mask=mask)
    tl.store(out_ptr + base_offset + 2, coord_j * coord_j + coord_i * coord_i, mask=mask)

@torch.fx.wrap
def optimized_coordinate_grid():
    """Optimized coordinate grid computation using Triton"""
    spatial_size = 196
    grid_size = 14
    
    # Create output tensor on GPU
    out = torch.empty((1, spatial_size, spatial_size, 3), 
                     dtype=torch.float32, device='cuda')
    
    # Use block-based execution for better GPU utilization
    block_size_x = 16
    block_size_y = 16
    
    grid_x = (spatial_size + block_size_x - 1) // block_size_x
    grid_y = (spatial_size + block_size_y - 1) // block_size_y
    grid = (grid_y, grid_x)
    
    coordinate_optimized_kernel[grid](
        out_ptr=out,
        grid_size=grid_size,
        spatial_size=spatial_size,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )
    
    return out

def replacement_func():
    return optimized_coordinate_grid