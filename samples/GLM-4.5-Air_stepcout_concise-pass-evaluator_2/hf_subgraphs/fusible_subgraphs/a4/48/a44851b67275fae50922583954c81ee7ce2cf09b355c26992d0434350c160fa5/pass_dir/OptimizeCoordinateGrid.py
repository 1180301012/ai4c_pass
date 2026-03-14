import torch
import triton
import triton.language as tl

# Pattern matching function for coordinate grid + distance computation
def pattern():
    """Pattern matches coordinate grid creation and distance computation"""
    # Create output tensor
    tmp_4 = torch.zeros(1, 196, 196, 3)
    
    # Create coordinate arrays
    tmp_5 = torch.arange(14)
    tmp_6 = tmp_5.view(1, -1)
    tmp_7 = torch.arange(14)
    tmp_8 = tmp_7.view(-1, 1)
    
    # Compute coordinate differences and grids
    tmp_9 = tmp_6 - tmp_8
    tmp_10 = tmp_9.repeat(14, 14)
    tmp_11 = tmp_9.repeat_interleave(14, dim=0)
    tmp_12 = tmp_11.repeat_interleave(14, dim=1)
    
    # Compute squared distances
    tmp_15 = tmp_10 ** 2 + tmp_12 ** 2
    
    # Assign to different channels
    tmp_16 = tmp_15.unsqueeze(0)
    tmp_18 = tmp_12.unsqueeze(0)
    tmp_20 = tmp_10.unsqueeze(0)
    tmp_4[..., 2] = tmp_16
    tmp_4[..., 1] = tmp_18
    tmp_4[..., 0] = tmp_20
    
    # Clean up intermediate values as in original
    tmp_17 = tmp_4
    tmp_16 = tmp_17 = None
    tmp_19 = tmp_4
    tmp_18 = tmp_19 = None
    tmp_21 = tmp_4
    tmp_20 = tmp_21 = None
    
    return tmp_4

# Argument extraction function
def replacement_args():
    return ()

# Optimized Triton kernel for coordinate grid + distance computation
@triton.jit
def coordinate_grid_kernel(
    out_ptr,
    grid_size: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position
    i = tl.program_id(0)  # row index (0-195)
    j = tl.program_id(1)  # column index (0-195)
    
    # Calculate which 14x14 block this position belongs to
    block_i = i // grid_size
    block_j = j // grid_size
    local_i = i % grid_size
    local_j = j % grid_size
    
    # Convert to coordinate system center in [-6.5, 6.5]
    coord_i = (local_i - grid_size // 2) if grid_size % 2 == 1 else (local_i - (grid_size - 1) / 2)
    coord_j = (local_j - grid_size // 2) if grid_size % 2 == 1 else (local_j - (grid_size - 1) / 2)
    
    # Calculate offsets for output tensor [1, 196, 196, 3]
    base_offset = i * spatial_size * 3 + j * 3
    
    # Channel 0: horizontal distance (x coordinate difference)
    tl.store(out_ptr + base_offset + 0, coord_j)
    
    # Channel 1: vertical distance (y coordinate difference)  
    tl.store(out_ptr + base_offset + 1, coord_i)
    
    # Channel 2: Euclidean distance squared
    distance_sq = coord_j * coord_j + coord_i * coord_i
    tl.store(out_ptr + base_offset + 2, distance_sq)

@triton.jit
def coordinate_grid_kernel_autotune(
    out_ptr,
    grid_size: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position
    i = tl.program_id(0)  # row index (0-195)
    j = tl.program_id(1)  # column index (0-195)
    
    # Calculate which 14x14 block this position belongs to
    block_i = i // grid_size
    block_j = j // grid_size
    local_i = i % grid_size
    local_j = j % grid_size
    
    # Convert to coordinate system center in [-6.5, 6.5]
    coord_i = (local_i - grid_size // 2) if grid_size % 2 == 1 else (local_i - (grid_size - 1) / 2)
    coord_j = (local_j - grid_size // 2) if grid_size % 2 == 1 else (local_j - (grid_size - 1) / 2)
    
    # Calculate offsets for output tensor [1, 196, 196, 3]
    base_offset = i * spatial_size * 3 + j * 3
    
    # Channel 0: horizontal distance (x coordinate difference)
    tl.store(out_ptr + base_offset + 0, coord_j)
    
    # Channel 1: vertical distance (y coordinate difference)  
    tl.store(out_ptr + base_offset + 1, coord_i)
    
    # Channel 2: Euclidean distance squared
    distance_sq = coord_j * coord_j + coord_i * coord_i
    tl.store(out_ptr + base_offset + 2, distance_sq)

# Optimized kernel with proper configuration
@triton.jit
def coordinate_grid_kernel(
    out_ptr,
    grid_size: tl.constexpr,
    spatial_size: tl.constexpr,
):
    # Each program handles one spatial position
    i = tl.program_id(0)  # row index (0-195)
    j = tl.program_id(1)  # column index (0-195)
    
    # Calculate local coordinates within 14x14 grid
    local_i = i % grid_size
    local_j = j % grid_size
    
    # Convert to coordinate system in [-6.5, 6.5]
    coord_i = local_j - (grid_size - 1) / 2.0
    coord_j = local_i - (grid_size - 1) / 2.0
    
    # Calculate offsets for output tensor [1, 196, 196, 3]
    base_offset = i * spatial_size * 3 + j * 3
    
    # Channel 0: horizontal distance (x coordinate difference)
    tl.store(out_ptr + base_offset + 0, coord_j)
    
    # Channel 1: vertical distance (y coordinate difference)  
    tl.store(out_ptr + base_offset + 1, coord_i)
    
    # Channel 2: Euclidean distance squared
    distance_sq = coord_j * coord_j + coord_i * coord_i
    tl.store(out_ptr + base_offset + 2, distance_sq)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_coordinate_grid():
    """Optimized coordinate grid computation using Triton"""
    spatial_size = 196
    grid_size = 14
    
    # Create output tensor on GPU
    out = torch.empty((1, spatial_size, spatial_size, 3), 
                     dtype=torch.float32, device='cuda')
    
    # Launch kernel with 2D grid for spatial positions
    grid = (spatial_size, spatial_size)
    
    # Launch the kernel
    coordinate_grid_kernel[grid](
        out_ptr=out,
        grid_size=grid_size,
        spatial_size=spatial_size,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_coordinate_grid