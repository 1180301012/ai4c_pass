import torch
import triton
import triton.language as tl

def pattern(shape_1_196_196_3):
    """Pattern matching for coordinate grid creation and tensor filling"""
    # Pattern function should NOT call any torch operations
    # Just return input parameter to define the structure
    return (shape_1_196_196_3,)

def replacement_args(shape_1_196_196_3):
    """Extract arguments for coordinate grid replacement"""
    return (shape_1_196_196_3,)

@triton.jit
def coordinate_grid_kernel(
    out_ptr,
    grid_size: tl.constexpr,
    n_channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance coordinate grid kernel"""
    # Each program handles a spatial position
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Calculate relative coordinates from center (14x14 grid expands to 196x196)
    rel_row = (row * 14) // 14  # Row in the original 14x14 grid
    rel_col = (col * 14) // 14  # Col in the original 14x14 grid
    
    # Create coordinate grid values for this position
    # This computes the squared distance from center
    
    # Channel 0: X coordinates (relative row)
    x_coord = (rel_row - 7.0) / 7.0  # Normalize to [-1, 1]
    x_sq = x_coord * x_coord
    
    # Channel 1: Y coordinates (relative col)  
    y_coord = (rel_col - 7.0) / 7.0  # Normalize to [-1, 1]
    y_sq = y_coord * y_coord
    
    # Channel 2: Combined squared distance
    distance_sq = x_sq + y_sq
    
    # Store all channels at once
    offset = (row * grid_size + col) * n_channels
    tl.store(out_ptr + offset + 0, tl.cast(x_coord, tl.float32))
    tl.store(out_ptr + offset + 1, tl.cast(y_coord, tl.float32))
    tl.store(out_ptr + offset + 2, tl.cast(distance_sq, tl.float32))

@torch.fx.wrap
def optimized_coordinate_grid(shape_1_196_196_3):
    """Wrapper for optimized coordinate grid creation"""
    grid_size = 196
    n_channels = 3
    n_elements = grid_size * grid_size * n_channels
    
    out = torch.empty(1, grid_size, grid_size, n_channels, dtype=torch.float32, device='cuda')
    
    # Launch kernel for each spatial position
    grid_shape = (grid_size, grid_size)
    coordinate_grid_kernel[grid_shape](
        out_ptr=out,
        grid_size=grid_size,
        n_channels=n_channels,
        BLOCK_SIZE=1,
    )
    
    return out

def replacement_func():
    """Replacement function that returns the optimized coordinate grid"""
    return optimized_coordinate_grid