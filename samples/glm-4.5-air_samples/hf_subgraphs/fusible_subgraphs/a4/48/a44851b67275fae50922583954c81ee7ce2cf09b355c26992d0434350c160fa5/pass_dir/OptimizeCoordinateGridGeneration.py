import torch
import triton
import triton.language as tl

def create_coordinate_pattern():
    """
    Matches the complex coordinate grid generation pattern:
    tmp_4 = torch.zeros(1, 196, 196, 3)
    tmp_5 = torch.arange(14)
    tmp_6 = tmp_5.view(1, -1)
    tmp_7 = torch.arange(14)  
    tmp_8 = tmp_7.view(-1, 1)
    tmp_9 = tmp_6 - tmp_8
    tmp_10 = tmp_9.repeat(14, 14)
    tmp_11 = tmp_9.repeat_interleave(14, dim=0)
    tmp_12 = tmp_11.repeat_interleave(14, dim=1)
    tmp_13 = tmp_10 ** 2
    tmp_14 = tmp_12 ** 2
    tmp_15 = tmp_13 + tmp_14
    tmp_16 = tmp_15.unsqueeze(0)
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_16
    tmp_18 = tmp_12.unsqueeze(0)
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_18
    tmp_20 = tmp_10.unsqueeze(0)
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_20
    return tmp_4
    """
    # This is a pattern that matches the entire coordinate generation sequence
    # We need to match on the fact that this produces a (1, 196, 196, 3) tensor
    # with specific coordinate values
    size = 196 // 14  # 14 because arange(14) is used
    tmp_4 = torch.zeros(1, size, size, 3)
    
    # This is just the declaration - the actual matching is more complex
    # since the pattern spans multiple lines with multiple intermediate variables
    return tmp_4

def pattern(x, y, z, w):
    # This pattern matches the full coordinate computation sequence
    # x, y, z, w are placeholders for the inputs that would create this pattern
    return create_coordinate_pattern()

def replacement_args(x, y, z, w):
    # We need to extract the size information - 196 comes from the input shapes
    return x.shape  # We'll use the second dimension (196) for grid size

@triton.jit
def coordinate_grid_kernel(
    out_ptr,
    grid_size,
    n_coords,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a 2D position in the coordinate grid
    batch_id = tl.program_id(0)  # Always 0 since batch=1
    h = tl.program_id(1)
    w = tl.program_id(2)
    coord_dim = tl.arange(0, 3)  # 3 coordinate dimensions
    
    # Compute coordinate offsets within each 14x14 block
    block_h = h // (grid_size // n_coords)
    block_w = w // (grid_size // n_coords)
    rel_h = h % (grid_size // n_coords)
    rel_w = w % (grid_size // n_coords)
    
    # Compute coordinate values
    coord_y = (block_h * n_coords + rel_h) / (grid_size - 1) * 2 - 1
    coord_x = (block_w * n_coords + rel_w) / (grid_size - 1) * 2 - 1
    
    # Compute squared coordinate distances
    coord_0 = coord_x  # X coordinate
    coord_1 = coord_y  # Y coordinate  
    coord_2 = (coord_x * coord_x + coord_y * coord_y)  # Squared distance from origin
    
    # Store coordinates
    out_idx = batch_id * grid_size * grid_size * 3 + h * grid_size * 3 + w * 3 + coord_dim
    
    mask = (h < grid_size) & (w < grid_size)
    tl.store(out_ptr + out_idx, tl.stack([coord_0, coord_1, coord_2]), mask=mask)

@torch.fx.wrap
def optimized_coordinate_grid(input_shape):
    batch, height, width = input_shape[:2]  # We need the spatial dimensions
    grid_size = height  # 196
    
    n_coords = 14  # from arange(14) in original
    
    # Calculate grid dimensions
    grid_h = tl.cdiv(grid_size, n_coords)
    grid_w = tl.cdiv(grid_size, n_coords)
    
    out = torch.zeros(1, grid_size, grid_size, 3, dtype=torch.float32)
    
    coordinate_grid_kernel[(1, grid_h, grid_w)](
        out_ptr=out,
        grid_size=grid_size,
        n_coords=n_coords,
        BLOCK_SIZE=1  # We handle different dimensions via program_id
    )
    
    return out

def replacement_func():
    return optimized_coordinate_grid