import torch
import triton
import triton.language as tl

@triton.jit
def coordinate_kernel(
    out_ptr,
    size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate coordinate grid with offset operations in a single kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (size * size)
    
    # Compute 2D coordinates from linear offset
    y = offsets // size
    x = offsets % size
    
    # Apply coordinate transformations similar to the original pattern
    # tmp_8 - tmp_9 operation (coordinate difference)
    coord_diff = x - y
    
    # Apply transformations: adding offset and scaling
    # tmp_13 += 31, tmp_16 += 31, tmp_19 *= 63 pattern
    processed_coord_0 = coord_diff + 31  # Addition operation
    processed_coord_1 = coord_diff + 31  # Addition operation
    scaled_coord_0 = processed_coord_0 * 63  # Scaling operation
    
    # Store coordinates in interleaved format
    coords_out = tl.make_tuple(
        processed_coord_0,
        processed_coord_1,
        scaled_coord_0
    )
    
    # Store the coordinates
    if mask:
        for i, coord in enumerate(coords_out):
            tl.store(out_ptr + (offsets * 3 + i), coord, mask)

@torch.fx.wrap
def optimized_coordinate_generation(size):
    """Optimized coordinate generation function"""
    total_elements = size * size * 3  # 3 output channels
    
    # Determine optimal block size and grid size
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(total_elements, dtype=torch.int64, device='cuda')
    
    coordinate_kernel[grid_size](out_ptr=out, size=size, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape to match the expected output format
    return out.view(size, size, 3)

def pattern(size1, size2):
    """
    Simplified pattern matching that avoids problematic slice operations.
    This pattern captures the key computational structure without complex indexing.
    """
    # Create meshgrid without complex slicing
    x = torch.arange(size1, dtype=torch.int64)
    y = torch.arange(size2, dtype=torch.int64)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Stack and flatten
    coords = torch.stack([X, Y])
    coords_flat = coords.flatten(1)  # Keep dimensions as (2, size*size)
    
    # Compute coordinate differences
    coord_diff = coords_flat[0] - coords_flat[1]  # X - Y
    
    # Reshape to match expected dimensions (size*size, 2)
    diffs = torch.stack([coord_diff, coord_diff], dim=1)
    result = diffs.reshape(size1, size2, 2)
    
    # Apply coordinate transformations using tensor operations
    result = result + 31  # Add 31 to both coordinates
    result[..., 0] = result[..., 0] * 63  # Scale first coordinate by 63
    
    # Add third channel with specific operations
    channel_3 = result[..., 0].clone()  # Copy scaled channel
    result = torch.cat([result, channel_3.unsqueeze(-1)], dim=-1)  # Add as third channel
    
    return result

@torch.fx.wrap
def optimized_coordinate_generation(size1, size2):
    """Optimized coordinate generation function"""
    coord_size = size1  # Use size1 for grid dimensions
    
    total_elements = coord_size * coord_size * 3  # 3 output channels
    
    # Determine optimal block size and grid size
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(total_elements, dtype=torch.int64, device='cuda')
    
    coordinate_kernel[grid_size](out_ptr=out, size=coord_size, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape to match the expected output format
    return out.view(coord_size, coord_size, 3)

def replacement_args(tmp_1, tmp_2):
    size1 = tmp_1.item() if isinstance(tmp_1, torch.Tensor) and tmp_1.numel() == 1 else tmp_1
    size2 = tmp_2.item() if isinstance(tmp_2, torch.Tensor) and tmp_2.numel() == 1 else tmp_2
    return (size1, size2)

def replacement_func():
    return optimized_coordinate_generation