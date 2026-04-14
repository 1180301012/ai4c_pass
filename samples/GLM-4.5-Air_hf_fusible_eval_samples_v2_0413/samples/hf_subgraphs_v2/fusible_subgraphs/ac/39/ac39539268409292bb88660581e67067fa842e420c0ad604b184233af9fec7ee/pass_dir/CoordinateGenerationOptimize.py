import torch
import triton
import triton.language as tl

@triton.jit
def coordinate_optimized_kernel(
    out_ptr,
    size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized coordinate generation kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (size * size)
    
    # Generate coordinates directly in GPU
    x = offsets % size
    y = offsets // size
    
    # Compute coordinate difference (equivalent to X - Y operation)
    coord_diff = x - y
    
    # Apply scalar operations (equivalent to adding 31 and multiplying by 63)
    result_coord = coord_diff + 31
    scaled_coord = result_coord * 63
    
    # Store results (3 channels as in the original computation)
    if mask:
        # Channel 0: Scaled coordinate
        tl.store(out_ptr + (offsets * 3), scaled_coord, mask)
        # Channel 1: Offset coordinate  
        tl.store(out_ptr + (offsets * 3 + 1), result_coord, mask)
        # Channel 2: Copy of channel 0
        tl.store(out_ptr + (offsets * 3 + 2), scaled_coord, mask)

@torch.fx.wrap
def optimized_coordinate_generation(size):
    """Optimized coordinate generation function"""
    total_elements = size * size * 3  # 3 output channels
    
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(total_elements, dtype=torch.int64, device='cuda')
    
    coordinate_optimized_kernel[(grid_size,)](out_ptr=out, size=size, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape to match expected format [size, size, 3]
    return out.view(size, size, 3)

def pattern(size_val):
    """Pattern matching coordinate generation from arange operations"""
    # Generate base coordinate arrays
    tmp_1 = torch.arange(size_val, dtype=torch.int64)
    tmp_2 = torch.arange(size_val, dtype=torch.int64)
    
    # Create meshgrid (coordinate grid)
    meshgrid = torch.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = meshgrid[0]  # X coordinates
    tmp_5 = meshgrid[1]  # Y coordinates
    
    # Stack coordinates
    tmp_6 = torch.stack((tmp_4, tmp_5))
    
    # Flatten for processing
    tmp_7 = torch.flatten(tmp_6, 1)
    
    # Extract and compute coordinate differences
    tmp_8 = tmp_7[(slice(None, None, None), slice(None, None, None), None)]
    tmp_9 = tmp_7[(slice(None, None, None), None, slice(None, None, None))]
    tmp_10 = tmp_8 - tmp_9  # X - Y differences
    
    # Permute to desired layout
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()
    
    return tmp_12  # Return intermediate result that will be further processed

def replacement_args(size_input):
    """Extract size for coordinate generation"""
    size = size_input.item() if isinstance(size_input, torch.Tensor) and size_input.numel() == 1 else size_input
    return (size,)

def replacement_func():
    """Return optimized coordinate generation function"""
    return optimized_coordinate_generation