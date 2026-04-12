import torch
import triton
import triton.language as tl

def pattern(start_tensor):
    """
    Pattern that matches the torch.zeros operation and basic structure.
    """
    # Match the beginning of the positional embedding computation
    result = torch.zeros(1, 196, 196, 3)
    return result

def replacement_args(start_tensor):
    """Extract arguments for the replacement function."""
    return (start_tensor,)

@triton.jit
def embedding_kernel(
    out_ptr,
    n_height,
    n_width, 
    n_channels,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """High-performance kernel for computing spatial coordinate embeddings."""
    # Compute grid positions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1) 
    
    # Create coordinate ranges - we need 14x14 base grid that gets expanded
    # The output is 196x196 (14*14 x 14*14)
    height_start = pid_m * BLOCK_SIZE_M
    width_start = pid_n * BLOCK_SIZE_N
    offsets_m = height_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = width_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    mask_m = offsets_m < n_height
    mask_n = offsets_n < n_width
    
    # Compute scaled coordinates in the range [-6.5, 6.5] for 14x14 grid
    # then scale to match the 196x196 output
    coord_range = 6.5  # range from center (0,0) to edge
    scale = 1.0 / (6.5 + 1e-6)  # normalize to [-1, 1] range
    
    # Compute x and y coordinates (relative to grid center)
    x_coords = (tl.cast(offsets_m, tl.float32) - 97.0) * scale  # 97.5 is center of 196
    y_coords = (tl.cast(offsets_n, tl.float32) - 97.0) * scale
    
    # Reshape for broadcasting
    x_coords = x_coords[:, None]  # shape: [BLOCK_SIZE_M, 1]  
    y_coords = y_coords[None, :]  # shape: [1, BLOCK_SIZE_N]
    
    # Compute coordinate features for each channel
    # Channel 0: x-coordinates
    # Channel 1: y-coordinates  
    # Channel 2: squared distance from center
    x_feat = x_coords
    y_feat = y_coords
    dist_feat = x_coords * x_coords + y_coords * y_coords
    
    # Store results in output tensor
    out_ptrs = out_ptr + offsets_m[:, None] * n_width * n_channels + offsets_n[None, :] * n_channels
    
    # Store all three channels
    tl.store(out_ptrs + 0, x_feat, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(out_ptrs + 1, y_feat, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(out_ptrs + 2, dist_feat, mask=mask_m[:, None] & mask_n[None, :])

@torch.fx.wrap
def optimized_embedding_computation(start_tensor):
    """Wrapper function that computes optimized positional embeddings directly."""
    # Create output tensor on the same device as input
    output = torch.zeros_like(start_tensor, dtype=torch.float32)
    
    # Set up grid for kernel launch
    n_height, n_width = 196, 196
    n_channels = 3
    block_size_m = 16  # Tune based on GPU
    block_size_n = 16  # Tune based on GPU
    
    grid_m = (n_height + block_size_m - 1) // block_size_m
    grid_n = (n_width + block_size_n - 1) // block_size_n
    
    # Launch kernel
    embedding_kernel[(grid_m, grid_n)](
        output,
        n_height,
        n_width,
        n_channels,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n
    )
    
    return output

def replacement_func():
    """Return the optimized function (without calling it)."""
    return optimized_embedding_computation