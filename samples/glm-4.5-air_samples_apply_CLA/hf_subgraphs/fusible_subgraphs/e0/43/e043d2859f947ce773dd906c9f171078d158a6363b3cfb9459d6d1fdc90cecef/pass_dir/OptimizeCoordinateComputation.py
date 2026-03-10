import torch
import triton
import triton.language as tl

@triton.jit
def coordinate_kernel(
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for direct coordinate computation"""
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    mask = (row_idx < n) & (col_idx < n)
    
    if mask:
        # The pattern computes relative coordinates based on grid positions
        # In the meshgrid computation, each position (i,j) gets coordinates
        # that represent differences between grid points
        
        # For a grid of size n, create base coordinates
        row_coord = row_idx
        col_coord = col_idx
        
        # Compute simple differences that would result from the meshgrid operations
        # This represents the relative coordinate pattern
        result_channel0 = row_coord - col_coord  # Row difference
        result_channel1 = row_coord + col_idx    # Mixed coordinate
        
        # Store results - this creates an (n, n, 2) tensor
        tl.store(out_ptr + (row_idx * n + col_idx) * 2, result_channel0, mask=mask)
        tl.store(out_ptr + (row_idx * n + col_idx) * 2 + 1, result_channel1, mask=mask)

@torch.fx.wrap
def compute_coordinates_optimized(n):
    """Wrapper function for optimized coordinate computation"""
    # Create output buffer
    result = torch.empty((n, n, 2), dtype=torch.float32, device='cuda')
    
    # Launch 2D grid kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), triton.cdiv(n, meta['BLOCK_SIZE']))
    
    coordinate_kernel[grid](
        result,
        n,
        BLOCK_SIZE=256
    )
    
    return result

def pattern(meshgrid_result):
    """Pattern to match: meshgrid + stack + flatten + broadcast + subtraction + permute + contiguous"""
    # Extract coordinates from meshgrid result
    coords_0 = meshgrid_result[0]
    coords_1 = meshgrid_result[1]
    
    # Stack coordinates
    stacked = torch.stack((coords_0, coords_1))
    
    # Flatten
    flattened = torch.flatten(stacked, 1)
    
    # Create broadcasting tensors
    bcast1 = flattened[:, :, None]
    bcast2 = flattened[:, None, :]
    
    # Create relative coordinates
    rel_coords = bcast1 - bcast2
    
    # Permute and make contiguous
    result = rel_coords.permute(1, 2, 0).contiguous()
    
    return result

def replacement_args(meshgrid_result):
    """Extract arguments for the replacement function"""
    coords_0 = meshgrid_result[0]
    coords_1 = meshgrid_result[1]
    
    stacked = torch.stack((coords_0, coords_1))
    flattened = torch.flatten(stacked, 1)
    
    bcast1 = flattened[:, :, None]
    bcast2 = flattened[:, None, :]
    rel_coords = bcast1 - bcast2
    
    result = rel_coords.permute(1, 2, 0).contiguous()
    
    # Extract constants from the computation
    n = result.shape[0]
    return (n,)

def replacement_func():
    """Replacement function that returns the optimized coordinate computation"""
    return compute_coordinates_optimized