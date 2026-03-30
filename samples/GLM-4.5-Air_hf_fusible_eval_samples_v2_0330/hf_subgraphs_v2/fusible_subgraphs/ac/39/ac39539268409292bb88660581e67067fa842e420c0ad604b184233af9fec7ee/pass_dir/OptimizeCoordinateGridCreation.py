import torch
import triton
import triton.language as tl

def pattern():
    # Match the exact coordinate grid creation pattern from the models
    tmp_1 = torch.arange(32)
    tmp_2 = torch.arange(32)
    meshgrid = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = meshgrid[0]
    tmp_5 = meshgrid[1]
    meshgrid = None
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_4 = tmp_5 = None
    tmp_7 = torch.flatten(tmp_6, 1)
    tmp_6 = None
    return tmp_7

@triton.jit
def coordinate_grid_kernel(N, out_ptr, BLOCK_SIZE: tl.constexpr):
    grid_size = N * N
    program_id = tl.program_id(0)
    
    if program_id * BLOCK_SIZE >= grid_size:
        return
    
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < grid_size
    
    # Convert 1D offset to 2D coordinates
    y_coords = offsets // N
    x_coords = offsets % N
    
    # Store coordinates as [x, y] pairs
    out_ptr[0, offsets] = x_coords
    out_ptr[1, offsets] = y_coords

@torch.fx.wrap
def optimized_coordinate_grid():
    # For this specific pattern with the concrete 32 value
    N = 32
    grid_size = N * N
    BLOCK_SIZE = 1024
    num_programs = (grid_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((2, grid_size), dtype=torch.int64, device='cuda')
    
    coordinate_grid_kernel[(num_programs,)](
        N=N,
        out_ptr=out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_args():
    return ()

def replacement_func():
    return optimized_coordinate_grid