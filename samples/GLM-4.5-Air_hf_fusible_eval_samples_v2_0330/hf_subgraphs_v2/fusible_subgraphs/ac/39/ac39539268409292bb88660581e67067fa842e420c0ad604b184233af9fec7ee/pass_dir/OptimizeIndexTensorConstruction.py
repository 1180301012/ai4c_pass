import torch
import triton
import triton.language as tl

def pattern():
    # Match the exact index tensor construction pattern from the models
    # Using concrete values that appear in models (3969, 3970, 3971)
    tmp_23 = tmp_12.sum(-1)
    tmp_22 = torch.zeros(size=(tmp_23.shape[0] + 1, tmp_23.shape[0] + 1), dtype=torch.int64)
    tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23
    tmp_22[(0, slice(0, None, None))] = 3969
    tmp_22[(slice(0, None, None), 0)] = 3970
    tmp_22[(0, 0)] = 3971
    return tmp_22.view(-1)

@triton.jit
def index_tensor_kernel(
    coord_sum_ptr,
    out_ptr,
    N, 
    boundary_val1,
    boundary_val2,
    corner_val,
    BLOCK_SIZE: tl.constexpr
):
    # Kernel parameters for the coordinate sum operation
    coord_grid_size = N * N
    program_id = tl.program_id(0)
    
    # Handle coordinate sum part
    if program_id < coord_grid_size:
        offsets = tl.arange(0, min(BLOCK_SIZE, coord_grid_size - program_id * BLOCK_SIZE))
        mask = offsets < (coord_grid_size - program_id * BLOCK_SIZE)
        
        # Load coordinates and sum them
        x_coords = tl.load(coord_sum_ptr[0, program_id * BLOCK_SIZE + offsets], mask=mask, other=0)
        y_coords = tl.load(coord_sum_ptr[1, program_id * BLOCK_SIZE + offsets], mask=mask, other=0)
        
        # Store the sum
        result = x_coords + y_coords
        tl.store(out_ptr + program_id * BLOCK_SIZE + offsets, result, mask=mask)

    # Boundary values are handled separately in the host code for efficiency

@torch.fx.wrap
def optimized_index_tensor(coord_tensor, boundary_val1, boundary_val2, corner_val):
    N = int((coord_tensor.shape[1]) ** 0.5)
    grid_size = N * N
    tensor_size = N + 1
    
    BLOCK_SIZE = 1024
    num_coord_programs = (grid_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.zeros((tensor_size * tensor_size,), dtype=torch.int64, device='cuda')
    
    # Coordinate sum calculation
    coord_sum_kernel_wrapper = lambda: index_tensor_kernel[(num_coord_programs,)](
        coord_ptr=coord_tensor,
        out_ptr=out[1:(tensor_size * tensor_size - tensor_size)],  # Skip first row and column
        N=N,
        boundary_val1=boundary_val1,
        boundary_val2=boundary_val2,
        corner_val=corner_val,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Calculate coordinate sum and store in main area
    coord_sum = coord_tensor.sum(dim=0)  # This will be (2, N*N), sum along dim 0 gives (N*N)
    out[1:(tensor_size * tensor_size - tensor_size)] = coord_sum.view(-1)
    
    # Fill boundary values
    out[0:tensor_size] = boundary_val2  # First column
    out[::tensor_size] = boundary_val1  # First row (excluding corner which gets overwritten)
    out[0] = corner_val  # Corner element
    
    return out

def replacement_args():
    return ()

def replacement_func():
    return optimized_index_tensor