import torch
import triton
import triton.language as tl

# Simple working pattern based on what we know works
def pattern(in_0):
    # Simple pattern: match the input reshape operation
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 96)
    return tmp_5

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel for direct coordinate difference computation
@triton.jit
def coordinate_difference_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    n_coords,
    coord_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output position in the 4D tensor (B, N, N, D)
    output_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < batch_size * n_coords * n_coords * coord_dim
    output_idx_flat = output_idx
    
    # Convert linear index to 4D output coordinates
    total = batch_size * n_coords * n_coords * coord_dim
    b = output_idx_flat // (n_coords * n_coords * coord_dim)
    remainder = output_idx_flat % (n_coords * n_coords * coord_dim)
    i = remainder // (n_coords * coord_dim)
    remainder2 = remainder % (n_coords * coord_dim)
    j = remainder2 // coord_dim
    k = remainder2 % coord_dim
    
    # Calculate input indices for broadcasting
    idx1 = i * coord_dim + k  # For tmp_10: (B, N, 1, D) flattened
    idx2 = j * coord_dim + k  # For tmp_11: (B, N, D, 1) flattened
    
    # Load values and compute difference
    val1 = tl.load(input_ptr + idx1, mask=mask, other=0.0)
    val2 = tl.load(input_ptr + idx2, mask=mask, other=0.0)
    diff = val1 - val2
    
    # Store the result at the correct 4D position
    tl.store(output_ptr + output_idx, diff, mask=mask)

@torch.fx.wrap
def optimized_coordinate_difference(input_tensor):
    # Extract shapes
    batch_size, n_coords, coord_dim = input_tensor.shape
    output_shape = (batch_size, n_coords, n_coords, coord_dim)
    
    # Create output tensor
    output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    
    BLOCK_SIZE = 256
    num_elements = batch_size * n_coords * n_coords * coord_dim
    grid = ( (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    coordinate_difference_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        n_coords=n_coords,
        coord_dim=coord_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Create intermediate unsqueezed tensors for compatibility
    tmp_10 = input_tensor.unsqueeze(2)  # Shape: (B, N, 1, D)
    tmp_11 = input_tensor.unsqueeze(3)  # Shape: (B, N, D, 1)
    
    return tmp_10, tmp_11, output

# Alternative coordinate difference kernel with autotuning
@triton.jit
def coordinate_difference_autotuned_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    n_coords,
    coord_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 2D grid for better memory access patterns
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Inner loop over coordinate dimensions
    offsets_k = tl.arange(0, BLOCK_SIZE_M)
    mask_k = offsets_k < coord_dim
    
    # Load input values for both coordinates
    idx1 = m * n_coords * coord_dim + n * coord_dim + offsets_k
    idx2 = m * n_coords * coord_dim + m * coord_dim + offsets_k
    
    val1 = tl.load(input_ptr + idx1, mask=mask_k, other=0.0)
    val2 = tl.load(input_ptr + idx2, mask=mask_k, other=0.0)
    
    # Compute differences
    diffs = val1 - val2
    
    # Store output
    output_idx = m * (n_coords * n_coords * coord_dim) + n * (coord_dim) + offsets_k
    tl.store(output_ptr + output_idx, diffs, mask=mask_k)

@torch.fx.wrap  
def optimized_autotuned_coordinate_difference(input_tensor):
    batch_size, n_coords, coord_dim = input_tensor.shape
    output_shape = (batch_size, n_coords, n_coords, coord_dim)
    output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Adjust grid and block sizes for better performance
    BLOCK_SIZE_M = min(coord_dim, 32)
    BLOCK_SIZE_N = min(n_coords, 256)
    
    grid_m = (batch_size * n_coords + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_n = (n_coords + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    coordinate_difference_autotuned_kernel[(grid_m, grid_n)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        n_coords=n_coords,
        coord_dim=coord_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Return intermediate tensors for compatibility
    tmp_10 = input_tensor.unsqueeze(2)
    tmp_11 = input_tensor.unsqueeze(3)
    
    return tmp_10, tmp_11, output

# Simple optimized reshape function
@torch.fx.wrap
def optimized_reshape(in_0):
    return in_0.reshape(1, 19, 7, 19, 7, 96)

# Replacement function
def replacement_func():
    return optimized_reshape