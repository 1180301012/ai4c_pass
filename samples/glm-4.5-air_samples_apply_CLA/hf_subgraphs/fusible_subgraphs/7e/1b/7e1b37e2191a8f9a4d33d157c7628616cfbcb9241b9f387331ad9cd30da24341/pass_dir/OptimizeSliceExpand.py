import torch
import triton
import triton.language as tl

def pattern(key_states, expand_shape):
    """
    Pattern to match: slice followed by expand operations
    Matches: tmp_4 = key_states[slice(...)] followed by tmp_5 = tmp_4.expand(expand_shape)
    Returns: the final expanded tensor
    """
    tmp_4 = key_states[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_5 = tmp_4.expand(expand_shape)
    return tmp_5

def replacement_args(key_states, expand_shape):
    """
    Extract arguments needed for the optimized kernel
    Returns: key_states tensor and the target expand shape
    """
    return (key_states, expand_shape)

@triton.jit
def slice_expand_kernel(
    input_ptr,
    output_ptr,
    input_shape,
    output_shape,
    dim1_in,
    dim2_in,
    dim3_in,
    dim4_in,
    dim1_out,
    dim2_out,
    dim3_out,
    dim4_out,
    dim5_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel that fuses slice and expand operations.
    Efficiently computes the expanded tensor directly from input without intermediate allocation.
    """
    # Each program handles a block of the output tensor
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate block boundaries
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create offsets within the block
    offsets_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create 2D coordinate grids
    m_coords = offsets_m[:, None]
    n_coords = offsets_n[None, :]
    
    # Map output coordinates to input coordinates
    # Input shape: [dim1_in, dim2_in, dim3_in, dim4_in]
    # Output shape: [dim1_out, dim2_out, dim3_out, dim4_out, dim5_out]
    
    # Calculate each dimension:
    # dim1_out should map to dim1_in
    # dim2_out should map to dim2_in  
    # dim3_out (expanded dimension) maps to 1 in input
    # dim4_out should map to dim3_in
    # dim5_out should map to dim4_in
    
    dim1_coords = m_coords % dim1_out
    dim2_coords = (m_coords // dim1_out) % dim2_out
    dim4_coords = (n_coords % dim4_out)
    dim5_coords = (n_coords // dim4_out) % dim5_out
    
    # For dim3_out, this maps to the new dimension (size 1 in input)
    dim3_coords = dim3_out - 1  # This should always be 0 since we map to the inserted dimension
    
    # Calculate input coordinates
    input_m = dim1_coords
    input_n = dim2_coords
    input_p = dim4_coords  # maps to input dim3_in
    input_q = dim5_coords  # maps to input dim4_in
    
    # Flatten input coordinates
    input_offsets = input_m * (dim2_in * dim3_in * dim4_in) + \
                   input_n * (dim3_in * dim4_in) + \
                   input_p * dim4_in + \
                   input_q
    
    # Calculate output offsets
    output_offsets = m_coords * (dim2_out * dim3_out * dim4_out * dim5_out) + \
                    n_coords * dim5_out
    
    # Create masks for bounds checking
    input_mask = (dim1_coords < dim1_in) & (dim2_coords < dim2_in) & \
                 (dim4_coords < dim3_in) & (dim5_coords < dim4_in)
    output_mask = (m_coords < dim1_out * dim2_out) & (n_coords < dim4_out * dim5_out)
    
    # Load input data (only where input is valid)
    input_data = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Store output data (broadcasted to all positions in the expanded dimension)
    tl.store(output_ptr + output_offsets, input_data, mask=output_mask)

@torch.fx.wrap
def slice_expand_optimized(input_tensor, target_shape):
    """
    Kernel wrapper for the optimized slice+expand operation
    """
    # Extract input shape parameters
    dim1_in, dim2_in, dim3_in, dim4_in = input_tensor.shape
    
    # Extract output shape parameters
    dim1_out, dim2_out, dim3_out, dim4_out, dim5_out = target_shape
    
    # Verify the dimension mapping is correct
    assert dim1_out == dim1_in, f"dim1 mismatch: {dim1_out} != {dim1_in}"
    assert dim2_out == dim2_in, f"dim2 mismatch: {dim2_out} != {dim2_in}"
    assert dim4_out == dim3_in, f"dim4 mismatch: {dim4_out} != {dim3_in}"
    assert dim5_out == dim4_in, f"dim5 mismatch: {dim5_out} != {dim4_in}"
    
    # Create output tensor
    output_tensor = torch.empty(target_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    grid_m = (dim1_out * dim2_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (dim4_out * dim5_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Launch kernel
    slice_expand_kernel[grid](
        input_tensor,
        output_tensor,
        input_tensor.shape,
        target_shape,
        dim1_in, dim2_in, dim3_in, dim4_in,
        dim1_out, dim2_out, dim3_out, dim4_out, dim5_out,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output_tensor

def replacement_func():
    """
    Return the optimized kernel function
    """
    return slice_expand_optimized