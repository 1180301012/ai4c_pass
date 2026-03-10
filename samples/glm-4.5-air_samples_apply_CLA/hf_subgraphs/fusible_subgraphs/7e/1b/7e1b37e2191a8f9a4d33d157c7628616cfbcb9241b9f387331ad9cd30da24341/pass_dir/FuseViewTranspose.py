import torch
import triton
import triton.language as tl

def pattern(linear_out, shape_args):
    """
    Pattern to match: view followed by transpose operations
    Matches: tmp_2 = linear_out.view(shape_args); tmp_3 = tmp_2.transpose(1, 2)
    Returns: the final transposed tensor
    """
    tmp_2 = linear_out.view(shape_args)
    tmp_3 = tmp_2.transpose(1, 2)
    return tmp_3

def replacement_args(linear_out, shape_args):
    """
    Extract arguments needed for the optimized kernel
    Returns: linear_out tensor and the view shape arguments
    """
    return (linear_out, shape_args)

@triton.jit
def view_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim1_size,
    dim2_size,
    dim3_size,
    dim4_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that fuses view and transpose operations.
    Uses simple 1D processing for each element to avoid complex coordinate bugs.
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    start_offset = pid * BLOCK_SIZE
    offsets = start_offset + tl.arange(0, BLOCK_SIZE)
    
    # Calculate element count for bounds checking
    total_output_elements = batch_size * dim2_size * dim1_size * dim4_size
    mask = offsets < total_output_elements
    
    # Convert linear offset to output coordinates
    # Output shape: [batch_size, dim2_size, dim1_size, dim4_size]
    coords = offsets
    
    batch_coords = coords // (dim2_size * dim1_size * dim4_size)
    remainder = coords % (dim2_size * dim1_size * dim4_size)
    
    dim2_coords = remainder // (dim1_size * dim4_size)
    remainder = remainder % (dim1_size * dim4_size)
    
    dim1_coords = remainder // dim4_size
    dim4_coords = remainder % dim4_size
    
    # Map output coordinates to input coordinates
    # Input shape after view: [batch_size, dim1_size, dim2_size, dim4_size]
    # The transpose(1, 2) swaps dim1 and dim2
    input_batch = batch_coords
    input_dim1 = dim1_coords  # This becomes dim2 in output
    input_dim2 = dim2_coords  # This becomes dim1 in output  
    input_dim4 = dim4_coords
    
    # Flatten input coordinates
    input_offset = (input_batch * (dim1_size * dim2_size * dim4_size) +
                   input_dim1 * (dim2_size * dim4_size) +
                   input_dim2 * dim4_size +
                   input_dim4)
    
    # Calculate output offset (it's the same as our starting offset)
    output_offset = offsets
    
    # Load input data
    input_data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store output data
    tl.store(output_ptr + output_offset, input_data, mask=mask)

@torch.fx.wrap
def view_transpose_fused(input_tensor, view_shape):
    """
    Kernel wrapper for the fused view+transpose operation
    """
    # Extract shape parameters and calculate the inferred (-1) dimension
    batch_size = view_shape[0]
    dim1_size = view_shape[1]
    dim2_size_placeholder = view_shape[2]  # This might be -1, the inferred dimension
    dim4_size = view_shape[3]
    
    # Calculate the inferred dimension if it's -1
    if dim2_size_placeholder == -1:
        # Calculate what -1 should be based on total elements
        total_elements = input_tensor.numel()
        dim2_size = total_elements // (batch_size * dim1_size * dim4_size)
    else:
        dim2_size = dim2_size_placeholder
    
    # For the transpose pattern (1, 2), the output shape is [batch_size, dim2_size, dim1_size, dim4_size]
    # In the kernel, dim3_size refers to the third dimension of the output, which is dim1_size
    dim3_size = dim1_size  # This is the third dimension in the output after transpose
    
    # Output shape after transpose: [batch_size, dim2_size, dim1_size, dim4_size]
    output_shape = [batch_size, dim2_size, dim1_size, dim4_size]
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions for 1D kernel
    BLOCK_SIZE = 1024
    total_elements = batch_size * dim2_size * dim1_size * dim4_size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_size,)
    
    # Launch kernel
    view_transpose_kernel[grid](
        input_tensor,
        output_tensor,
        batch_size,
        dim1_size,
        dim2_size,
        dim3_size,
        dim4_size,
        BLOCK_SIZE,
    )
    
    return output_tensor

def replacement_func():
    """
    Return the optimized kernel function
    """
    return view_transpose_fused