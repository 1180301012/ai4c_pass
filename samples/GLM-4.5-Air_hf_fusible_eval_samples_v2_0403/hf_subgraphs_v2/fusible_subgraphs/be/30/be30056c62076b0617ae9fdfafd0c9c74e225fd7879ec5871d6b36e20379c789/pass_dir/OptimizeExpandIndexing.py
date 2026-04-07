import torch
import triton
import triton.language as tl

@torch.fx.wrap
def optimized_expand_indexing(input_tensor):
    """Main wrapper for optimized tensor expansion"""
    # Get input shape
    input_shape = input_tensor.shape  # e.g., [2, 128]
    
    # Calculate output shape based on the pattern [None, None, slice(...)]
    # This creates [1, 1, original_0, original_1]
    output_shape = (1, 1) + input_shape  # e.g., [1, 1, 2, 128]
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions
    total_elements = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    BLOCK_SIZE = 1024  # Optimal block size for GPU
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    expand_indexing_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_shape=input_shape,
        output_shape=output_shape,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@triton.jit
def expand_indexing_kernel(
    input_ptr,
    output_ptr,
    input_shape,
    output_shape,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for tensor expansion using None indexing"""
    # Get input dimensions
    input_dim0, input_dim1 = input_shape  # e.g., [2, 128]
    output_dim0, output_dim1, output_dim2, output_dim3 = output_shape  # e.g., [1, 1, 2, 128]
    
    # Calculate total elements in each dimension
    total_output_elements = output_dim0 * output_dim1 * output_dim2 * output_dim3
    
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_output_elements
    
    # Convert linear offset to multi-dimensional coordinates
    def linear_to_multi(idx, shape):
        coords = []
        for dim in reversed(shape):
            coords.append(idx % dim)
            idx = idx // dim
        return list(reversed(coords))
    
    # Process each element in the block
    for i, offset in enumerate(offsets):
        if offset < total_output_elements:
            # Convert output coordinate to input coordinate
            out_coords = linear_to_multi(offset, output_shape)
            
            # Map expansion: [out_dim0, out_dim1, out_dim2, out_dim3] -> [input_dim0, input_dim1]
            # The pattern [None, None, slice(...)] means:
            # - output[0, :, :, :] maps to input[:, :] (first None at dim 0 creates new size 1 dim)
            # - output[1, :, :, :] maps to input[:, :] (second None at dim 1 creates new size 1 dim)
            # - output[:, :, i, :] maps to input[i, :] (slice at dim 2 maps to input dim 0)
            # - output[:, :, :, j] maps to input[:, j] (slice at dim 3 maps to input dim 1)
            
            input_dim0_idx = out_coords[2]  # Third dim in output maps to first dim in input
            input_dim1_idx = out_coords[3]  # Fourth dim in output maps to second dim in input
            
            # Calculate input index
            input_idx = input_dim0_idx * input_dim1 + input_dim1_idx
            
            # Calculate output index
            output_idx = offset
            
            # Load from input and store to output
            if input_dim0_idx < input_dim0 and input_dim1_idx < input_dim1:
                input_val = tl.load(input_ptr + input_idx, other=0.0)
                tl.store(output_ptr + output_idx, input_val, other=0.0)

def pattern(input_tensor):
    """Pattern: Advanced indexing to expand tensor dimensions"""
    # This operation expands [2, 128] to [1, 1, 2, 128] using None indexing and slice
    result = input_tensor[(None, None, slice(None, None, None))]
    return result

def replacement_args(input_tensor):
    """Extract arguments needed for replacement - just the input tensor"""
    return (input_tensor,)

def replacement_func():
    """Return the optimized kernel implementation for tensor expansion"""
    return optimized_expand_indexing