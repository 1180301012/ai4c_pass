import torch
import triton
import triton.language as tl

# Pattern matching for reshape -> transpose -> reshape chain
def pattern(tmp_0):
    # Match the tensor manipulation chain
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    # Return the intermediate and final results that need to be preserved
    return tmp_7, tmp_9

# Argument extraction function
def replacement_args(tmp_0):
    return (tmp_0,)

# Optimized kernel for direct reshaping with transpose semantics
@triton.jit
def reshape_with_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim1_in, dim2_in, dim3_in, dim4_in,
    dim1_out, dim2_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output position
    output_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < batch_size * dim1_out * dim2_out
    output_idx_flat = output_idx
    
    # Convert linear index to output dimensions
    b = output_idx_flat // (dim1_out * dim2_out)
    remainder = output_idx_flat % (dim1_out * dim2_out)
    i = remainder // dim2_out
    j = remainder % dim2_out
    
    # Map back to input dimensions with transpose semantics
    # Output: (1, 361, 49) corresponds to input: (1, 19, 19, 7, 7) after transpose
    original_i = i // 19  # First dimension after transpose
    original_j = i % 19   # Second dimension after transpose  
    original_k = j // 7   # Third dimension after transpose
    original_l = j % 7    # Fourth dimension after transpose
    
    # Calculate flattened input index with transpose(2,3) semantics
    # Input was (1, 19, 7, 19, 7), after transpose(2,3) becomes (1, 19, 19, 7, 7)
    input_idx_flat = b * (dim1_in * dim2_in * dim3_in * dim4_in) + \
                    original_i * (dim2_in * dim3_in * dim4_in) + \
                    original_j * (dim3_in * dim4_in) + \
                    original_k * dim4_in + original_l
    
    # Load from input and write to output
    val = tl.load(input_ptr + input_idx_flat, mask=mask, other=0.0)
    tl.store(output_ptr + output_idx, val, mask=mask)

@torch.fx.wrap
def optimized_reshape_transpose_reshape(input_tensor):
    # Extract shapes
    input_shape = input_tensor.shape
    
    # Input shape after first reshape: (1, 19, 7, 19, 7)
    # After transpose(2,3): (1, 19, 19, 7, 7)  
    # After final reshape: (1, 361, 49)
    
    batch_size_in = 1
    dim1_in, dim2_in = 19, 19  # After transpose
    dim3_in, dim4_in = 7, 7     # After transpose
    
    batch_size_out = 1
    dim1_out, dim2_out = 361, 49
    
    # Create output tensor
    output_shape = (batch_size_out, dim1_out, dim2_out)
    output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    
    BLOCK_SIZE = 1024
    num_elements = batch_size_out * dim1_out * dim2_out
    grid = ( (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    reshape_with_transpose_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size_in,
        dim1_in=dim1_in, dim2_in=dim2_in, dim3_in=dim3_in, dim4_in=dim4_in,
        dim1_out=dim1_out, dim2_out=dim2_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Also return the intermediate reshape result for compatibility
    intermediate_reshape = input_tensor.reshape(1, 19, 7, 19, 7)
    
    return intermediate_reshape, output

# Replacement function
def replacement_func():
    return optimized_reshape_transpose_reshape