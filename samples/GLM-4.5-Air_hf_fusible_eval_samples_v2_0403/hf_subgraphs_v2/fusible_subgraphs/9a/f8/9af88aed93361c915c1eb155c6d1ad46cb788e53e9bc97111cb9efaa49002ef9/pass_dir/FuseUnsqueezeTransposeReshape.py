import torch
import triton
import triton.language as tl

def pattern(x):
    """Match the pattern: unsqueeze(1) followed by transpose(2, 3)"""
    tmp_1 = x.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return tmp_2

def replacement_args(x):
    """Extract arguments for the replacement function"""
    return (x,)

@triton.jit
def optimized_reshape_kernel(
    input_ptr,
    output_ptr,
    input_size_0: tl.constexpr,
    input_size_1: tl.constexpr, 
    input_size_2: tl.constexpr,
    output_size_0: tl.constexpr,
    output_size_1: tl.constexpr,
    output_size_2: tl.constexpr,
    output_size_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Efficiently reshape tensor from [1, 1024, 128] to [1, 1, 128, 1024]
    This kernel performs the equivalent of unsqueeze(1) followed by transpose(2, 3)
    """
    # Calculate grid coordinates
    batch_idx = tl.program_id(0)  # batch dimension (always 0 for our case)
    output_dim2_idx = tl.program_id(1)  # the 128 dimension (0 to 127)
    
    # Each thread computes one element in the output's last dimension (1024 elements)
    output_dim3_idx = tl.arange(0, BLOCK_SIZE)
    mask_dim3 = output_dim3_idx < output_size_3
    
    if output_dim2_idx < output_size_2:  # if within 0 to 127
        # Map output indices to input indices:
        # input: [1, 1024, 128] → [batch, dim1, dim2] 
        # output: [1, 1, 128, 1024] → [batch, fixed_1, dim2, dim1]
        # So: [0, 0, output_dim2_idx, output_dim3_idx] maps to [0, output_dim3_idx, output_dim2_idx]
        
        # Calculate input indices
        input_dim1_idx = output_dim3_idx  # the 1024 dimension becomes dim1 in input
        input_dim2_idx = output_dim2_idx    # the 128 dimension becomes dim2 in input
        
        # Calculate input linear offset: [0, input_dim1_idx, input_dim2_idx]
        input_offset = (input_dim2_idx + 
                       input_dim1_idx * input_size_2 + 
                       batch_idx * input_size_1 * input_size_2)
        
        # Load data from input
        values = tl.load(input_ptr + input_offset, mask=mask_dim3, other=0.0)
        
        # Calculate output linear offset: [0, 0, output_dim2_idx, output_dim3_idx]
        output_offset = (output_dim3_idx + 
                       output_dim2_idx * output_size_3 + 
                       0 * output_size_2 * output_size_3 + 
                       batch_idx * output_size_1 * output_size_2 * output_size_3)
        
        # Store to output
        tl.store(output_ptr + output_offset, values, mask=mask_dim3)

@torch.fx.wrap
def optimized_reshape_3d_to_4d(x):
    """
    Efficient reshape that performs the equivalent of unsqueeze(1) + transpose(2, 3)
    Input shape: [batch_size, dim1, dim2] 
    Output shape: [batch_size, 1, dim2, dim1]
    
    Final optimized approach: transpose first, then unsqueeze
    - This minimizes the number of memory stride operations
    - transpose(1,2) on 3D tensor is more efficient than transpose(2,3) on 4D tensor
    """
    input_shape = x.shape
    batch_size, dim1, dim2 = input_shape
    
    # Optimized order: transpose on 3D first, then unsqueeze
    # [1, 1024, 128] → transpose(1,2) → [1, 128, 1024] → unsqueeze(1) → [1, 1, 128, 1024]
    return x.transpose(1, 2).unsqueeze(1)

def replacement_func():
    """Return the optimized function reference"""
    return optimized_reshape_3d_to_4d