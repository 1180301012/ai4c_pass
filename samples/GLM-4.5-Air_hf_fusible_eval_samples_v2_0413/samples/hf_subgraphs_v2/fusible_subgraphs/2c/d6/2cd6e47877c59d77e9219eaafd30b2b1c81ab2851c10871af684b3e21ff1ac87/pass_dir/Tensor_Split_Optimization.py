import torch
import triton
import triton.language as tl
import math

def pattern(input_tensor):
    """
    Pattern for tensor_split(2, -1) operation followed by tensor access
    tensor_split = in_0.tensor_split(2, -1)
    tmp_14 = tensor_split[0]
    tmp_15 = tensor_split[1]
    """
    tensor_split = input_tensor.tensor_split(2, -1)
    tmp_14 = tensor_split[0]
    tmp_15 = tensor_split[1]
    
    # Return both results as they are used in the computation
    return tmp_14, tmp_15

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def tensor_split_kernel(
    input_ptr, 
    output1_ptr, 
    output2_ptr,
    input_size_first_dim,
    input_size_last_dim,
    split_size_last_dim,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a 2D block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets for 2D tensor
    m_offset = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for bounds checking
    m_mask = m_offset < input_size_first_dim
    n_mask = n_offset < input_size_last_dim
    
    # Load input data
    input_offsets = m_offset[:, None] * input_size_last_dim + n_offset[None, :]
    input_data = tl.load(input_ptr + input_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Split along the last dimension: first half goes to output1, second half to output2
    # Check which part of the split we're processing
    split_mask = n_offset < split_size_last_dim
    
    # For output1 (first half of last dimension)
    output1_offsets = m_offset[:, None] * split_size_last_dim + n_offset[None, :]
    output1_mask = m_mask[:, None] & split_mask[None, :]
    output1_data = tl.where(split_mask[None, :], input_data, 0.0)
    tl.store(output1_ptr + output1_offsets, output1_data, mask=output1_mask)
    
    # For output2 (second half of last dimension)
    output2_offsets = m_offset[:, None] * split_size_last_dim + (n_offset[None, :] - split_size_last_dim)
    output2_mask = m_mask[:, None] & (~split_mask[None, :]) & (n_offset[None, :] < input_size_last_dim)
    output2_data = tl.where(~split_mask[None, :], input_data, 0.0)
    tl.store(output2_ptr + output2_offsets, output2_data, mask=output2_mask)

@torch.fx.wrap
def tensor_split_optimized(input_tensor):
    # Get tensor dimensions (assuming 2D tensor based on weight_meta.py)
    first_dim = input_tensor.shape[0]
    last_dim = input_tensor.shape[-1]
    
    # Determine split size (should be half of last dimension for split=2)
    split_size = last_dim // 2
    
    # Set block size
    BLOCK_SIZE = 64
    
    # Calculate grid dimensions
    grid_m = (first_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_n = (last_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors with half the last dimension
    output1_shape = list(input_tensor.shape)
    output1_shape[-1] = split_size
    output1 = torch.empty(output1_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    output2_shape = list(input_tensor.shape)
    output2_shape[-1] = split_size
    output2 = torch.empty(output2_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch optimized kernel
    tensor_split_kernel[(grid_m, grid_n)](
        input_tensor,
        output1,
        output2,
        first_dim,
        last_dim,
        split_size,
        BLOCK_SIZE
    )
    
    return output1, output2

def replacement_func():
    return tensor_split_optimized