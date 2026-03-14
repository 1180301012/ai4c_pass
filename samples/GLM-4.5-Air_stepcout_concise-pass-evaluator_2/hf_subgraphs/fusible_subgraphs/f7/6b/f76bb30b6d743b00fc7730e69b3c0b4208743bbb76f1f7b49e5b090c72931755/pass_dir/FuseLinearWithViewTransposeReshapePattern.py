import torch
import triton
import triton.language as tl

def pattern(key_states, query_states):
    """
    Pattern matching: View/transpose/reshape operations on key_states and query_states
    This matches the transformation without the linear operation to avoid blocked APIs.
    
    Original pattern:
    - key_states: in_3 -> view(1,-1,16,64) -> transpose(1,2) -> reshape(16,-1,64) -> transpose(1,2)
    - query_states: in_4 -> view(1,-1,16,64) -> transpose(1,2) -> reshape(16,-1,64)
    """
    # Process key_states transformation: 
    # tmp_3 = key_states.view(1, -1, 16, 64)
    # tmp_4 = tmp_3.transpose(1, 2) 
    # tmp_10 = tmp_4.reshape(16, -1, 64)
    # tmp_12 = tmp_10.transpose(1, 2)
    tmp_3 = key_states.view(1, -1, 16, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_10 = tmp_4.reshape(16, -1, 64)
    tmp_12 = tmp_10.transpose(1, 2)
    
    # Process query_states transformation:
    # tmp_7 = query_states.view(1, -1, 16, 64) 
    # tmp_8 = tmp_7.transpose(1, 2)
    # tmp_9 = tmp_8.reshape(16, -1, 64)
    tmp_7 = query_states.view(1, -1, 16, 64)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_9 = tmp_8.reshape(16, -1, 64)
    
    # Return the two transformed tensors
    return (tmp_12, tmp_9)

def replacement_args(key_states, query_states):
    return (key_states, query_states)

@triton.jit
def optimized_query_transform_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, feature_size,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for query_states transformation"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    final_seq_len = batch_size * seq_len
    
    # Map from final [16, batch_size*seq_len, 64] to input [batch_size, seq_len, 1024]
    # For each position in final output, find corresponding input position
    final_row = offsets // final_seq_len  # 0-15 
    final_col = offsets % final_seq_len   # 0 to batch_size*seq_len-1
    
    # Convert flat final_col to batch and sequence indices
    batch_idx = final_col // seq_len
    seq_idx = final_col % seq_len
    
    # Map to input: [batch_idx, seq_idx, final_row*64:(final_row+1)*64]
    input_offset = batch_idx * (seq_len * feature_size) + seq_idx * feature_size + final_row * 64
    data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store directly in final output layout
    output_offset = final_row * (final_seq_len * 64) + final_col * 64
    tl.store(output_ptr + output_offset, data, mask=mask)

@torch.fx.wrap
def optimized_query_states_transform(query_states):
    """Optimized query_states transformation using Triton"""
    batch_size, seq_len, feature_size = query_states.shape
    intermediate_seq_len = batch_size * seq_len
    final_shape = (16, intermediate_seq_len, 64)
    
    output = torch.empty(final_shape, dtype=query_states.dtype, device=query_states.device)
    
    total_elements = 16 * intermediate_seq_len
    BLOCK_SIZE = 1024
    grid_size = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    optimized_query_transform_kernel[grid_size](
        query_states, output,
        batch_size, seq_len, feature_size,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@triton.jit
def optimized_key_transform_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, feature_size,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for key_states transformation (with final transpose)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    final_seq_len = batch_size * seq_len
    
    # For key_states with final transpose: the result is [16, 64, batch_size*seq_len]
    # instead of [16, batch_size*seq_len, 64]
    
    # Map from final [16, 64, batch_size*seq_len] to input [batch_size, seq_len, 1024]
    final_row = offsets // (64 * final_seq_len)  # 0-15 for the first dimension
    final_dim2 = (offsets % (64 * final_seq_len)) // final_seq_len  # 0-63 for the second dimension
    final_col = offsets % final_seq_len   # 0 to batch_size*seq_len-1
    
    # Convert flat final_col to batch and sequence indices
    batch_idx = final_col // seq_len
    seq_idx = final_col % seq_len
    
    # Map to input: [batch_idx, seq_idx, final_row*64:(final_row+1)*64]
    # We need data from position final_dim2 within each 64-element block
    input_offset = batch_idx * (seq_len * feature_size) + seq_idx * feature_size + final_row * 64 + final_dim2
    data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store in transposed layout: [16, 64, batch_size*seq_len]
    output_offset = final_row * (64 * final_seq_len) + final_dim2 * final_seq_len + final_col
    tl.store(output_ptr + output_offset, data, mask=mask)

@torch.fx.wrap
def optimized_key_states_transform(key_states):
    """Optimized key_states transformation using Triton"""
    batch_size, seq_len, feature_size = key_states.shape
    intermediate_seq_len = batch_size * seq_len
    
    # The kernel produces [16, 64, batch_size*seq_len] layout
    output_shape = (16, 64, intermediate_seq_len)
    
    output = torch.empty(output_shape, dtype=key_states.dtype, device=key_states.device)
    
    total_elements = 16 * 64 * intermediate_seq_len
    BLOCK_SIZE = 1024
    grid_size = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    optimized_key_transform_kernel[grid_size](
        key_states, output,
        batch_size, seq_len, feature_size,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return as [16, batch_size*seq_len, 64] to match original pattern
    return output.transpose(1, 2)

def replacement_func():
    """Return the optimized computation function for two tensor transformations"""
    def optimized_transformation(key_states, query_states):
        # Apply optimized transformations to both tensors using Triton kernels
        tmp_12 = optimized_key_states_transform(key_states)
        tmp_9 = optimized_query_states_transform(query_states)
        
        return (tmp_12, tmp_9)
    
    return optimized_transformation