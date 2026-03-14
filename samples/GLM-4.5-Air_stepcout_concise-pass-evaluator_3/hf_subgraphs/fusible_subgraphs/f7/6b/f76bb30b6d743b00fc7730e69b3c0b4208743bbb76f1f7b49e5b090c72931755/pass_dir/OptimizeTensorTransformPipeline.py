import torch
import triton
import triton.language as tl

def pattern(key_input, query_input):
    """
    Pattern matching for tensor transformation pipeline:
    - Key: view(1, -1, 16, 64) -> transpose(1, 2) -> reshape(16, -1, 64)
    - Query: view(1, 1, 16, 64) -> transpose(1, 2) -> reshape(16, -1, 64)
    - Final transpose on one output: transpose(1, 2)
    """
    # Key tensor transformations
    tmp_3 = key_input.view(1, -1, 16, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_10 = tmp_4.reshape(16, -1, 64)
    
    # Query tensor transformations
    tmp_7 = query_input.view(1, 1, 16, 64)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_9 = tmp_8.reshape(16, -1, 64)
    
    # Final transpose for one of the outputs
    tmp_12 = tmp_10.transpose(1, 2)
    
    return (tmp_9, tmp_12)

@triton.jit
def optimized_tensor_transform_kernel(
    key_input_ptr, query_input_ptr,
    output1_ptr, output2_ptr,
    batch_size, seq_len_key, seq_len_query,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized kernel for tensor transformations that eliminates intermediate memory allocations
    and performs direct memory layout conversion
    """
    pid = tl.program_id(0)
    
    # Process query tensor transformation (simpler, fixed seq_len)
    if seq_len_query == 1:
        # Direct reshape for query: [1, 1, 1024] -> [16, 64]
        query_offset = pid * 1024 + tl.arange(0, 1024)
        query_data = tl.load(query_input_ptr + query_offset, mask=query_offset < 1024, other=0.0)
        
        # Reshape and store directly as [16, 64]
        output1_offset = pid * 16 * 64
        for i in range(16):
            row_offset = output1_offset + i * 64
            row_data = query_data[i*64:(i+1)*64]
            tl.store(output1_ptr + row_offset, row_data)
    
    # Process key tensor transformation (variable seq_len)
    if seq_len_key > 0:
        key_offset = pid * seq_len_key * 1024 + tl.arange(0, seq_len_key * 1024)
        key_data = tl.load(key_input_ptr + key_offset, mask=tl.arange(0, seq_len_key * 1024) < seq_len_key * 1024, other=0.0)
        
        # Direct reshape for key: [seq_len, 1024] -> [seq_len, 16, 64] -> [16, seq_len, 64] -> [16, seq_len*64]
        # This combines view(1, -1, 16, 64) -> transpose(1, 2) -> reshape(16, -1, 64)
        key_reshaped = key_data.reshape(seq_len_key, 16, 64)
        key_transposed = key_reshaped.transpose(0, 1)  # [seq_len, 16, 64] -> [16, seq_len, 64]
        key_final = key_transposed.reshape(16, seq_len_key * 64)
        
        output2_offset = pid * 16 * seq_len_key * 64
        for i in range(16):
            row_offset = output2_offset + i * seq_len_key * 64
            row_data = key_final[i, :]
            tl.store(output2_ptr + row_offset, row_data)

@torch.fx.wrap
def optimized_tensor_transform(key_input, query_input):
    """
    Wrapper function for optimized tensor transformations using Triton
    """
    batch_size = key_input.shape[0]
    
    # Determine sequence lengths
    if len(key_input.shape) == 3:
        seq_len_key = key_input.shape[1]
    else:
        seq_len_key = 1
    
    if len(query_input.shape) == 3:
        seq_len_query = query_input.shape[1] 
    else:
        seq_len_query = 1
    
    # Calculate output sizes
    query_output_size = 16 * 64
    key_output_size = 16 * seq_len_key * 64
    
    # Create output tensors
    output1 = torch.empty((batch_size, query_output_size), dtype=key_input.dtype, device=key_input.device)
    output2 = torch.empty((batch_size, key_output_size), dtype=key_input.dtype, device=key_input.device)
    
    # Optimize block sizes for better GPU occupancy
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid_size = batch_size
    
    optimized_tensor_transform_kernel[grid_size](
        key_input_ptr=key_input,
        query_input_ptr=query_input,
        output1_ptr=output1,
        output2_ptr=output2,
        batch_size=batch_size,
        seq_len_key=seq_len_key,
        seq_len_query=seq_len_query,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Apply final transpose to output2 as required
    if seq_len_key > 1:
        output2_final = output2.reshape(batch_size, 16, seq_len_key * 64).transpose(1, 2)
    else:
        output2_final = output2.reshape(batch_size, 16, -1).transpose(1, 2)
    
    return output1, output2_final

def replacement_args(model, in_0, in_1, in_2, in_3, in_4):
    """Extract arguments for tensor transformation optimization"""
    # This pass matches the tensor transformation part of the computation
    return (in_2, in_3)  # key_input and query_input (might vary by graph pattern)

def replacement_func():
    """Return the optimized tensor transformation function"""
    return optimized_tensor_transform