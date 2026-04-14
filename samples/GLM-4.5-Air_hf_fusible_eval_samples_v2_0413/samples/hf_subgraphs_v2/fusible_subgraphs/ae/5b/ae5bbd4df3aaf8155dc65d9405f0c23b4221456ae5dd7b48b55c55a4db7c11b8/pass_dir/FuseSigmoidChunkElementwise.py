import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(input_tensor, gru_const):
    """Fuse sigmoid activation, chunking, and element-wise operations
    Original sequence: sigmoid -> chunk -> element-wise math
    Optimized: Directly compute complex element-wise operations
    """
    # Original computation from model.py:
    sigmoid_out = torch.sigmoid(input_tensor)
    chunk = sigmoid_out.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * gru_const
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    result = tmp_12 + 2.0
    
    # Return the final result - this is what's used downstream
    return result

# Argument extraction function
def replacement_args(input_tensor, gru_const):
    return (input_tensor, gru_const)

# Optimized kernel that fuses sigmoid + chunk + element-wise operations
@triton.jit
def fused_sigmoid_elementwise_kernel(
    input_ptr,
    gru_const_ptr,
    output_ptr,
    batch_size,
    num_heads, 
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID mapping
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_id = tl.program_id(2)
    
    # Calculate offset in input tensor [batch_size, num_heads, seq_len, 2]
    input_offset = (batch_id * num_heads * seq_len * 2 + 
                   head_id * seq_len * 2 + 
                   seq_id * 2 + 
                   tl.arange(0, BLOCK_SIZE))
    
    # Calculate output offset - same shape as input but flattened differently
    output_offset = (batch_id * num_heads * seq_len * 1 +  # Final output is [1, num_heads, seq_len, 1] essentially
                    head_id * seq_len * 1 + 
                    seq_id * 1)
    
    # Compute how many elements we need to process per thread
    total_elements = batch_size * num_heads * seq_len * 2
    elements_per_thread = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Load input tensor values (both chunks: [batch_size, num_heads, seq_len, 2])
    input_vals = tl.load(input_ptr + input_offset, mask=input_offset < total_elements, other=0.0)
    
    # Load gru constant (shape [1, num_heads, 1, 1] -> broadcast to current head)
    gru_const = tl.load(gru_const_ptr + head_id * 1 * 1 * 1 + tl.arange(0, 1))
    
    # Compute sigmoid for both chunks
    sigmoid_val_1 = 1.0 / (1.0 + tl.exp(-input_vals[0:BLOCK_SIZE:2]))  # First chunk (tmp_8)
    sigmoid_val_2 = 1.0 / (1.0 + tl.exp(-input_vals[1:BLOCK_SIZE:2]))  # Second chunk (tmp_9)
    
    # Fuse all element-wise operations:
    # tmp_10 = tmp_9 * in_2
    # tmp_11 = tmp_10 - 1.0  
    # tmp_12 = tmp_8 * tmp_11
    # result = tmp_12 + 2.0
    fused_result = sigmoid_val_1 * (sigmoid_val_2 * gru_const - 1.0) + 2.0
    
    # Store result
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE), 
             fused_result, mask=tl.arange(0, BLOCK_SIZE) < elements_per_thread)



# Kernel wrapper
@torch.fx.wrap
def fused_sigmoid_elementwise(input_tensor, gru_const):
    """Fuse sigmoid activation, chunking, and element-wise operations"""
    
    # Get input dimensions 
    batch_size, num_heads, seq_len, input_dim = input_tensor.shape
    # Input should be [batch_size, num_heads, seq_len, 2] after chunking
    
    # Output shape is [batch_size, num_heads, seq_len, 1] after fusion
    output_shape = (batch_size, num_heads, seq_len, 1)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Handle gru constant broadcasting - make it [num_heads, 1, 1]
    gru_const_bcast = gru_const.squeeze(0).squeeze(-1).squeeze(-1)  # [num_heads]
    
    # Set up grid dimensions
    BLOCK_SIZE = 1024
    grid = (
        batch_size,
        num_heads, 
        seq_len,
        (batch_size * num_heads * seq_len * 2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    )
    
    # Launch kernel
    fused_sigmoid_elementwise_kernel[grid](
        input_tensor,
        gru_const_bcast,
        output,
        batch_size,
        num_heads,
        seq_len,
        BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_sigmoid_elementwise