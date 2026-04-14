import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(query, weight, bias):
    """Fuse linear transformation, view operation, and sum reduction for base models (12 heads)
    Original sequence: linear -> view(1,12,199,2,4) -> sum
    Optimized: Directly compute linear with sum along last dimension
    """
    # Exact computation from wavlm_base model.py:
    linear = torch.nn.functional.linear(query, weight, bias)
    tmp_4 = linear.view(1, 12, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    
    # Return the result that would be observable
    return tmp_5

# Argument extraction function
def replacement_args(query, weight, bias):
    return (query, weight, bias)

# Optimized kernel that fuses linear + view + sum
@triton.jit
def fused_linear_sum_kernel(
    query_ptr, 
    weight_ptr, 
    bias_ptr,
    output_ptr,
    batch_size,
    num_heads,
    seq_len,
):
    # Program ID mapping: each program handles one (batch, head, seq) position
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_id = tl.program_id(2)
    
    # Use literal constants directly for Triton compatibility
    # Calculate base offset for this position using literal 64
    query_offset = (batch_id * num_heads * seq_len * 64 + 
                   head_id * seq_len * 64 + 
                   seq_id * 64)
    
    # Load query vector [64] using literal constant
    query_vec = tl.load(query_ptr + query_offset + tl.arange(0, 64))
    
    # Process both chunks (each chunk has 4 weights/outputs that we need to sum)
    # and store results directly without using lists
    
    # Chunk 0: weights [0:4], bias [0:4]
    bias_offset0 = 0
    bias0 = tl.load(bias_ptr + bias_offset0 + tl.arange(0, 4))
    weight_offset0 = bias_offset0 * 64
    weight_chunk0 = tl.load(weight_ptr + weight_offset0 + 
                           tl.arange(0, 64)[:, None] + tl.arange(0, 4)[None, :])
    linear_result0 = tl.sum(query_vec[:, None] * weight_chunk0, 0) + bias0
    chunk_sum0 = tl.sum(linear_result0)
    
    # Chunk 1: weights [4:8], bias [4:8]  
    bias_offset1 = 4
    bias1 = tl.load(bias_ptr + bias_offset1 + tl.arange(0, 4))
    weight_offset1 = bias_offset1 * 64
    weight_chunk1 = tl.load(weight_ptr + weight_offset1 + 
                           tl.arange(0, 64)[:, None] + tl.arange(0, 4)[None, :])
    linear_result1 = tl.sum(query_vec[:, None] * weight_chunk1, 0) + bias1
    chunk_sum1 = tl.sum(linear_result1)
    
    # Store the 2 results (one for each chunk) separately
    output_offset0 = (batch_id * num_heads * seq_len * 2 + 
                     head_id * seq_len * 2 + 
                     seq_id * 2)
    
    output_offset1 = output_offset0 + 1
    
    tl.store(output_ptr + output_offset0 + tl.arange(0, 1), chunk_sum0)
    tl.store(output_ptr + output_offset1 + tl.arange(0, 1), chunk_sum1)

# Kernel wrapper
@torch.fx.wrap
def fused_linear_sum(query, weight, bias):
    # Get dimensions based on input shapes
    batch_size, num_heads, seq_len, hidden_dim = query.shape
    
    # Output shape is [batch_size, num_heads, seq_len, 2] 
    output_shape = (batch_size, num_heads, seq_len, 2)
    output = torch.empty(output_shape, dtype=query.dtype, device=query.device)
    
    # Set up grid dimensions: [batch_size, num_heads, seq_len]
    # Each program handles one (batch, head, seq) position
    grid = (
        batch_size,
        num_heads, 
        seq_len
    )
    
    # Launch kernel
    fused_linear_sum_kernel[grid](
        query,
        weight,
        bias,
        output,
        batch_size,
        num_heads,
        seq_len
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_linear_sum