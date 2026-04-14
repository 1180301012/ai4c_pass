import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(query, weight, bias):
    """Fuse linear transformation, view operation, and sum reduction
    Original sequence: linear -> view -> sum
    Optimized: Directly compute linear with sum along last dimension
    """
    # Original computation from model.py:
    linear = torch.nn.functional.linear(query, weight, bias)
    tmp_4 = linear.view(1, linear.size(1), linear.size(2), 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    
    # Note: The pattern must return what would be observable
    # In this case, tmp_5 is used in subsequent operations
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
    hidden_dim,
    output_dim,
):
    # Program ID mapping
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_id = tl.program_id(2)
    chunk_id = tl.program_id(3)  # 0 for first chunk, 1 for second chunk
    
    # Calculate output offset for this position and chunk
    # Output shape: [batch_size, num_heads, seq_len, output_dim//2]
    output_offset = (batch_id * num_heads * seq_len * (output_dim // 2) + 
                    head_id * seq_len * (output_dim // 2) + 
                    seq_id * (output_dim // 2) + 
                    chunk_id)
    
    # Check bounds
    max_output_size = batch_size * num_heads * seq_len * (output_dim // 2)
    if output_offset >= max_output_size:
        return
    
    # Load bias for this chunk (4 elements)
    bias_offset = chunk_id * 4
    bias = tl.load(bias_ptr + bias_offset + tl.arange(0, 4))
    
    # Load weight for this chunk [4, hidden_dim]
    weight_offset = bias_offset * hidden_dim
    weight_chunk = tl.load(weight_ptr + weight_offset + tl.arange(0, hidden_dim)[:, None] + tl.arange(0, 4)[None, :])
    
    # Calculate query offset for current position
    query_offset = (batch_id * num_heads * seq_len * hidden_dim + 
                   head_id * seq_len * hidden_dim + 
                   seq_id * hidden_dim)
    
    # Load query vector [hidden_dim]
    query_vec = tl.load(query_ptr + query_offset + tl.arange(0, hidden_dim))
    
    # Compute linear combination for this chunk: bias + query_vec @ weight_chunk.T
    # Need to transpose weight_chunk for matmul
    result = tl.sum(query_vec[:, None] * weight_chunk, 0) + bias
    
    # Store the summed result
    tl.store(output_ptr + output_offset, result)

# Kernel wrapper
@torch.fx.wrap
def fused_linear_sum(query, weight, bias):
    # Get dimensions based on input shapes
    batch_size, num_heads, seq_len, hidden_dim = query.shape
    # Last dimension of weight is hidden_dim, first dimension is output_dim
    output_dim = weight.shape[0]
    
    output_shape = (batch_size, num_heads, seq_len, output_dim // 2)
    output = torch.empty(output_shape, dtype=query.dtype, device=query.device)
    
    # Set up grid dimensions: [batch_size, num_heads, seq_len, num_chunks]
    num_chunks = output_dim // 4  # Should be 2 (8/4)
    grid = (
        batch_size,
        num_heads, 
        seq_len,
        num_chunks
    )
    
    # Launch kernel
    fused_linear_sum_kernel[grid](
        query,
        weight,
        bias,
        output,
        batch_size,
        num_heads,
        seq_len,
        hidden_dim,
        output_dim
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_linear_sum