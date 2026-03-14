import torch
import triton
import triton.language as tl

def pattern(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    """Match the embedding lookup operation"""
    return torch.nn.functional.embedding(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

def replacement_args(input_ids, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return (input_ids, weight)

@triton.jit
def embedding_kernel(
    input_ids_ptr, 
    weight_ptr, 
    output_ptr,
    batch_size, 
    seq_len, 
    num_embeddings,
    embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for embedding lookup"""
    # Calculate program position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate linear index for output
    linear_idx = batch_idx * seq_len + seq_idx
    
    # Only process if within bounds
    if linear_idx >= batch_size * seq_len:
        return
        
    # Compute memory addresses
    output_offset = linear_idx * embedding_dim
    input_id = tl.load(input_ids_ptr + linear_idx)
    
    # Ensure input_id is within valid range
    if 0 <= input_id < num_embeddings:
        weight_offset = input_id * embedding_dim
        # Load embedding vector
        embedding_vector = tl.load(weight_ptr + weight_offset, mask=(tl.arange(0, BLOCK_SIZE) < embedding_dim))
        # Store output
        tl.store(output_ptr + output_offset, embedding_vector, mask=(tl.arange(0, BLOCK_SIZE) < embedding_dim))
    else:
        # Handle padding or invalid IDs by zeroing out
        zeros = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        tl.store(output_ptr + output_offset, zeros, mask=(tl.arange(0, BLOCK_SIZE) < embedding_dim))

@torch.fx.wrap
def optimized_embedding_lookup(input_ids, weight):
    """High-performance embedding lookup using Triton"""
    # Get input shapes
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = weight.shape
    
    # Create output tensor (same dtype as weight)
    output = torch.empty((batch_size, seq_len, embedding_dim), 
                        dtype=weight.dtype, device=weight.device)
    
    # Set block size based on embedding dimension
    BLOCK_SIZE = max(32, embedding_dim)  # Use larger blocks for better utilization
    
    # Calculate grid size
    grid = (
        batch_size,
        seq_len,
    )
    
    # Launch kernel
    embedding_kernel[grid](
        input_ids,
        weight,
        output,
        batch_size,
        seq_len,
        num_embeddings,
        embedding_dim,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_embedding_lookup