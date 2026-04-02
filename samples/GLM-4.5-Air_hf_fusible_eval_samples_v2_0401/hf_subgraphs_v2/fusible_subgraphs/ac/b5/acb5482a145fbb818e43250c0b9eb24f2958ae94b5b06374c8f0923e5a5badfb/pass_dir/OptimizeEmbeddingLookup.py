import torch
import triton
import triton.language as tl
import math

def pattern(input_ids, embedding_weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    """Match the embedding lookup operation"""
    embedding_result = torch.nn.functional.embedding(input_ids, embedding_weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    return embedding_result

def replacement_args(input_ids, embedding_weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return (input_ids, embedding_weights)

@triton.jit
def triton_embedding_lookup_kernel(
    input_ids_ptr,
    embedding_weights_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    input_batch,
    input_seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one element in the output tensor
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # sequence dimension
    pid_k = tl.program_id(2)  # embedding dimension
    
    # Calculate the output index
    batch_offset = pid_m * input_seq_len * embedding_dim
    seq_offset = pid_n * embedding_dim
    embed_offset = pid_k
    
    output_index = batch_offset + seq_offset + embed_offset
    
    # Boundary checks
    if pid_m >= input_batch or pid_n >= input_seq_len or pid_k >= embedding_dim:
        return
    
    # Load the input ID (index into embedding table)
    input_id = tl.load(input_ids_ptr + pid_m * input_seq_len + pid_n)
    
    # Handle potential out-of-bound indices
    if input_id < 0 or input_id >= num_embeddings:
        # Return zeros for out-of-bound or padding indices
        tl.store(output_ptr + output_index, tl.float32(0.0))
        return
    
    # Calculate the embedding weight index
    embed_weight_index = input_id * embedding_dim + pid_k
    
    # Load the embedding weight
    embedding_weight = tl.load(embedding_weights_ptr + embed_weight_index, mask=embed_weight_index < (num_embeddings * embedding_dim), other=0.0)
    
    # Store the result
    tl.store(output_ptr + output_index, embedding_weight)

@torch.fx.wrap
def triton_embedding_lookup(input_ids, embedding_weights):
    """
    Optimized Triton implementation of embedding lookup.
    
    Args:
        input_ids: Tensor of shape [batch_size, seq_len] with indices
        embedding_weights: Tensor of shape [num_embeddings, embedding_dim] with embedding vectors
        
    Returns:
        Tensor of shape [batch_size, seq_len, embedding_dim] with embedding results
    """
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = embedding_weights.shape
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, embedding_dim), dtype=embedding_weights.dtype, device=input_ids.device)
    
    # Block sizes for better GPU utilization
    BLOCK_SIZE_M = 4  # Process 4 batch items per program
    BLOCK_SIZE_N = 32  # Process 32 sequence positions per program
    BLOCK_SIZE_K = 64   # Process 64 embedding dimensions per program
    
    # Calculate grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (embedding_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch the kernel
    triton_embedding_lookup_kernel[(grid_m, grid_n, grid_k)](
        input_ids_ptr=input_ids,
        embedding_weights_ptr=embedding_weights,
        output_ptr=output,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        input_batch=batch_size,
        input_seq_len=seq_len,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return triton_embedding_lookup