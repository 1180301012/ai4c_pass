import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - exact match from model.py
def pattern(indices, weight_table, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Match the embedding operation from the target computation"""
    tmp_1 = indices.to(device(type='cuda', index=0))
    result = torch.nn.functional.embedding(tmp_1, weight_table, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    return result

# Argument extraction function
def replacement_args(indices, weight_table, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return (indices, weight_table, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

# Optimized Triton kernel for embedding lookup
@triton.jit
def embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """High-performance embedding lookup kernel using 2D grid"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Each program handles one embedding dimension
    embed_idx = pid_m
    output_idx = pid_n
    
    # Load index for this output position
    index = tl.load(indices_ptr + output_idx)
    
    # Ensure index is within bounds using tl.where for type consistency
    valid_index = (index >= 0) & (index < num_embeddings)
    index = tl.where(valid_index, index, 0)
    
    # Weight offset for this embedding dimension
    weight_offset = index * embedding_dim + embed_idx
    
    # Output offset
    output_offset = output_idx * embedding_dim + embed_idx
    
    # Load embedding weight and store to output
    if embed_idx < embedding_dim:
        weight_val = tl.load(weight_ptr + weight_offset)
        tl.store(output_ptr + output_offset, weight_val)

# Kernel wrapper
@torch.fx.wrap
def optimized_embedding_lookup(indices, weight_table, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Optimized embedding lookup using Triton"""
    # Get input shapes
    seq_len, batch_size = indices.shape
    num_embeddings, embedding_dim = weight_table.shape
    
    # Initialize output tensor
    output = torch.empty(seq_len, batch_size, embedding_dim, 
                        dtype=weight_table.dtype, device=weight_table.device)
    
    # Calculate grid dimensions: embedding_dim x (seq_len * batch_size)
    grid_m = embedding_dim
    grid_n = seq_len * batch_size
    
    # Choose block sizes for better memory coalescing
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Launch kernel
    embedding_kernel[(grid_m, grid_n)](
        indices_ptr=indices,
        weight_ptr=weight_table,
        output_ptr=output,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# Replacement function (must return function reference)
def replacement_func():
    return optimized_embedding_lookup