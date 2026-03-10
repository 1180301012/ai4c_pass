import torch
import triton
import triton.language as tl
from torch import nn
import math

def pattern(input_ids, position_ids, word_embeddings, position_embeddings, attention_mask):
    # Word embedding lookup
    word_emb = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    # Position embedding lookup  
    pos_emb = torch.nn.functional.embedding(position_ids, position_embeddings, 1, None, 2.0, False, False)
    # Add embeddings
    combined_emb = word_emb + pos_emb
    # Unsqueeze attention mask and apply
    attention_mask_expanded = attention_mask.unsqueeze(-1)
    masked_emb = combined_emb * attention_mask_expanded
    # Type conversion
    result = masked_emb.to(torch.float32)
    return result

def replacement_args(input_ids, position_ids, word_embeddings, position_embeddings, attention_mask):
    return (input_ids, position_ids, word_embeddings, position_embeddings, attention_mask)

@triton.jit
def fused_embedding_kernel(
    input_ids_ptr,
    position_ids_ptr,
    word_embeddings_ptr,
    position_embeddings_ptr,
    attention_mask_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    word_emb_stride_0, word_emb_stride_1,
    pos_emb_stride_0, pos_emb_stride_1,
    out_stride_0, out_stride_1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID mapping for 3D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, batch_size)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min(n_start + BLOCK_SIZE_N, seq_len)
    k_start = pid_k * BLOCK_SIZE_K
    k_end = min(k_start + BLOCK_SIZE_K, hidden_dim)
    
    # Shared memory for cooperative loading
    word_smem = tl.arange(0, BLOCK_SIZE_K * BLOCK_SIZE_M)
    pos_smem = tl.arange(0, BLOCK_SIZE_K * BLOCK_SIZE_M) 
    
    # Process block
    for m in range(m_start, m_end, BLOCK_SIZE_M):
        for n in range(n_start, n_end, BLOCK_SIZE_N):
            for k in range(k_start, k_end, BLOCK_SIZE_K):
                # Get word embedding indices for this block
                batch_idx = tl.arange(0, BLOCK_SIZE_M) + m
                seq_idx = tl.arange(0, BLOCK_SIZE_N) + n
                word_indices = tl.load(input_ids_ptr + batch_idx[:, None] * seq_len + seq_idx[None, :])
                
                # Load word embeddings for this block
                word_offsets = word_indices * hidden_dim + k
                word_vals = tl.load(word_embeddings_ptr + word_offsets, mask=batch_idx[:, None] < batch_size & seq_idx[None, :] < seq_len, other=0.0)
                
                # Load position embeddings for this block  
                pos_offsets = (batch_idx[:, None] * seq_len + seq_idx[None, :]) * pos_emb_stride_0 + k
                pos_vals = tl.load(position_embeddings_ptr + pos_offsets, mask=batch_idx[:, None] < batch_size & seq_idx[None, :] < seq_len, other=0.0)
                
                # Load attention mask for this block
                mask_vals = tl.load(attention_mask_ptr + batch_idx[:, None] * seq_len + seq_idx[None, :], mask=batch_idx[:, None] < batch_size & seq_idx[None, :] < seq_len, other=1.0)
                
                # Add embeddings and apply mask
                combined = (word_vals + pos_vals) * mask_vals.float()
                
                # Store result as float32
                out_offsets = (batch_idx[:, None] * out_stride_0 + seq_idx[None, :] * out_stride_1 + k)
                tl.store(output_ptr + out_offsets, combined, mask=batch_idx[:, None] < batch_size & seq_idx[None, :] < seq_len & k < hidden_dim)

@torch.fx.wrap
def fused_embedding_forward(input_ids, position_ids, word_embeddings, position_embeddings, attention_mask):
    batch_size, seq_len = input_ids.shape
    hidden_dim = word_embeddings.shape[1]
    
    output = torch.empty(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=input_ids.device)
    
    # Autotune block sizes based on input dimensions
    BLOCK_SIZE_M = min(32, batch_size)
    BLOCK_SIZE_N = min(128, seq_len) 
    BLOCK_SIZE_K = min(32, hidden_dim)
    
    # Calculate grid sizes
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (hidden_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernel with optimized configuration
    fused_embedding_kernel[grid_m, grid_n, grid_k,](
        input_ids,
        position_ids,
        word_embeddings,
        position_embeddings,
        attention_mask,
        output,
        batch_size,
        seq_len,
        hidden_dim,
        word_embeddings.stride(0), word_embeddings.stride(1),
        position_embeddings.stride(0), position_embeddings.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return fused_embedding_forward