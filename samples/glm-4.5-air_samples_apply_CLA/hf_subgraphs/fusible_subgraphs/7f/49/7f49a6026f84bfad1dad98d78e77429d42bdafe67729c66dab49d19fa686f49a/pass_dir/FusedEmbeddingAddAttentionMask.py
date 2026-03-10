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



@torch.fx.wrap
def fused_embedding_forward(input_ids, position_ids, word_embeddings, position_embeddings, attention_mask):
    batch_size, seq_len = input_ids.shape
    hidden_dim = word_embeddings.shape[1]
    
    output = torch.empty(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=input_ids.device)
    
    # Determine block sizes based on input dimensions
    BLOCK_SIZE_M = 64  # Process multiple sequences in parallel
    BLOCK_SIZE_N = 128  # Process multiple positions in parallel
    
    # Calculate grid sizes
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_embedding_kernel[grid_m, grid_n,](
        input_ids,
        position_ids,
        word_embeddings,
        position_embeddings,
        output,
        batch_size,
        seq_len,
        word_embeddings.shape[0],
        hidden_dim,
        word_embeddings.stride(0), word_embeddings.stride(1),
        position_embeddings.stride(0), position_embeddings.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return fused_embedding_forward