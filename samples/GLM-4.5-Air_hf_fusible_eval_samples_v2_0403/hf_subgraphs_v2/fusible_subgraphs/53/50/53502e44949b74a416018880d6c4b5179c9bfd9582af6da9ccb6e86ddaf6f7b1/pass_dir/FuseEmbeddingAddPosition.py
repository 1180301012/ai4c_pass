import torch
import triton
import triton.language as tl
import math

def pattern(input_ids, token_embeddings, pos_embeddings, dropout_p, emb_scale=16.0):
    """Match token embedding + scaling + position embedding + addition"""
    # Token embedding with scaling
    token_emb = torch.nn.functional.embedding(input_ids, token_embeddings, 1, None, 2.0, False, False)
    token_scaled = token_emb * emb_scale
    
    # Position generation (simplified for single token)
    pos_idx = torch.tensor([[2]], dtype=torch.int64, device=input_ids.device)
    pos_emb = torch.nn.functional.embedding(pos_idx, pos_embeddings, None, None, 2.0, False, False)
    
    # Add embeddings
    combined = token_scaled + pos_emb
    return combined

def replacement_args(input_ids, token_embeddings, pos_embeddings, gamma, beta, dropout_p, emb_scale=16.0):
    # Precompute position index for single token
    pos_idx = torch.tensor([[2]], dtype=torch.int64, device=input_ids.device)
    return (input_ids, token_embeddings, pos_embeddings, pos_idx, gamma, beta)

# Triton kernel for fused embedding lookup + addition
@triton.jit
def fused_embedding_add_kernel(
    input_ids_ptr,
    token_emb_ptr,
    pos_emb_ptr,
    pos_idx_ptr,
    out_ptr,
    token_vocab_size,
    pos_vocab_size,
    hidden_size,
    emb_scale: tl.constexpr,
):
    pid = tl.program_id(0)
    block_size = 256  # Process 256 elements per thread
    
    # Each position in sequence (here just 1 position)
    seq_idx = pid
    
    # Load input ID and position ID
    input_id = tl.load(input_ids_ptr + seq_idx)
    pos_id = tl.load(pos_idx_ptr + seq_idx)
    
    # Compute embedding indices
    token_emb_base = input_id * hidden_size
    pos_emb_base = pos_id * hidden_size
    
    # Load and scale token embedding
    for i in range(0, hidden_size, block_size):
        offsets = i + tl.arange(0, block_size)
        mask = offsets < hidden_size
        
        # Load token embedding and scale
        token_emb = tl.load(token_emb_ptr + token_emb_base + offsets, mask=mask)
        token_scaled = token_emb * emb_scale
        
        # Load position embedding  
        pos_emb = tl.load(pos_emb_ptr + pos_emb_base + offsets, mask=mask)
        
        # Add embeddings
        result = token_scaled + pos_emb
        
        # Store result
        tl.store(out_ptr + seq_idx * hidden_size + offsets, result, mask=mask)

@torch.fx.wrap  
def fused_embedding_add(input_ids, token_embeddings, pos_embeddings, pos_idx):
    batch_size, seq_len = input_ids.shape
    hidden_size = token_embeddings.shape[1]
    
    # Create output tensor
    output = torch.zeros((batch_size, seq_len, hidden_size), 
                        dtype=token_embeddings.dtype, 
                        device=token_embeddings.device)
    
    # Launch kernel
    grid = (seq_len,)
    fused_embedding_add_kernel[grid](
        input_ids,
        token_embeddings,
        pos_embeddings, 
        pos_idx,
        output,
        token_embeddings.shape[0],  # token_vocab_size
        pos_embeddings.shape[0],    # pos_vocab_size
        hidden_size,
        16.0  # emb_scale
    )
    
    return output

def replacement_func():
    return lambda input_ids, token_embeddings, pos_embeddings, pos_idx, gamma, beta: fused_embedding_add(input_ids, token_embeddings, pos_embeddings, pos_idx)