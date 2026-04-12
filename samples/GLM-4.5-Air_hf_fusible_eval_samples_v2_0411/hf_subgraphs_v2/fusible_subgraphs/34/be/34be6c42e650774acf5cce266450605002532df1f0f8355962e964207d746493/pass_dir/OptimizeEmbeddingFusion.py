import torch
import triton
import triton.language as tl

@triton.jit
def optimized_position_generation_kernel(
    output_ptr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < seq_len
    
    # Generate positions: cumsum(ones) - ones + 2
    # For position i, this gives: i + 2
    positions = tl.cast(idx, tl.int64) + 2
    
    tl.store(output_ptr + idx, positions, mask=mask)

@triton.jit
def optimized_lookup_kernel(
    input_ids_ptr,
    weights_ptr,
    output_ptr,
    vocab_size: tl.constexpr,
    embed_dim: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = seq_idx < seq_len
    
    # Load input IDs
    input_ids = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx, mask=mask, other=0)
    
    # Embbedding lookup (simple implementation)
    output = tl.zeros((BLOCK_SIZE, embed_dim), dtype=tl.float32)
    
    for k in range(0, BLOCK_SIZE, 1):
        if seq_idx[k] < seq_len and tl.constexpr(True):
            idx = input_ids[k]
            if idx < vocab_size and idx >= 0:
                # Load embedding vector
                embed_start = idx * embed_dim
                embed_vec = tl.load(weights_ptr + embed_start + tl.arange(0, embed_dim))
                output[k, :] = embed_vec
    
    # Store output (transpose for memory layout)
    output = output.T
    for j in range(embed_dim):
        tl.store(output_ptr + (batch_idx * embed_dim * seq_len) + (seq_idx * embed_dim) + j, 
                output[j, :], mask=mask)

@torch.fx.wrap
def optimize_embedding_fusion(input_ids, word_embeddings, position_embeddings, embed_dim=768):
    seq_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]
    
    # Generate positions optimized
    positions = torch.empty((batch_size, seq_len), dtype=torch.int64, device=input_ids.device)
    num_elements = batch_size * seq_len
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_position_generation_kernel[(num_programs,)](
        output_ptr=positions,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Word embedding lookup
    word_emb_out = torch.empty((batch_size, seq_len, embed_dim), 
                              dtype=torch.float32, device=input_ids.device)
    vocab_size_word = word_embeddings.shape[0]
    
    grid_word = (batch_size, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    optimized_lookup_kernel[grid_word](
        input_ids_ptr=input_ids,
        weights_ptr=word_embeddings,
        output_ptr=word_emb_out,
        vocab_size=vocab_size_word,
        embed_dim=embed_dim,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Position embedding lookup
    pos_emb_out = torch.empty((batch_size, seq_len, embed_dim), 
                             dtype=torch.float32, device=input_ids.device)
    vocab_size_pos = position_embeddings.shape[0]
    
    grid_pos = (batch_size, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    optimized_lookup_kernel[grid_pos](
        input_ids_ptr=positions,
        weights_ptr=position_embeddings,
        output_ptr=pos_emb_out,
        vocab_size=vocab_size_pos,
        embed_dim=embed_dim,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Add embeddings
    result = word_emb_out + pos_emb_out
    
    return result

def pattern(input_ids, word_embeddings, position_embeddings):
    tmp_10 = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    
    tmp_11 = torch.ones((1, input_ids.shape[1]), dtype=torch.int64, device=torch.device(type='cuda', index=0))
    tmp_12 = torch.cumsum(tmp_11, dim=1)
    tmp_13 = tmp_12 - tmp_11
    tmp_14 = tmp_13 + 2
    
    tmp_15 = torch.nn.functional.embedding(tmp_14, position_embeddings, 1, None, 2.0, False, False)
    
    tmp_16 = tmp_10 + tmp_15
    
    return tmp_16

def replacement_args(input_ids, word_embeddings, position_embeddings):
    return (input_ids, word_embeddings, position_embeddings)

def replacement_func():
    return optimize_embedding_fusion