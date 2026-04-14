import torch
import triton
import triton.language as tl

def pattern(input_ids, word_embeddings, cumsum_result, position_embeddings):
    # Word embedding lookup
    word_emb = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    # Position embedding lookup  
    pos_emb = torch.nn.functional.embedding(cumsum_result, position_embeddings, 1, None, 2.0, False, False)
    # Addition of embeddings
    combined_emb = word_emb + pos_emb
    return combined_emb, word_emb

def replacement_args(input_ids, word_embeddings, cumsum_result, position_embeddings):
    return (input_ids, word_embeddings, cumsum_result, position_embeddings)

@triton.jit
def fused_embedding_kernel(
    input_ids_ptr,
    cumsum_result_ptr,
    word_embeddings_ptr,
    position_embeddings_ptr,
    output_ptr,
    word_emb_ptr,
    batch_size,
    seq_len,
    embedding_dim,
    vocab_size_word,
    vocab_size_pos,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program ID
    pid = tl.program_id(0)
    # Each program handles one position in the sequence
    seq_idx = pid % seq_len
    batch_idx = (pid // seq_len) % batch_size
    
    # Load input IDs and position IDs
    input_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)
    pos_id = tl.load(cumsum_result_ptr + batch_idx * seq_len + seq_idx)
    
    # Calculate embedding pointers
    word_emb_base = word_embeddings_ptr + input_id * embedding_dim
    pos_emb_base = position_embeddings_ptr + pos_id * embedding_dim
    output_base = output_ptr + (batch_idx * seq_len + seq_idx) * embedding_dim
    word_emb_out_base = word_emb_ptr + (batch_idx * seq_len + seq_idx) * embedding_dim
    
    # Load embeddings unrolled for better performance
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < embedding_dim
    
    # Load word embeddings
    word_vec = tl.load(word_emb_base + offsets, mask=mask, other=0.0)
    pos_vec = tl.load(pos_emb_base + offsets, mask=mask, other=0.0)
    
    # Add embeddings and store
    combined_vec = word_vec + pos_vec
    tl.store(output_base + offsets, combined_vec, mask=mask)
    tl.store(word_emb_out_base + offsets, word_vec, mask=mask)

@torch.fx.wrap
def fused_embedding_lookup(input_ids, word_embeddings, cumsum_result, position_embeddings):
    # Get tensor shapes
    batch_size, seq_len = input_ids.shape
    embedding_dim = word_embeddings.shape[1]
    vocab_size_word = word_embeddings.shape[0]
    vocab_size_pos = position_embeddings.shape[0]
    
    # Output tensors
    output = torch.empty((batch_size, seq_len, embedding_dim), dtype=word_embeddings.dtype, device=word_embeddings.device)
    word_emb_out = torch.empty((batch_size, seq_len, embedding_dim), dtype=word_embeddings.dtype, device=word_embeddings.device)
    
    # Block size for Triton kernel
    BLOCK_SIZE = 128
    grid_size = batch_size * seq_len
    
    # Launch kernel
    fused_embedding_kernel[grid_size](
        input_ids,
        cumsum_result,
        word_embeddings,
        position_embeddings,
        output,
        word_emb_out,
        batch_size,
        seq_len,
        embedding_dim,
        vocab_size_word,
        vocab_size_pos,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, word_emb_out

def replacement_func():
    return fused_embedding_lookup