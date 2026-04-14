import torch
import triton
import triton.language as tl

# Pattern matching function for embedding addition
def embedding_addition_pattern(input_ids, position_ids, word_embeddings, position_embeddings):
    # Two embedding operations
    tmp_5 = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(position_ids, position_embeddings, 1, None, 2.0, False, False)
    
    # Element-wise addition
    result = tmp_5 + tmp_6
    return result

def pattern(input_ids, position_ids, word_embeddings, position_embeddings):
    return embedding_addition_pattern(input_ids, position_ids, word_embeddings, position_embeddings)

def replacement_args(input_ids, position_ids, word_embeddings, position_embeddings):
    return (input_ids, position_ids, word_embeddings, position_embeddings)

# Optimized Triton kernel for fused embedding and addition
@triton.jit
def fused_embedding_add_kernel(
    input_ids_ptr,
    position_ids_ptr,
    word_emb_ptr,
    pos_emb_ptr,
    out_ptr,
    input_ids_size,
    position_ids_size,
    embedding_dim,
    vocab_size_word,
    vocab_size_pos,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program handles one output element
    idx = tl.program_id(0)
    
    # Create a mask for valid input positions
    mask = idx < input_ids_size
    
    # Load input IDs and position IDs
    input_id = tl.load(input_ids_ptr + idx, mask=mask, other=0)
    position_id = tl.load(position_ids_ptr + idx, mask=mask, other=0)
    
    if mask:
        # Compute embedding indices (skip padding index 0 with +1 offset as in original)
        word_idx = input_id + 1
        pos_idx = position_id + 1
        
        # Clamp indices to valid vocabulary range
        word_idx = tl.minimum(word_idx, vocab_size_word - 1)
        pos_idx = tl.minimum(pos_idx, vocab_size_pos - 1)
        
        # Load embedding vectors for this element
        # Each embedding is accessed sequentially since we're processing one element at a time
        word_emb_start = word_idx * embedding_dim
        pos_emb_start = pos_idx * embedding_dim
        
        # Initialize result vector
        result_vec = tl.zeros(embedding_dim, dtype=tl.float32)
        
        # Process embedding in chunks for better memory access
        for offset in range(0, embedding_dim, BLOCK_SIZE_N):
            end = tl.minimum(offset + BLOCK_SIZE_N, embedding_dim)
            
            # Load word embedding chunk
            word_emb = tl.load(
                word_emb_ptr + word_emb_start + offset,
                mask=offset + tl.arange(0, end - offset) < embedding_dim,
                other=0.0
            )
            
            # Load position embedding chunk  
            pos_emb = tl.load(
                pos_emb_ptr + pos_emb_start + offset,
                mask=offset + tl.arange(0, end - offset) < embedding_dim,
                other=0.0
            )
            
            # Add embeddings
            chunk_result = word_emb + pos_emb
            result_vec = tl.where(
                offset + tl.arange(0, end - offset) < embedding_dim,
                chunk_result,
                result_vec
            )
        
        # Store the result
        tl.store(out_ptr + idx * embedding_dim, result_vec)

@torch.fx.wrap  
def optimized_embedding_addition(input_ids, position_ids, word_embeddings, position_embeddings):
    # Get input dimensions
    batch_size, seq_len = input_ids.shape
    embedding_dim = word_embeddings.shape[1]
    
    # Flatten input IDs for processing
    flat_input_ids = input_ids.flatten()
    flat_position_ids = position_ids.flatten()
    
    # Get vocabulary sizes
    vocab_size_word = word_embeddings.shape[0]
    vocab_size_pos = position_embeddings.shape[0]
    
    # Create output tensor
    output_shape = (batch_size * seq_len, embedding_dim)
    out = torch.empty(output_shape, dtype=torch.float32, device='cuda')
    
    # Calculate grid dimensions
    total_elements = batch_size * seq_len
    BLOCK_SIZE_M = 64  # Process 64 elements per block
    BLOCK_SIZE_N = 128  # Process 128 embedding dimensions per chunk
    
    grid = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    fused_embedding_add_kernel[grid](
        input_ids_ptr=flat_input_ids,
        position_ids_ptr=flat_position_ids,
        word_emb_ptr=word_embeddings,
        pos_emb_ptr=position_embeddings,
        out_ptr=out,
        input_ids_size=total_elements,
        position_ids_size=total_elements,
        embedding_dim=embedding_dim,
        vocab_size_word=vocab_size_word,
        vocab_size_pos=vocab_size_pos,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return optimized_embedding_addition