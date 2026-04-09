import torch
import triton
import triton.language as tl

# Pattern matching function for embedding addition fusion
def pattern(in_0, in_4, in_5, in_3):
    # Two embedding operations followed by addition
    word_emb = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    pos_emb = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    result = word_emb + pos_emb
    return result

# Argument extraction function
def replacement_args(in_0, in_4, in_5, in_3):
    return (in_0, in_4, in_5, in_3)

# Optimized Triton kernel for fused embedding addition
@triton.jit
def fused_embedding_add_kernel(
    input_ids_ptr,
    word_embeddings_ptr,
    position_ids_ptr,
    position_embeddings_ptr,
    output_ptr,
    input_ids_stride_0,
    input_ids_stride_1,
    word_embeddings_stride_0,
    word_embeddings_stride_1,
    position_embeddings_stride_0,
    position_embeddings_stride_1,
    output_stride_0,
    output_stride_1,
    vocab_size_word,
    embedding_dim_word,
    vocab_size_pos,
    embedding_dim_pos,
    n_elements_batch,
    n_elements_seq,
    BLOCK_SIZE: tl.constexpr,
):
    # Batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate output pointer offset
    output_offset = batch_idx * output_stride_0 + seq_idx * output_stride_1
    
    # Load input IDs
    input_id = tl.load(input_ids_ptr + batch_idx * input_ids_stride_0 + seq_idx * input_ids_stride_1)
    
    # Load position IDs
    position_id = tl.load(position_ids_ptr + batch_idx * input_ids_stride_0 + seq_idx * input_ids_stride_1)
    
    # Ensure IDs are within bounds using conditional logic
    input_id = tl.where(input_id < vocab_size_word, input_id, vocab_size_word - 1)
    position_id = tl.where(position_id < vocab_size_pos, position_id, vocab_size_pos - 1)
    
    # Calculate word embedding offset
    word_emb_offset = input_id * word_embeddings_stride_0
    
    # Calculate position embedding offset
    pos_emb_offset = position_id * position_embeddings_stride_0
    
    # Load word embedding
    word_emb = tl.load(
        word_embeddings_ptr + word_emb_offset + tl.arange(0, BLOCK_SIZE),
        mask=tl.arange(0, BLOCK_SIZE) < embedding_dim_word,
        other=0.0
    )
    
    # Load position embedding
    pos_emb = tl.load(
        position_embeddings_ptr + pos_emb_offset + tl.arange(0, BLOCK_SIZE),
        mask=tl.arange(0, BLOCK_SIZE) < embedding_dim_pos,
        other=0.0
    )
    
    # Add embeddings
    result = word_emb + pos_emb
    
    # Store result
    tl.store(
        output_ptr + output_offset + tl.arange(0, BLOCK_SIZE),
        result,
        mask=tl.arange(0, BLOCK_SIZE) < min(embedding_dim_word, embedding_dim_pos)
    )

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_embedding_add(input_ids, word_embeddings, position_ids, position_embeddings):
    # Get tensor shapes
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)
    embedding_dim = word_embeddings.size(1)
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, embedding_dim), dtype=word_embeddings.dtype, device=input_ids.device)
    
    # Calculate grid dimensions
    batch_programs = batch_size
    seq_programs = seq_len
    embedding_programs = (embedding_dim + 1023) // 1024
    
    # Launch kernel
    fused_embedding_add_kernel[(batch_programs, seq_programs, embedding_programs)](
        input_ids_ptr=input_ids,
        word_embeddings_ptr=word_embeddings,
        position_ids_ptr=position_ids,
        position_embeddings_ptr=position_embeddings,
        output_ptr=output,
        input_ids_stride_0=input_ids.stride(0),
        input_ids_stride_1=input_ids.stride(1),
        word_embeddings_stride_0=word_embeddings.stride(0),
        word_embeddings_stride_1=word_embeddings.stride(1),
        position_embeddings_stride_0=position_embeddings.stride(0),
        position_embeddings_stride_1=position_embeddings.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        vocab_size_word=word_embeddings.size(0),
        embedding_dim_word=word_embeddings.size(1),
        vocab_size_pos=position_embeddings.size(0),
        embedding_dim_pos=position_embeddings.size(1),
        n_elements_batch=batch_size,
        n_elements_seq=seq_len,
        BLOCK_SIZE=1024
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_embedding_add