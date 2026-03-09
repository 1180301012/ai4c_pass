import torch
import triton
import triton.language as tl

def pattern(inputs_embeds, token_type_embeddings, position_ids, position_embeddings):
    """
    Pattern to match the embedding addition fusion:
    tmp_4 = inputs_embeds + token_type_embeddings  
    tmp_5 = embedding(position_ids, position_embeddings)
    tmp_4 += tmp_5
    This results in: inputs_embeds + token_type_embeddings + embedding_result
    """
    tmp_4 = inputs_embeds + token_type_embeddings
    tmp_5 = torch.nn.functional.embedding(position_ids, position_embeddings, 1, None, 2.0, False, False)
    result = tmp_4 + tmp_5
    return result

def replacement_args(inputs_embeds, token_type_embeddings, position_ids, position_embeddings):
    """Return arguments for embedding addition fusion"""
    return (inputs_embeds, token_type_embeddings, position_ids, position_embeddings)

@triton.jit
def embedding_addition_kernel_32d(
    inputs_embeds_ptr,
    token_type_embeddings_ptr,
    position_embeddings_ptr,
    position_ids_ptr,
    out_ptr,
    batch_size,
    seq_len,
    embed_dim,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for embedding lookup and addition operations"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * seq_len * embed_dim)
    
    # Compute linear indices for batch and sequence
    linear_idx = offsets // embed_dim
    seq_idx = offsets % embed_dim
    
    batch_idx = linear_idx // seq_len
    local_seq_idx = linear_idx % seq_len
    
    if not tl.any(mask):
        return
    
    # Load inputs_embeds
    inputs_embeds_offset = linear_idx * embed_dim + seq_idx
    inputs_embeds = tl.load(inputs_embeds_ptr + inputs_embeds_offset, mask=mask, other=0.0)
    
    # Load token_type_embeddings  
    token_type_embeddings_offset = (batch_idx * seq_len + local_seq_idx) * embed_dim + seq_idx
    token_type_embeddings = tl.load(token_type_embeddings_ptr + token_type_embeddings_offset, mask=mask, other=0.0)
    
    # Load position_ids and get embedding index for this position
    # Load position_id for current sequence position
    if local_seq_idx < seq_len:
        pos_id_offset = batch_idx * seq_len + local_seq_idx
        position_id = tl.load(position_ids_ptr + pos_id_offset, other=0)
        position_id = tl.max(0, tl.min(position_id, vocab_size - 1))  # Clamp to valid range
    else:
        position_id = 0
    
    # Load embedding for this position (broadcast across embedding dimension)
    # Only one thread per position loads the embedding to avoid excessive memory reads
    if seq_idx == 0:  # Only embed_dim=0 thread loads per position
        embedding_offset = position_id * embed_dim
        embedding = tl.load(position_embeddings_ptr + embedding_offset + tl.arange(0, embed_dim), 
                           mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
    else:
        # Other threads in the warp get the embedding via shared memory or computation
        # For simplicity, we'll compute this step differently
        pid_flat = batch_idx * seq_len + local_seq_idx
        embedding_offset = position_id * embed_dim
        embedding = tl.load(position_embeddings_ptr + embedding_offset + tl.arange(0, embed_dim), 
                           mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
    
    # Fused computation: inputs_embeds + token_type_embeddings + position_embedding
    result = inputs_embeds + token_type_embeddings + embedding
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_embedding_addition_32d(inputs_embeds, token_type_embeddings, position_ids, position_embeddings):
    """Optimized fused embedding addition for 32-dimensional models"""
    # Validate input shapes
    if inputs_embeds.dim() != 3 or token_type_embeddings.dim() != 3:
        # Fallback to separate operations
        tmp_4 = inputs_embeds + token_type_embeddings
        tmp_5 = torch.nn.functional.embedding(position_ids, position_embeddings, 1, None, 2.0, False, False)
        return tmp_4 + tmp_5
    
    batch_size, seq_len, embed_dim = inputs_embeds.shape
    
    # Validate dimensions
    if embed_dim != 32:
        # Fallback for non-32 dimensions
        tmp_4 = inputs_embeds + token_type_embeddings  
        tmp_5 = torch.nn.functional.embedding(position_ids, position_embeddings, 1, None, 2.0, False, False)
        return tmp_4 + tmp_5
        
    # Validate other tensor shapes
    if (token_type_embeddings.shape != inputs_embeds.shape or
        position_ids.shape[0] != batch_size or 
        position_ids.shape[1] != seq_len or
        position_embeddings.shape[1] != embed_dim):
        # Fallback for shape mismatches
        tmp_4 = inputs_embeds + token_type_embeddings
        tmp_5 = torch.nn.functional.embedding(position_ids, position_embeddings, 1, None, 2.0, False, False)
        return tmp_4 + tmp_5
    
    vocab_size = position_embeddings.shape[0]
    n_elements = batch_size * seq_len * embed_dim
    
    out = torch.empty_like(inputs_embeds)
    
    # Launch kernel
    BLOCK_SIZE = 256  # Optimal block size for good GPU occupancy
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    embedding_addition_kernel_32d[(num_programs,)](
        inputs_embeds_ptr=inputs_embeds,
        token_type_embeddings_ptr=token_type_embeddings,
        position_embeddings_ptr=position_embeddings,
        position_ids_ptr=position_ids,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_embedding_addition_32d