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

@triton.autotune(
    configs=[
        triton.Config(num_warps=4, num_stages=2), 
        triton.Config(num_warps=8, num_stages=2),
        triton.Config(num_warps=4, num_stages=3),
        triton.Config(num_warps=8, num_stages=3),
        triton.Config(num_warps=16, num_stages=2),
        triton.Config(num_warps=16, num_stages=3),
    ],
    key=['batch_size', 'seq_len', 'vocab_size'],
)
@triton.jit
def embedding_addition_kernel_768d(
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
    """Fused kernel for embedding lookup and addition operations (768D optimized)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * seq_len * embed_dim)
    
    if not tl.any(mask):
        return
    
    # Compute indices
    linear_idx = offsets // embed_dim
    seq_idx = offsets % embed_dim
    batch_idx = linear_idx // seq_len
    local_seq_idx = linear_idx % seq_len
    
    # Calculate memory offsets
    inputs_embeds_offset = batch_idx * seq_len * embed_dim + local_seq_idx * embed_dim + seq_idx
    token_type_embeddings_offset = batch_idx * seq_len * embed_dim + local_seq_idx * embed_dim + seq_idx
    
    # Load inputs_embeds and token_type_embeddings
    inputs_embeds = tl.load(inputs_embeds_ptr + inputs_embeds_offset, mask=mask, other=0.0)
    token_type_embeddings = tl.load(token_type_embeddings_ptr + token_type_embeddings_offset, mask=mask, other=0.0)
    
    # Load position_id for current sequence position
    pos_id_offset = batch_idx * seq_len + local_seq_idx
    position_id = tl.load(position_ids_ptr + pos_id_offset, other=0)
    position_id = tl.max(0, tl.min(position_id, vocab_size - 1))
    
    # Load position embedding for this position
    embedding_offset = position_id * embed_dim + seq_idx
    
    # Efficient loading for 768D: coalesced memory accesses
    embedding = tl.load(position_embeddings_ptr + embedding_offset, mask=seq_idx < embed_dim, other=0.0)
    
    # Fused computation: inputs_embeds + token_type_embeddings + position_embedding
    result = inputs_embeds + token_type_embeddings + embedding
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_embedding_addition_768d(inputs_embeds, token_type_embeddings, position_ids, position_embeddings):
    """Optimized fused embedding addition for 768-dimensional models"""
    # Validate input shapes
    if inputs_embeds.dim() != 3 or token_type_embeddings.dim() != 3:
        # Fallback to separate operations
        tmp_4 = inputs_embeds + token_type_embeddings
        tmp_5 = torch.nn.functional.embedding(position_ids, position_embeddings, 1, None, 2.0, False, False)
        return tmp_4 + tmp_5
    
    batch_size, seq_len, embed_dim = inputs_embeds.shape
    
    # Validate dimensions
    if embed_dim != 768:
        # Fallback for non-768 dimensions
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
    
    # Launch kernel with autotuning for 768D
    BLOCK_SIZE = 1024  # Optimal block size for 768-dimensional data
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use autotuned kernel if available
    try:
        embedding_addition_kernel_768d[(num_programs,)](
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
    except:
        # Fallback to simpler kernel if autotuning fails
        @triton.jit
        def fallback_kernel(
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
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < (batch_size * seq_len * embed_dim)
            
            if not tl.any(mask):
                return
            
            linear_idx = offsets // embed_dim
            seq_idx = offsets % embed_dim
            batch_idx = linear_idx // seq_len
            local_seq_idx = linear_idx % seq_len
            
            inputs_embeds_offset = batch_idx * seq_len * embed_dim + local_seq_idx * embed_dim + seq_idx
            inputs_embeds = tl.load(inputs_embeds_ptr + inputs_embeds_offset, mask=mask, other=0.0)
            
            token_type_embeddings_offset = batch_idx * seq_len * embed_dim + local_seq_idx * embed_dim + seq_idx
            token_type_embeddings = tl.load(token_type_embeddings_ptr + token_type_embeddings_offset, mask=mask, other=0.0)
            
            pos_id_offset = batch_idx * seq_len + local_seq_idx
            position_id = tl.load(position_ids_ptr + pos_id_offset, other=0)
            position_id = tl.max(0, tl.min(position_id, vocab_size - 1))
            
            embedding_offset = position_id * embed_dim + seq_idx
            embedding = tl.load(position_embeddings_ptr + embedding_offset, mask=seq_idx < embed_dim, other=0.0)
            
            result = inputs_embeds + token_type_embeddings + embedding
            tl.store(out_ptr + offsets, result, mask=mask)
        
        fallback_kernel[(num_programs,)](
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
    return optimized_embedding_addition_768d