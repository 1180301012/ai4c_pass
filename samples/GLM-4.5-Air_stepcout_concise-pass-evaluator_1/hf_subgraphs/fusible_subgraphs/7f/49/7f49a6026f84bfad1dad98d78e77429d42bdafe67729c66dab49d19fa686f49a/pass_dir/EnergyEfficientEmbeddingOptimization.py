import torch
import triton
import triton.language as tl

def pattern(attn_mask, input_ids, pos_embeddings, word_embeddings, position_ids):
    # Embedding operations
    tmp_4 = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    tmp_5 = torch.nn.functional.embedding(position_ids, pos_embeddings, 1, None, 2.0, False, False)
    
    # Add embeddings
    tmp_6 = tmp_4 + tmp_5
    
    # Apply attention mask
    tmp_7 = attn_mask.unsqueeze(-1)
    tmp_8 = tmp_6 * tmp_7
    
    # Type conversion
    tmp_9 = tmp_8.to(torch.float32)
    
    return tmp_9

def replacement_args(attn_mask, input_ids, pos_embeddings, word_embeddings, position_ids):
    return (attn_mask, input_ids, pos_embeddings, word_embeddings, position_ids)

@triton.jit
def energy_optimized_embedding_kernel(
    # Optimized parameter order for memory efficiency
    word_embeddings_ptr,
    attn_mask_ptr,
    input_ids_ptr,
    pos_embeddings_ptr,
    position_ids_ptr,
    output_ptr,
    # Shape constants
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    embedding_dim: tl.constexpr,
    vocab_total: tl.constexpr,
    # Optimized block size
    BLOCK_K: tl.constexpr,
):
    # Energy-efficient grid with minimal thread divergence
    pid = tl.program_id(0)
    total_elements = batch_size * seq_len
    linear_pid = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = linear_pid < total_elements
    
    # Convert to batch/sequence coordinates
    seq_idx = linear_pid % seq_len
    batch_idx = linear_pid // seq_len
    
    # Load inputs with shared memory access pattern
    input_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx, mask=mask)
    mask_val = tl.load(attn_mask_ptr + batch_idx * seq_len + seq_idx, mask=mask)
    
    # Process embedding dimensions with vectorization
    embed_idx = tl.arange(0, BLOCK_K)
    embed_mask = embed_idx < embedding_dim
    
    # Word embedding lookup with optimized addressing
    word_offset = input_id * embedding_dim + embed_idx
    word_vecs = tl.load(word_embeddings_ptr + word_offset, mask=embed_mask)
    
    # Position embedding lookup - need to access position_ids, not input_ids
    position_id = tl.load(position_ids_ptr + batch_idx * seq_len + seq_idx, mask=mask)
    pos_offset = position_id * embedding_dim + embed_idx
    pos_vecs = tl.load(pos_embeddings_ptr + pos_offset, mask=embed_mask)
    
    # Fused computation with energy-optimized operations
    result = (tl.cast(word_vecs, tl.float32) + tl.cast(pos_vecs, tl.float32)) * mask_val
    
    # Store result with optimized addressing
    flat_idx = batch_idx * seq_len + seq_idx
    output_offset = flat_idx * embedding_dim + embed_idx
    tl.store(output_ptr + output_offset, result, mask=embed_mask)

@torch.fx.wrap
def energy_optimized_forward(attn_mask, input_ids, pos_embeddings, word_embeddings, position_ids):
    # Get input shapes
    batch_size, seq_len = attn_mask.shape
    _, embedding_dim = word_embeddings.shape
    
    # Energy-optimized block size selection
    total_elements = batch_size * seq_len
    if embedding_dim <= 32:
        BLOCK_K = 32
    elif embedding_dim <= 128:
        BLOCK_K = 64
    else:
        BLOCK_K = 128
    
    # Calculate efficient grid
    grid = ((total_elements + BLOCK_K - 1) // BLOCK_K,)
    
    # Create output tensor
    output = torch.empty(batch_size, seq_len, embedding_dim, dtype=torch.float32, device=attn_mask.device)
    
    # Calculate vocabulary size without forbidden operations
    vocab_total = word_embeddings.shape[0] + pos_embeddings.shape[0]
    
    # Launch energy-optimized kernel with reordered parameters for efficiency
    energy_optimized_embedding_kernel[grid](
        word_embeddings,
        attn_mask,
        input_ids,
        pos_embeddings,
        position_ids,
        output,
        batch_size,
        seq_len,
        embedding_dim,
        vocab_total,
        BLOCK_K,
    )
    
    return output

def replacement_func():
    return energy_optimized_forward