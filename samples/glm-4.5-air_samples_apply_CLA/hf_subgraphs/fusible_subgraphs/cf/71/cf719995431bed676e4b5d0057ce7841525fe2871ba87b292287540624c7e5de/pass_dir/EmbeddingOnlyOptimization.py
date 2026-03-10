import torch
import triton
import triton.language as tl

# Pattern: standalone embedding operation (no slicing)
def pattern(input_ids, weights):
    tmp_0 = weights
    result = torch.nn.functional.embedding(input_ids, tmp_0, 0, None, 2.0, False, False)
    tmp_0 = None
    return result

# Extract arguments for the replacement function
def replacement_args(input_ids, weights):
    return (input_ids, weights)

# Triton kernel for optimized embedding only
@triton.jit
def embedding_only_kernel(
    input_ids_ptr,
    weights_ptr,
    output_ptr,
    n_batch,
    n_seq,
    n_embed,
    vocab_size,
    padding_idx,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_EMBED: tl.constexpr,
):
    # Each program handles one batch x sequence position combination
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    embed_offset = tl.arange(0, BLOCK_SIZE_EMBED)
    
    # Compute input offset
    input_offset = batch_idx * n_seq + seq_idx
    
    # Load input IDs with bounds checking
    input_id = tl.load(input_ids_ptr + input_offset, mask=(input_offset < n_batch * n_seq), other=padding_idx)
    
    # Skip processing if padding index
    if input_id == padding_idx:
        # Store zeros for padding positions
        for embed_idx in range(0, n_embed, BLOCK_SIZE_EMBED):
            embed_offsets = embed_idx + embed_offset
            mask = embed_offsets < n_embed
            tl.store(output_ptr + input_offset * n_embed + embed_offsets, 0.0, mask=mask)
        return
    
    # Check bounds for vocabulary index
    if input_id >= vocab_size:
        input_id = vocab_size - 1  # Use last embedding as fallback
    
    # Load embedding vector with bounds checking
    embed_start = input_id * n_embed
    weights_offsets = embed_start + embed_offset
    weights_mask = weights_offsets < (vocab_size * n_embed)
    
    # Load embedding vector
    embed_vector = tl.load(weights_ptr + weights_offsets, mask=weights_mask, other=0.0)
    
    # Calculate output offset
    output_offset = input_offset * n_embed + embed_offset
    
    # Store result with bounds checking
    output_mask = embed_offset < n_embed
    tl.store(output_ptr + output_offset, embed_vector, mask=output_mask)

@torch.fx.wrap
def optimized_embedding_only(input_ids, weights):
    # Get input shapes
    batch_size, seq_len = input_ids.shape
    embed_dim = weights.shape[1]
    vocab_size = weights.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, embed_dim), dtype=weights.dtype, device=weights.device)
    
    # Set up grid dimensions
    batch_blocks = (batch_size + 63) // 64
    seq_blocks = (seq_len + 63) // 64
    
    # Launch kernel with autotuning support
    embedding_only_kernel[(batch_blocks, seq_blocks, 1)](
        input_ids_ptr=input_ids,
        weights_ptr=weights,
        output_ptr=output,
        n_batch=batch_size,
        n_seq=seq_len,
        n_embed=embed_dim,
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        BLOCK_SIZE_BATCH=64,
        BLOCK_SIZE_SEQ=64,
        BLOCK_SIZE_EMBED=128,
    )
    
    return output

def replacement_func():
    return optimized_embedding_only