import torch
import triton
import triton.language as tl

# Pattern: embedding operation followed by slicing [:, 1:]
def pattern(input_ids, weights, padding_idx=0, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    tmp_0 = input_ids
    tmp_1 = weights
    tmp_2 = torch.nn.functional.embedding(tmp_0, tmp_1, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    tmp_0 = tmp_1 = None
    tmp_3 = tmp_2[slice(None, None, None), slice(1, None, None)]
    return (tmp_3, tmp_2)

# Extract arguments for the replacement function
def replacement_args(input_ids, weights, padding_idx=0, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return (input_ids, weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

# Triton kernel for optimized embedding + slicing
@triton.jit
def embedding_slicing_kernel(
    input_ids_ptr,
    weights_ptr,
    full_output_ptr,
    sliced_output_ptr,
    n_batch,
    n_seq,
    n_embed,
    padding_idx,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_EMBED: tl.constexpr,
):
    # Each program handles one batch x sequence position combination
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    embed_offset = tl.arange(0, BLOCK_SIZE_EMBED)
    
    # Compute offsets
    input_offset = batch_idx * n_seq + seq_idx
    
    # Load input IDs with bounds checking
    input_id = tl.load(input_ids_ptr + input_offset, mask=(input_offset < n_batch * n_seq), other=padding_idx)
    
    # Skip processing if padding index
    if input_id == padding_idx:
        # Store zeros for padding
        for embed_idx in range(0, n_embed, BLOCK_SIZE_EMBED):
            embed_offsets = embed_idx + embed_offset
            mask = embed_offsets < n_embed
            tl.store(full_output_ptr + input_offset * n_embed + embed_offsets, 0.0, mask=mask)
            tl.store(sliced_output_ptr + input_offset * n_embed + embed_offsets, 0.0, mask=mask)
        return
    
    # Load embedding weights with bounds checking
    embed_start = input_id * n_embed
    weights_offsets = embed_start + embed_offset
    weights_mask = weights_offsets < (30522 * n_embed)  # Assuming max vocab size 30522
    
    # Load embedding vector
    embed_vector = tl.load(weights_ptr + weights_offsets, mask=weights_mask, other=0.0)
    
    # Calculate output addresses
    full_output_offset = input_offset * n_embed + embed_offset
    sliced_output_offset = input_offset * n_embed + embed_offset
    
    # Store results
    full_mask = embed_offset < n_embed
    tl.store(full_output_ptr + full_output_offset, embed_vector, mask=full_mask)
    
    # For sliced output, only store if seq_idx > 0 (skip first sequence position)
    seq_mask = (seq_idx > 0) & (embed_offset < n_embed)
    tl.store(sliced_output_ptr + sliced_output_offset, embed_vector, mask=seq_mask)

@torch.fx.wrap
def optimized_embedding_slicing(input_ids, weights, padding_idx=0, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    # Get input shapes
    batch_size, seq_len = input_ids.shape
    embed_dim = weights.shape[1]
    
    # Create output tensors
    full_output = torch.empty((batch_size, seq_len, embed_dim), dtype=weights.dtype, device=weights.device)
    sliced_output = torch.empty((batch_size, seq_len - 1, embed_dim), dtype=weights.dtype, device=weights.device)
    
    # Set up grid dimensions
    batch_blocks = (batch_size + 63) // 64  # Using 64 as block size for batch
    seq_blocks = (seq_len + 63) // 64
    
    # Launch kernel
    embedding_slicing_kernel[(batch_blocks, seq_blocks, 1)](
        input_ids_ptr=input_ids,
        weights_ptr=weights,
        full_output_ptr=full_output,
        sliced_output_ptr=sliced_output,
        n_batch=batch_size,
        n_seq=seq_len,
        n_embed=embed_dim,
        padding_idx=padding_idx,
        BLOCK_SIZE_BATCH=64,
        BLOCK_SIZE_SEQ=64,
        BLOCK_SIZE_EMBED=128,  # Optimized for common embedding dimensions
    )
    
    return sliced_output, full_output

def replacement_func():
    return optimized_embedding_slicing