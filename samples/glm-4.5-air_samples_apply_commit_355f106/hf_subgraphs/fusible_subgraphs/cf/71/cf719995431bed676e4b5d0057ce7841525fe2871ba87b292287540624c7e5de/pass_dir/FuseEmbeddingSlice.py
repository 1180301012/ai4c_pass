import torch
import triton
import triton.language as tl


def pattern(input_ids, embedding_weight):
    """
    Pattern: torch.nn.functional.embedding + slice from index 1
    This optimization avoids computing and then discarding the embedding for index 0
    """
    embedding = torch.nn.functional.embedding(input_ids, embedding_weight, 0, None, 2.0, False, False)
    sliced_embedding = embedding[slice(None, None, None), slice(1, None, None)]
    return sliced_embedding, embedding


def replacement_args(input_ids, embedding_weight):
    """
    Extract arguments needed for the replacement function
    """
    return input_ids, embedding_weight


@triton.jit
def embedding_slice_kernel(
    input_ids_ptr,
    weight_ptr,
    out_sliced_ptr,
    out_full_ptr,
    num_embeddings,
    embedding_dim,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that computes embeddings and slices off the first sequence element.
    Only computes embeddings for positions 1+ to avoid wasted computation.
    """
    # Program identifiers for 2D grid (batch, sequence)
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Check bounds
    batch_mask = batch_idx < batch_size
    seq_mask = seq_idx < seq_len - 1  # Only process positions 1+
    
    if batch_mask and seq_mask:
        # Calculate input IDs pointer offset (original sequence position = seq_idx + 1)
        input_offset = batch_idx * seq_len + (seq_idx + 1)
        input_id = tl.load(input_ids_ptr + input_offset, mask=batch_mask)
        
        # Load embedding weights for this token
        # Each program handles a block of the embedding vector
        embed_offsets = tl.arange(0, BLOCK_SIZE)
        embed_mask = embed_offsets < embedding_dim
        
        # Compute weight offset: input_id * embedding_dim + embed_offset
        weight_offset = input_id * embedding_dim + embed_offsets
        
        # Load embedding vector values
        embed_values = tl.load(
            weight_ptr + weight_offset, 
            mask=(input_id < num_embeddings) & embed_mask,
            other=0.0
        )
        
        # Store to sliced output (final result after slice)
        # Output shape: [batch_size, seq_len-1, embedding_dim]
        sliced_out_offset = (batch_idx * (seq_len - 1) + seq_idx) * embedding_dim + embed_offsets
        tl.store(out_sliced_ptr + sliced_out_offset, embed_values, mask=embed_mask)
        
        # Also store to full output for compatibility (return same structure as original)
        # Original shape: [batch_size, seq_len, embedding_dim]
        full_out_offset = (batch_idx * seq_len + (seq_idx + 1)) * embedding_dim + embed_offsets
        tl.store(out_full_ptr + full_out_offset, embed_values, mask=embed_mask)


@torch.fx.wrap
def optimized_embedding_slice(input_ids, embedding_weight):
    """
    Optimized function that fuses embedding lookup with slice operation.
    Only computes embeddings for sequence positions 1+ to avoid wasted computation.
    """
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = embedding_weight.shape
    
    # Handle edge cases
    if seq_len <= 1:
        # Return empty sliced output and zero full output
        sliced_output = torch.empty((batch_size, 0, embedding_dim), 
                                   device=input_ids.device, dtype=embedding_weight.dtype)
        full_output = torch.zeros((batch_size, seq_len, embedding_dim), 
                                device=input_ids.device, dtype=embedding_weight.dtype)
        return sliced_output, full_output
    
    # Create output tensors
    sliced_output = torch.empty((batch_size, seq_len - 1, embedding_dim), 
                               device=input_ids.device, dtype=embedding_weight.dtype)
    full_output = torch.empty((batch_size, seq_len, embedding_dim), 
                            device=input_ids.device, dtype=embedding_weight.dtype)
    
    # Launch kernel for positions 1+
    grid = (batch_size, seq_len - 1)  # Only compute for positions 1+
    block_size = min(128, embedding_dim)  # Use appropriate block size
    
    embedding_slice_kernel[grid](
        input_ids_ptr=input_ids,
        weight_ptr=embedding_weight,
        out_sliced_ptr=sliced_output,
        out_full_ptr=full_output,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=block_size
    )
    
    # Compute and store position 0 embedding separately for full output compatibility
    if batch_size > 0:
        pos_0_embeddings = torch.nn.functional.embedding(
            input_ids[:, :1], embedding_weight, 0, None, 2.0, False, False
        )
        full_output[:, :1, :] = pos_0_embeddings
    
    return sliced_output, full_output


def replacement_func():
    """
    Return the optimized function that fuses embedding + slice
    """
    return optimized_embedding_slice