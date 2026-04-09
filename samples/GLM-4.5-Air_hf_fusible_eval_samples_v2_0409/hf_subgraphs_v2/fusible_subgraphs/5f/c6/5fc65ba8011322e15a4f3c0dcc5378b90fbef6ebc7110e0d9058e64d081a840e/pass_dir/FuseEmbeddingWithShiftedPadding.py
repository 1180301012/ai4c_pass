import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern: Embedding lookup with shifted and padded versions concatenated
    """
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_embedding_shifted_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embed_dim,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that fuses embedding lookup with shifted padding operations.
    Creates three versions:
    1. Right-shifted with trailing zero
    2. Original embeddings
    3. Left-shifted with leading zero
    Then concatenates them along the embedding dimension.
    """
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Compute output offsets for the three concatenated parts
    # Each part has embed_dim features, so total embed_dim * 3
    base_offset = (batch_idx * seq_len * embed_dim * 3) + (seq_idx * embed_dim * 3)
    
    # Load the embedding for the current token
    token_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)
    if token_id < vocab_size and token_id >= 0:
        # Load embedding vector
        embed_offset = token_id * embed_dim
        embed_vec = tl.load(weight_ptr + embed_offset + tl.arange(0, embed_dim))
    else:
        # Handle out-of-vocab tokens with zeros
        embed_vec = tl.zeros(embed_dim, dtype=tl.float16)
    
    # Create right-shifted version: embed_vec[1:] with trailing zero
    if seq_idx < seq_len - 1:
        next_token_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx + 1)
        if next_token_id < vocab_size and next_token_id >= 0:
            next_embed_offset = next_token_id * embed_dim
            right_shifted = tl.load(weight_ptr + next_embed_offset + tl.arange(0, embed_dim))
        else:
            right_shifted = tl.zeros(embed_dim, dtype=tl.float16)
        # Insert zero at beginning
        right_shifted = tl.cat(tl.zeros(1, dtype=tl.float16), right_shifted) if embed_dim > 1 else tl.zeros(1, dtype=tl.float16)
    else:
        # Last position - all zeros
        right_shifted = tl.zeros(embed_dim, dtype=tl.float16)
    
    # Create left-shifted version: embed_vec[:-1] with leading zero
    if seq_idx > 0:
        prev_token_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx - 1)
        if prev_token_id < vocab_size and prev_token_id >= 0:
            prev_embed_offset = prev_token_id * embed_dim
            left_shifted = tl.load(weight_ptr + prev_embed_offset + tl.arange(0, embed_dim))
        else:
            left_shifted = tl.zeros(embed_dim, dtype=tl.float16)
        # Append zero at end
        left_shifted = tl.cat(left_shifted, tl.zeros(1, dtype=tl.float16)) if embed_dim > 1 else tl.zeros(1, dtype=tl.float16)
    else:
        # First position - all zeros
        left_shifted = tl.zeros(embed_dim, dtype=tl.float16)
    
    # Concatenate right_shifted, embed_vec, left_shifted
    if embed_dim > 1:
        combined = tl.cat(right_shifted, tl.cat(embed_vec, left_shifted))
    else:
        combined = tl.cat(tl.cat(right_shifted, embed_vec), left_shifted)
    
    # Store the result
    output_offsets = base_offset + tl.arange(0, embed_dim * 3)
    tl.store(output_ptr + output_offsets, combined)

@torch.fx.wrap
def fused_embedding_with_shifted_padding(input_ids, weight):
    """
    Optimized function that fuses embedding lookup with shifted padding operations.
    Returns concatenated right-shifted, original, and left-shifted embeddings.
    """
    batch_size, seq_len = input_ids.shape
    embed_dim = weight.shape[1]
    vocab_size = weight.shape[0]
    
    # Output shape: [batch_size, seq_len, embed_dim * 3]
    output_shape = (batch_size, seq_len, embed_dim * 3)
    output = torch.empty(output_shape, dtype=weight.dtype, device=weight.device)
    
    # Block size for optimal GPU occupancy
    BLOCK_SIZE = 256
    
    # Calculate number of programs needed
    total_elements = batch_size * seq_len
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_embedding_shifted_kernel[(num_programs,)](
        input_ids_ptr=input_ids,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_embedding_with_shifted_padding