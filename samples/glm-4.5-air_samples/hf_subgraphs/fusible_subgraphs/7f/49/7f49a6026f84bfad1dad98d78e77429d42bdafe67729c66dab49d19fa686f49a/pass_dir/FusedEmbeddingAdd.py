import torch
import triton
import triton.language as tl

def pattern(input_ids, position_ids, word_embeddings, position_embeddings, attention_mask):
    # Pattern matches the embedding + addition section of the computation
    # This follows the exact structure from the reference pattern example
    embedding1 = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    embedding2 = torch.nn.functional.embedding(position_ids, position_embeddings, 1, None, 2.0, False, False)
    result = embedding1 + embedding2
    return result

def replacement_args(input_ids, position_ids, word_embeddings, position_embeddings, attention_mask):
    return (input_ids, position_ids, word_embeddings, position_embeddings)

@triton.jit
def fused_embedding_add_kernel(
    input_ids_ptr,
    position_ids_ptr,
    word_embeddings_ptr,
    position_embeddings_ptr,
    output_ptr,
    num_sequences,
    seq_len,
    vocab_size,
    num_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles sequences in chunks
    seq_idx = tl.program_id(0)
    token_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = token_idx < seq_len * num_sequences
    
    # Reshape indices for batch processing
    flat_token_idx = token_idx
    seq_offsets = tl.arange(0, num_sequences) * seq_len
    token_seq_idx = flat_token_idx // seq_len  # Get sequence index for each token
    token_pos_idx = flat_token_idx % seq_len   # Get position index for each token
    
    # Load input IDs and position IDs
    input_id = tl.load(input_ids_ptr + flat_token_idx, mask=mask, other=0)
    position_id = tl.load(position_ids_ptr + flat_token_idx, mask=mask, other=0)
    
    # Clamp indices to valid range for embeddings
    input_id = tl.minimum(input_id, vocab_size - 1)
    position_id = tl.minimum(position_id, seq_len - 1)
    
    # Calculate embedding indices
    word_emb_idx = input_id * num_features
    pos_emb_idx = position_id * num_features
    
    # Load embeddings with masking
    word_embeddings = tl.load(
        word_embeddings_ptr + word_emb_idx + tl.arange(0, num_features),
        mask=tl.broadcast_to(mask[:, None], (mask.shape[0], num_features)),
        other=0.0
    )
    
    position_embeddings = tl.load(
        position_embeddings_ptr + pos_emb_idx + tl.arange(0, num_features),
        mask=tl.broadcast_to(mask[:, None], (mask.shape[0], num_features)),
        other=0.0
    )
    
    # Add embeddings and store result
    result = word_embeddings + position_embeddings
    tl.store(output_ptr + flat_token_idx * num_features, result, mask=tl.broadcast_to(mask[:, None], (mask.shape[0], num_features)))

@torch.fx.wrap
def fused_embedding_add(input_ids, position_ids, word_embeddings, position_embeddings, attention_mask):
    seq_len = input_ids.shape[1]
    num_sequences = input_ids.shape[0]
    num_features = word_embeddings.shape[1]
    vocab_size = word_embeddings.shape[0]
    
    # Output should be [num_sequences, seq_len, num_features]
    output_size = (num_sequences, seq_len, num_features)
    output = torch.empty(output_size, dtype=torch.float32, device=input_ids.device)
    
    # Flatten input tokens for processing
    flat_input_ids = input_ids.view(-1)
    flat_position_ids = position_ids.view(-1)
    flat_output = output.view(-1, num_features)
    
    BLOCK_SIZE = 512  # Can be tuned
    grid = (
        num_sequences,
        (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    
    fused_embedding_add_kernel[grid](
        flat_input_ids,
        flat_position_ids,
        word_embeddings,
        position_embeddings,
        flat_output,
        num_sequences,
        seq_len,
        vocab_size,
        num_features,
        BLOCK_SIZE,
    )
    
    # Return the value that needs to be observable in the original computation
    return output

def replacement_func():
    return fused_embedding_add