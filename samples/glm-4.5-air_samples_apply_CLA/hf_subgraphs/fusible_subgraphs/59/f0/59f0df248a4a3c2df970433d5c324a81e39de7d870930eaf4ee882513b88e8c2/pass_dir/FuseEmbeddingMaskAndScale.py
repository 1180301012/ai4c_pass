import torch
import triton
import triton.language as tl

def pattern(input_ids, weights):
    # Embedding lookup
    embedding = torch.nn.functional.embedding(input_ids, weights, 1, None, 2.0, False, False)
    
    # Create mask for padding token (assuming token 2 is padding)
    padding_mask = input_ids.__eq__(2)
    padding_mask = padding_mask.unsqueeze(-1)
    
    # Apply mask and scale
    masked_embedding = embedding.masked_fill(padding_mask, 0.0)
    scaled_embedding = masked_embedding * 0.88
    
    return scaled_embedding

def replacement_args(attention_mask, input_ids, weights):
    return (input_ids, weights)

@triton.jit
def fused_embedding_kernel(
    input_ids_ptr, 
    weights_ptr, 
    embedding_ptr,
    vocab_size,
    embedding_dim,
    seq_len,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused embedding lookup + masking + scaling kernel"""
    pid = tl.program_id(0)
    
    # Calculate global position
    row = pid // seq_len
    col = pid % seq_len
    
    if row >= batch_size or col >= seq_len:
        return
    
    # Load input_id
    input_id = tl.load(input_ids_ptr + row * seq_len + col)
    
    # Compute weight offset
    weight_offset = input_id * embedding_dim
    
    # Load embedding vector
    for k in range(0, embedding_dim, BLOCK_SIZE):
        offset = k + tl.arange(0, BLOCK_SIZE)
        if k + BLOCK_SIZE <= embedding_dim:
            # Load weights
            weights = tl.load(weights_ptr + weight_offset + offset)
            # Check if padding token (token 2)
            is_padding = input_id == 2
            # Apply zeroing and scaling
            result = tl.where(is_padding, 0.0, weights * 0.88)
            # Store result
            tl.store(embedding_ptr + (row * seq_len + col) * embedding_dim + offset, result)
        else:
            # Handle partial block
            mask = offset < embedding_dim
            weights = tl.load(weights_ptr + weight_offset + offset, mask=mask, other=0.0)
            is_padding = input_id == 2
            result = tl.where(is_padding, 0.0, weights * 0.88)
            tl.store(embedding_ptr + (row * seq_len + col) * embedding_dim + offset, result, mask=mask)

@torch.fx.wrap 
def fused_embedding_lookup(input_ids, weights):
    batch_size, seq_len = input_ids.shape
    vocab_size, embedding_dim = weights.shape
    
    # Create output tensor
    embedding = torch.empty(batch_size, seq_len, embedding_dim, dtype=weights.dtype, device=weights.device)
    
    # Set block size based on embedding dimension
    BLOCK_SIZE = 128
    total_elements = batch_size * seq_len
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_embedding_kernel[(num_programs,)](
        input_ids_ptr=input_ids,
        weights_ptr=weights,
        embedding_ptr=embedding,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return embedding

def replacement_func():
    return fused_embedding_lookup