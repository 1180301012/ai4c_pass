import torch
import triton
import triton.language as tl

def pattern(embedding_1, embedding_2, embedding_3):
    """
    Pattern for adding multiple embedding tensors
    Example: tmp_42 = tmp_16 + tmp_17 + tmp_19 + ... (all embeddings combined)
    """
    return embedding_1 + embedding_2 + embedding_3

def replacement_args(embedding_1, embedding_2, embedding_3):
    return (embedding_1, embedding_2, embedding_3)

@triton.jit
def optimized_addition_kernel(
    emb1_ptr, emb2_ptr, emb3_ptr,
    output_ptr,
    batch_size, seq_len, hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * seq_len)
    
    if not mask.any():
        return
    
    # Load all three embeddings
    emb1 = tl.load(emb1_ptr + offsets * hidden_dim, mask=mask, other=0.0)
    emb2 = tl.load(emb2_ptr + offsets * hidden_dim, mask=mask, other=0.0)
    emb3 = tl.load(emb3_ptr + offsets * hidden_dim, mask=mask, other=0.0)
    
    # Sum all embeddings
    result = emb1 + emb2 + emb3
    
    # Store result
    tl.store(output_ptr + offsets * hidden_dim, result, mask=mask)

@torch.fx.wrap
def optimized_embedding_addition(embedding_1, embedding_2, embedding_3):
    """
    Optimized function for adding three embedding tensors
    Input: three embedding tensors of same shape (batch_size, seq_len, hidden_dim)
    """
    # Validate all embeddings have same shape
    shape = embedding_1.shape
    if embedding_2.shape != shape:
        raise ValueError(f"All embeddings must have same shape, got {shape} vs {embedding_2.shape}")
    if embedding_3.shape != shape:
        raise ValueError(f"All embeddings must have same shape, got {shape} vs {embedding_3.shape}")
    
    batch_size, seq_len, hidden_dim = shape
    
    output = torch.empty_like(embedding_1)
    
    BLOCK_SIZE = 1024
    num_programs = (batch_size * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_addition_kernel[(num_programs,)](
        embedding_1, embedding_2, embedding_3,
        output,
        batch_size, seq_len, hidden_dim,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_embedding_addition