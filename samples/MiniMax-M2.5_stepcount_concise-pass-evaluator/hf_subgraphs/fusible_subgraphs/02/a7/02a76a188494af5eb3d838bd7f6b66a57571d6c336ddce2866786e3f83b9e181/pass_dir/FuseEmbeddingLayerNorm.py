import torch
import triton
import triton.language as tl


# Pattern matching function - matches embedding followed by layer_norm
def pattern(input_ids, embedding_weight, layer_norm_weight):
    # Embedding lookup
    embeddings = torch.nn.functional.embedding(
        input_ids, embedding_weight, 50283, None, 2.0, False, False
    )
    # Layer normalization
    normalized = torch.nn.functional.layer_norm(
        embeddings, (768,), layer_norm_weight, None, 1e-05
    )
    # Dropout with p=0.0 is a no-op, return normalized output
    return normalized


# Extract arguments needed for the replacement
def replacement_args(input_ids, embedding_weight, layer_norm_weight):
    return (input_ids, embedding_weight, layer_norm_weight)


# Optimized Triton kernel that fuses embedding lookup + layer norm
# Each program processes BLOCK_TOKENS tokens for better parallelism
@triton.jit
def embedding_layernorm_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    layer_norm_weight_ptr,
    output_ptr,
    # Tensor dimensions
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    # Block sizes
    BLOCK_H: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """
    Fused kernel for embedding lookup + layer normalization.
    Each program processes BLOCK_TOKENS tokens for better GPU utilization.
    """
    # Get program IDs
    pid = tl.program_id(0)
    
    # Calculate token range for this program
    token_start = pid * BLOCK_TOKENS
    if token_start >= num_tokens:
        return
    
    token_end = min(token_start + BLOCK_TOKENS, num_tokens)
    
    # Process each token in this block
    for local_idx in range(token_end - token_start):
        token_idx = token_start + local_idx
        
        # Load token ID
        token_id = tl.load(input_ids_ptr + token_idx)
        
        # Clamp to valid vocab range
        token_id = tl.minimum(token_id, vocab_size - 1)
        token_id = tl.maximum(token_id, 0)
        
        # Compute mean and variance
        sum_vals = 0.0
        sum_sq = 0.0
        
        for h in range(0, hidden_size, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            mask_h = h_offsets < hidden_size
            
            # Embedding lookup
            emb_offset = token_id * hidden_size + h_offsets
            emb_val = tl.load(embedding_weight_ptr + emb_offset, mask=mask_h, other=0.0)
            
            sum_vals += tl.sum(emb_val, axis=0)
            sum_sq += tl.sum(emb_val * emb_val, axis=0)
        
        # Compute mean and variance
        mean = sum_vals / hidden_size
        variance = (sum_sq / hidden_size) - (mean * mean)
        variance = variance + 1e-05  # eps
        inv_std = 1.0 / tl.sqrt(variance)
        
        # Normalize and store output
        for h in range(0, hidden_size, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            mask_h = h_offsets < hidden_size
            
            # Load layer norm weight
            ln_weight = tl.load(layer_norm_weight_ptr + h_offsets, mask=mask_h, other=0.0)
            
            # Load embedding
            emb_offset = token_id * hidden_size + h_offsets
            emb_val = tl.load(embedding_weight_ptr + emb_offset, mask=mask_h, other=0.0)
            
            # Layer normalization
            normalized = (emb_val - mean) * inv_std * ln_weight
            
            # Store output
            output_off = token_idx * hidden_size + h_offsets
            tl.store(output_ptr + output_off, normalized, mask=mask_h)


@torch.fx.wrap
def fused_embedding_layernorm(input_ids, embedding_weight, layer_norm_weight):
    """
    Wrapper function that launches the fused kernel.
    
    Args:
        input_ids: [batch_size, seq_len] - token IDs
        embedding_weight: [vocab_size, hidden_size] - embedding table
        layer_norm_weight: [hidden_size] - layer norm weight
    
    Returns:
        output: [batch_size, seq_len, hidden_size] - normalized embeddings
    """
    batch_size, seq_len = input_ids.shape
    vocab_size, hidden_size = embedding_weight.shape
    num_tokens = batch_size * seq_len
    
    # Output tensor
    output = torch.empty((batch_size, seq_len, hidden_size), 
                         device=input_ids.device, 
                         dtype=embedding_weight.dtype)
    
    # Block sizes - power of 2 for tl.arange
    BLOCK_H = 128  # for hidden dimension
    BLOCK_TOKENS = 64  # tokens per program
    
    # Grid: need enough programs to cover all tokens
    num_programs = (num_tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    
    embedding_layernorm_kernel[num_programs](
        input_ids,
        embedding_weight,
        layer_norm_weight,
        output,
        num_tokens,
        hidden_size,
        vocab_size,
        BLOCK_H,
        BLOCK_TOKENS,
    )
    
    return output


def replacement_func():
    return fused_embedding_layernorm