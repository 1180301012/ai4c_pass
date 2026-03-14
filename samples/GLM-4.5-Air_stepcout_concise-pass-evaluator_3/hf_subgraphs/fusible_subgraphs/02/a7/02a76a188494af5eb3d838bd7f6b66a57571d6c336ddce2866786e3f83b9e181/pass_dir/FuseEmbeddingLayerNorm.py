import torch
import triton
import triton.language as tl

def pattern(input_ids, norm_weight, embedding_weight):
    # Embedding lookup
    embedded = torch.nn.functional.embedding(input_ids, embedding_weight, 50283, None, 2.0, False, False)
    
    # Layer normalization 
    normalized = torch.nn.functional.layer_norm(embedded, (768,), norm_weight, None, 1e-05)
    
    # Dropout with p=0.0 (no-op)
    dropout_out = torch.nn.functional.dropout(normalized, 0.0, False, False)
    
    return dropout_out

def replacement_args(input_ids, norm_weight, embedding_weight):
    return (input_ids, norm_weight, embedding_weight)

@triton.jit
def optimized_embedding_kernel(
    input_ids_ptr,
    norm_weight_ptr,
    embedding_weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embedding_vocab_size,
    embedding_hidden_dim,
    norm_hidden_dim,
    BLOCK_HIDDEN: tl.constexpr,
):
    # Program IDs for 2D grid (batch x sequence)
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate linear index for this token
    token_idx = batch_idx * seq_len + seq_idx
    
    # Load input token ID (scalar)
    token_id = tl.load(input_ids_ptr + token_idx)
    
    # Calculate embedding offset
    embed_start = token_id * embedding_hidden_dim
    
    # Vectorized load for entire hidden dimension
    hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
    
    # Load embedding vector for this token
    embed_ptrs = embedding_weight_ptr + embed_start + hidden_offsets
    embed_vector = tl.load(embed_ptrs, mask=hidden_offsets < embedding_hidden_dim, other=0.0)
    
    # Compute mean for layer normalization
    mean = tl.sum(embed_vector) / embedding_hidden_dim
    
    # Compute variance for layer normalization  
    variance = tl.sum((embed_vector - mean) * (embed_vector - mean)) / embedding_hidden_dim
    
    # Compute normalized output
    epsilon = 1e-05
    normalized = (embed_vector - mean) * tl.rsqrt(variance + epsilon)
    
    # Load normalization weights and apply
    norm_ptrs = norm_weight_ptr + hidden_offsets  
    norm_weight = tl.load(norm_ptrs, mask=hidden_offsets < norm_hidden_dim, other=0.0)
    
    # Apply layer normalization scaling
    out_vector = normalized * norm_weight
    
    # Store final result (dropout with p=0.0 is a no-op)
    out_ptrs = output_ptr + token_idx * embedding_hidden_dim + hidden_offsets
    tl.store(out_ptrs, out_vector, mask=hidden_offsets < embedding_hidden_dim)

@torch.fx.wrap
def optimized_embedding_layer_norm(input_ids, norm_weight, embedding_weight):
    """Optimized embedding + layer normalization kernel"""
    batch_size, seq_len = input_ids.shape
    vocab_size, hidden_dim = embedding_weight.shape
    
    output = torch.empty((batch_size, seq_len, hidden_dim), 
                        dtype=torch.float32, 
                        device=input_ids.device)
    
    # Reshape inputs to be flat for easier kernel access
    flat_input_ids = input_ids.contiguous().view(-1)
    flat_output = output.contiguous().view(-1, hidden_dim)
    
    # Calculate grid dimensions
    grid_x = batch_size
    grid_y = seq_len
    
    # Launch fused kernel with 2D grid
    optimized_embedding_kernel[(grid_x, grid_y)](
        input_ids_ptr=flat_input_ids,
        norm_weight_ptr=norm_weight.contiguous(),
        embedding_weight_ptr=embedding_weight.contiguous(), 
        output_ptr=flat_output,
        batch_size=batch_size,
        seq_len=seq_len,
        embedding_vocab_size=vocab_size,
        embedding_hidden_dim=hidden_dim,
        norm_hidden_dim=norm_weight.shape[0],
        BLOCK_HIDDEN=256,
    )
    
    return output

def replacement_func():
    return optimized_embedding_layer_norm