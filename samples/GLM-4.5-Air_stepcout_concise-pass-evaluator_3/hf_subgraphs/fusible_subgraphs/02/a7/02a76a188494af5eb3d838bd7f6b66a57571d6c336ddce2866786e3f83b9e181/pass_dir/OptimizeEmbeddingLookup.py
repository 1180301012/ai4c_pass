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
def optimized_embedding_only_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embedding_vocab_size,
    embedding_hidden_dim,
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
    
    # Load embedding vector for this token with efficient memory access
    embed_ptrs = embedding_weight_ptr + embed_start + hidden_offsets
    embed_vector = tl.load(embed_ptrs, mask=hidden_offsets < embedding_hidden_dim, other=0.0)
    
    # Store embedding result (keeping layer norm and dropout separate for correctness)
    out_ptrs = output_ptr + token_idx * embedding_hidden_dim + hidden_offsets
    tl.store(out_ptrs, embed_vector, mask=hidden_offsets < embedding_hidden_dim)

@triton.jit
def fast_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_HIDDEN: tl.constexpr,
):
    # Program IDs for 2D grid (batch x sequence)
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate linear index for this token
    token_idx = batch_idx * seq_len + seq_idx
    token_offset = token_idx * hidden_dim
    
    # Load input for layer normalization
    hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
    input_ptrs = input_ptr + token_offset + hidden_offsets
    input_vector = tl.load(input_ptrs, mask=hidden_offsets < hidden_dim, other=0.0)
    
    # Load weight
    weight_ptrs = weight_ptr + hidden_offsets
    weight_vector = tl.load(weight_ptrs, mask=hidden_offsets < hidden_dim, other=0.0)
    
    # Optimized layer normalization
    mean = tl.sum(input_vector) / hidden_dim
    variance = tl.sum((input_vector - mean) * (input_vector - mean)) / hidden_dim
    epsilon = 1e-05
    normalized = (input_vector - mean) * tl.rsqrt(variance + epsilon)
    out_vector = normalized * weight_vector
    
    # Store output
    out_ptrs = output_ptr + token_offset + hidden_offsets
    tl.store(out_ptrs, out_vector, mask=hidden_offsets < hidden_dim)

@torch.fx.wrap
def optimized_embedding_lookup(input_ids, norm_weight, embedding_weight):
    """Optimized embedding lookup with separate layer norm for correctness"""
    batch_size, seq_len = input_ids.shape
    vocab_size, hidden_dim = embedding_weight.shape
    
    # Temporary storage for embedding
    embedded = torch.empty((batch_size, seq_len, hidden_dim), 
                          dtype=torch.float32, 
                          device=input_ids.device)
    
    final_output = torch.empty((batch_size, seq_len, hidden_dim), 
                              dtype=torch.float32, 
                              device=input_ids.device)
    
    # Reshape inputs for kernel access
    flat_input_ids = input_ids.contiguous().view(-1)
    flat_embedded = embedded.contiguous().view(-1, hidden_dim)
    flat_output = final_output.contiguous().view(-1, hidden_dim)
    
    # Grid dimensions
    grid_x = batch_size
    grid_y = seq_len
    
    # Launch optimized embedding kernel
    optimized_embedding_only_kernel[(grid_x, grid_y)](
        input_ids_ptr=flat_input_ids,
        embedding_weight_ptr=embedding_weight.contiguous(), 
        output_ptr=flat_embedded,
        batch_size=batch_size,
        seq_len=seq_len,
        embedding_vocab_size=vocab_size,
        embedding_hidden_dim=hidden_dim,
        BLOCK_HIDDEN=256,
    )
    
    # Launch optimized layer norm kernel
    fast_layer_norm_kernel[(grid_x, grid_y)](
        input_ptr=flat_embedded,
        weight_ptr=norm_weight.contiguous(),
        output_ptr=flat_output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_HIDDEN=256,
    )
    
    # Dropout with p=0.0 is a no-op, so we just return the layer norm result
    return final_output

def replacement_func():
    return optimized_embedding_lookup