import torch
import triton
import triton.language as tl
from torch import device

def pattern(input_ids, layer_norm_weight, layer_norm_bias, pos_embeddings, word_embeddings):
    tmp_10 = torch.nn.functional.embedding(input_ids, word_embeddings, 1, None, 2.0, False, False)
    tmp_11 = torch.ones((1, 15), dtype = torch.int64, device = device(type='cuda', index=0))
    tmp_12 = torch.cumsum(tmp_11, dim = 1)
    tmp_13 = tmp_12 - tmp_11
    tmp_13 += 2
    tmp_14 = tmp_13
    tmp_15 = torch.nn.functional.embedding(tmp_14, pos_embeddings, 1, None, 2.0, False, False)
    tmp_16 = tmp_10 + tmp_15
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), layer_norm_weight, layer_norm_bias, 1e-05)
    return tmp_17

def replacement_args(input_ids, layer_norm_weight, layer_norm_bias, pos_embeddings, word_embeddings):
    return (input_ids, layer_norm_weight, layer_norm_bias, pos_embeddings, word_embeddings)

@triton.jit
def optimized_embedding_kernel(
    input_ids_ptr,
    word_embeddings_ptr,
    pos_embeddings_ptr,
    output_ptr,
    n_elements,
    vocab_size,
    embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of the output tensor
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Get input IDs - each position has one ID
    seq_len = n_elements // embed_dim
    pos = offsets // embed_dim  # Position in sequence
    elem_idx = offsets % embed_dim  # Element within embedding
    
    # Load input IDs (batch_size=1, so pos directly indexes)
    input_ids = tl.load(input_ids_ptr + pos, mask=offsets < seq_len, other=0)
    
    # Load word embedding (direct lookup)
    word_offset = input_ids * embed_dim + elem_idx
    word_embedding = tl.load(word_embeddings_ptr + word_offset, mask=mask, other=0.0)
    
    # Calculate position embedding (2-based indexing)
    pos_id = tl.load(pos_embeddings_ptr + pos * embed_dim + elem_idx, mask=mask, other=0.0)
    
    # Add embeddings
    combined = word_embedding + pos_id
    
    # Store intermediate result for normalization
    tl.store(output_ptr + offsets, combined, mask=mask)

@triton.jit
def optimized_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    embed_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and normalize along feature dimension
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + (offsets % embed_dim), mask=offsets < embed_dim, other=1.0)
    bias = tl.load(bias_ptr + (offsets % embed_dim), mask=offsets < embed_dim, other=0.0)
    
    # Layer norm computation
    mean = tl.sum(input_vals, axis=0) / embed_dim
    var = tl.sum((input_vals - mean) * (input_vals - mean), axis=0) / embed_dim
    normalized = (input_vals - mean) / tl.sqrt(var + eps)
    
    # Apply weight and bias
    result = normalized * weight + bias
    
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_embeddings_and_layer_norm(input_ids, layer_norm_weight, layer_norm_bias, pos_embeddings, word_embeddings):
    batch_size, seq_len = input_ids.shape
    embed_dim = layer_norm_weight.shape[0]
    n_elements = batch_size * seq_len * embed_dim
    
    # Create combined embeddings
    combined = torch.empty_like(word_embeddings[:batch_size*seq_len])
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_embedding_kernel[(num_programs,)](
        input_ids_ptr=input_ids,
        word_embeddings_ptr=word_embeddings,
        pos_embeddings_ptr=pos_embeddings,
        output_ptr=combined,
        n_elements=n_elements,
        vocab_size=word_embeddings.shape[0],
        embed_dim=embed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply layer norm
    output = torch.empty_like(combined)
    optimized_layer_norm_kernel[(num_programs,)](
        input_ptr=combined,
        weight_ptr=layer_norm_weight,
        bias_ptr=layer_norm_bias,
        output_ptr=output,
        n_elements=n_elements,
        embed_dim=embed_dim,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to proper output shape [batch_size, seq_len, embed_dim]
    output = output.reshape(batch_size, seq_len, embed_dim)
    
    return output

def replacement_func():
    return optimized_embeddings_and_layer_norm