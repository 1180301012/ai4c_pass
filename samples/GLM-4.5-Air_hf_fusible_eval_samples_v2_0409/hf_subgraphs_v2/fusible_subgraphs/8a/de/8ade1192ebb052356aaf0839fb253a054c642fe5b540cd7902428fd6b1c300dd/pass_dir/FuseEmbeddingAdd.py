import torch
import triton
import triton.language as tl

def pattern(cache_positions, embedding_weights, input_embeddings):
    """
    Pattern: Embedding lookup + Addition fusion
    Matches: cache_positions -> unsqueeze(0) + 2 -> embedding lookup -> addition with input_embeddings
    """
    cache_pos_unsqueeze = cache_positions.unsqueeze(0)
    cache_pos_plus_2 = cache_pos_unsqueeze + 2
    embedding_result = torch.nn.functional.embedding(cache_pos_plus_2, embedding_weights, None, None, 2.0, False, False)
    # Note: The original has .to(device(type='cuda', index=0)) which is redundant since everything is on GPU
    # The embeddings are directly added to input_embeddings
    result = input_embeddings + embedding_result
    return result, embedding_result

def replacement_args(cache_positions, embedding_weights, input_embeddings):
    return (cache_positions, embedding_weights, input_embeddings)

@triton.jit
def embedding_add_kernel(
    cache_ptr,           # [cache_len]
    weight_ptr,         # [vocab_size, hidden_dim]
    input_ptr,          # [batch_size, seq_len, hidden_dim]
    output_ptr,         # [batch_size, seq_len, hidden_dim]
    embedding_ptr,      # [batch_size, seq_len, hidden_dim]
    batch_size,
    seq_len,
    hidden_dim,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sequence position in the batch
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Load cache position
    cache_pos = tl.load(cache_ptr + seq_idx).to(tl.int32)
    adjusted_pos = cache_pos + 2
    
    # Boundary check
    if adjusted_pos >= vocab_size:
        adjusted_pos = vocab_size - 1
    
    # Calculate offset for this batch and sequence position
    output_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim
    input_offset = output_offset
    
    # Load input tensor for this position
    input_vals = tl.load(input_ptr + input_offset, mask=input_offset < batch_size * seq_len * hidden_dim)
    
    # Load embedding for this position
    weight_offset = adjusted_pos * hidden_dim
    embedding_vals = tl.load(weight_ptr + weight_offset, mask=weight_offset < vocab_size * hidden_dim)
    
    # Add input and embedding
    result_vals = input_vals + embedding_vals
    
    # Store results
    tl.store(output_ptr + output_offset, result_vals, mask=output_offset < batch_size * seq_len * hidden_dim)
    tl.store(embedding_ptr + output_offset, embedding_vals, mask=output_offset < batch_size * seq_len * hidden_dim)

@torch.fx.wrap
def fused_embedding_add(cache_positions, embedding_weights, input_embeddings):
    batch_size, seq_len, hidden_dim = input_embeddings.shape
    vocab_size = embedding_weights.shape[0]
    
    # Allocate output tensors
    output = torch.empty_like(input_embeddings)
    embedding_result = torch.empty_like(input_embeddings)
    
    # Calculate grid size
    BLOCK_SIZE = 256
    grid = (
        batch_size,
        seq_len,
    )
    
    # Launch kernel
    embedding_add_kernel[grid](
        cache_ptr=cache_positions,
        weight_ptr=embedding_weights,
        input_ptr=input_embeddings,
        output_ptr=output,
        embedding_ptr=embedding_result,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, embedding_result

def replacement_func():
    return fused_embedding_add