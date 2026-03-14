import torch
import triton
import triton.language as tl


def pattern(input_ids, norm_weight, embedding_weights):
    # Matches the embedding + layer_norm sequence but focuses on optimizing embedding access
    embeddings = torch.nn.functional.embedding(input_ids, embedding_weights, 50283, None, 2.0, False, False)
    normalized = torch.nn.functional.layer_norm(embeddings, (embedding_weights.shape[1],), norm_weight, None, 1e-05)
    return embeddings, normalized


def replacement_args(input_ids, norm_weight, embedding_weights):
    return (input_ids, norm_weight, embedding_weights)


@triton.jit
def optimized_embedding_kernel(
    input_ids_ptr,
    embedding_weights_ptr, 
    output_ptr,
    batch_size,
    seq_len,
    embedding_dim,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized embedding lookup with better memory access patterns"""
    pid = tl.program_id(0)
    
    # Each program handles one token position across all batch elements
    seq_idx = pid % seq_len
    batch_idx = pid // seq_len
    
    if batch_idx >= batch_size:
        return  # Out of bounds for this program
        
    # Load input token ID
    token_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx, mask=batch_idx < batch_size, other=0)
    
    if token_id >= vocab_size:
        token_id = 0  # Handle out-of-vocabulary tokens
        
    # Calculate embedding offset
    emb_offset = token_id * embedding_dim
    
    # Load embedding vector efficiently
    emb_mask = tl.arange(0, embedding_dim) < embedding_dim
    embedding = tl.load(
        embedding_weights_ptr + emb_offset + tl.arange(0, embedding_dim), 
        mask=emb_mask, 
        other=0.0
    )
    
    # Store result
    output_offset = (batch_idx * seq_len + seq_idx) * embedding_dim
    tl.store(
        output_ptr + output_offset + tl.arange(0, embedding_dim), 
        embedding, 
        mask=emb_mask
    )


@torch.fx.wrap
def optimize_embedding_lookup(input_ids, norm_weight, embedding_weights):
    """Optimized embedding lookup with efficient memory access"""
    if input_ids.is_cuda:
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        embedding_dim = embedding_weights.shape[1]
        vocab_size = embedding_weights.shape[0]
        
        # Total number of tokens to process
        total_tokens = batch_size * seq_len
        block_size = 256  # Optimal for token-wise parallelism
        n_programs = (total_tokens + block_size - 1) // block_size
        
        # Create output tensor
        output = torch.empty(batch_size, seq_len, embedding_dim, dtype=torch.float32, device='cuda')
        
        # Launch optimized kernel
        optimized_embedding_kernel[(n_programs,)](
            input_ids_ptr=input_ids,
            embedding_weights_ptr=embedding_weights,
            output_ptr=output,
            batch_size=batch_size,
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            BLOCK_SIZE=block_size,
        )
        
        # Apply layer norm (keep this separate for now to ensure correctness)
        normalized = torch.nn.functional.layer_norm(output, (embedding_dim,), norm_weight, None, 1e-05)
        
        return output, normalized
    else:
        # Fallback to PyTorch for non-GPU
        embeddings = torch.nn.functional.embedding(input_ids, embedding_weights, 50283, None, 2.0, False, False)
        normalized = torch.nn.functional.layer_norm(embeddings, (embedding_weights.shape[1],), norm_weight, None, 1e-05)
        return embeddings, normalized


def replacement_func():
    return optimize_embedding_lookup