import torch
import triton
import triton.language as tl

# Simple pattern matching function
def pattern(weights, positions):
    # Match the embedding computation with index manipulation
    range_base = torch.arange(128, dtype=torch.int64)
    range_view = range_base.view(1, -1)
    indices = positions - range_view
    final_indices = indices + 2047
    embeddings = torch.nn.functional.embedding(final_indices, weights)
    return embeddings.to(torch.float32)

# Argument extraction function
def replacement_args(weights, positions):
    return (weights, positions)

@triton.jit
def fused_embedding_kernel_128(
    weight_ptr,  # embedding weights [num_embeddings, embedding_dim]
    position_ids_ptr,  # position_ids [batch_size, seq_len]
    output_ptr,  # output [batch_size, seq_len, embedding_dim]
    num_embeddings,
    embedding_dim,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output tensor
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Calculate the embedding index
    base_val = tl.arange(seq_len, dtype=tl.int64)[seq_idx]
    position_id = tl.load(position_ids_ptr + batch_idx * seq_len + seq_idx)
    embedding_idx = position_id - base_val + 2048 - 1
    
    # Ensure index is within bounds
    embedding_idx = tl.maximum(embedding_idx, tl.constant(0))
    embedding_idx = tl.minimum(embedding_idx, num_embeddings - 1)
    
    # Load embedding weights for this sequence position
    weight_offset = embedding_idx * embedding_dim
    embeddings = tl.load(weight_ptr + weight_offset, mask=tl.arange(embedding_dim) < embedding_dim)
    
    # Store result
    output_offset = pid * embedding_dim
    tl.store(output_ptr + output_offset, embeddings)

@torch.fx.wrap
def fused_embedding_computation_128(weight, position_ids):
    batch_size, seq_len = position_ids.shape
    num_embeddings, embedding_dim = weight.shape
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, embedding_dim), 
                        dtype=torch.float32, device=weight.device)
    
    # Calculate grid size
    total_elements = batch_size * seq_len
    BLOCK_SIZE = 128  # Can be tuned
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_embedding_kernel_128[(num_programs,)](
        weight_ptr=weight,
        position_ids_ptr=position_ids,
        output_ptr=output,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.squeeze(-1)  # Remove last dimension to match original output shape

# Replacement function
def replacement_func():
    return fused_embedding_computation_128