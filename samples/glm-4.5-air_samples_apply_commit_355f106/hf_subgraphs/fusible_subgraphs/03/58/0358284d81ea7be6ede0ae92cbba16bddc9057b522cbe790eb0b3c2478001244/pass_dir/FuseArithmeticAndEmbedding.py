import torch
import triton
import triton.language as tl

# Pattern matching function - matches the arithmetic operations + embedding computation
def pattern(in_0, in_1):
    # tmp_1 = torch.arange(sequence_length, dtype=torch.int64, device=device(type='cuda', index=0))
    # tmp_2 = tmp_1.view(1, -1)
    # Use arange based on the second dimension of in_1 (seq_len)
    seq_len = in_1.shape[1]
    base_range = torch.arange(seq_len, dtype=torch.int64, device=in_0.device)
    base_range_view = base_range.view(1, -1)
    
    # tmp_3 = in_1 - tmp_2
    # tmp_4 = tmp_3 + 2048
    # tmp_5 = tmp_4 - 1
    indices = in_1 - base_range_view
    adjusted_indices = indices + 2048
    final_indices = adjusted_indices - 1
    
    # tmp_6 = torch.nn.functional.embedding(tmp_5, in_0, None, None, 2.0, False, False)
    embeddings = torch.nn.functional.embedding(final_indices, in_0, None, None, 2.0, False, False)
    
    # tmp_7 = tmp_6.to(dtype=torch.float32)
    result = embeddings.to(dtype=torch.float32)
    
    return result

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_embedding_kernel(
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
def fused_embedding_computation(weight, position_ids):
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
    fused_embedding_kernel[(num_programs,)](
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
    return fused_embedding_computation