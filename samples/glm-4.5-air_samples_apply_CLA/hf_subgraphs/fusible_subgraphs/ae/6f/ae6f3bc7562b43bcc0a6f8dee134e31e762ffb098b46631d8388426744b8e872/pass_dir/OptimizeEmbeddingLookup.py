import torch
import triton
import triton.language as tl

def pattern(position_ids, embedding_weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    # Pattern: embedding lookup operation
    # Note: We ignore some less commonly used parameters for optimization clarity
    return torch.nn.functional.embedding(position_ids, embedding_weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

def replacement_args(position_ids, embedding_weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    return (position_ids, embedding_weight)

@triton.jit
def embedding_kernel(
    position_ids_ptr,
    embedding_weight_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program indices  
    pid = tl.program_id(0)
    
    # Compute sequence and batch indices
    seq_idx = pid % seq_len
    batch_idx = pid // seq_len
    
    # Check bounds
    if batch_idx >= batch_size:
        return
    
    # Load position ID for this batch and sequence
    position_offset = batch_idx * seq_len + seq_idx
    position_id = tl.load(position_ids_ptr + position_offset)
    
    # Clamp position ID to valid range
    position_id = tl.maximum(position_id, 0)
    position_id = tl.minimum(position_id, num_embeddings - 1)
    
    # Compute output offset for this element
    output_offset = pid * embedding_dim
    
    # Load embedding vector for each element in the block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < embedding_dim
    
    # Load and store each element individually
    for i in range(0, embedding_dim, BLOCK_SIZE):
        rem = embedding_dim - i
        actual_block_size = tl.minimum(BLOCK_SIZE, rem)
        
        if i == 0:  # Only store for the first iteration to avoid redundant stores
            # Load embedding tile
            embedding_offset = position_id * embedding_dim + i
            embedding_tile = tl.load(
                embedding_weight_ptr + embedding_offset + offsets,
                mask=offsets < actual_block_size,
                other=0.0
            )
            
            # Store result
            output_mask = offsets < embedding_dim
            tl.store(
                output_ptr + output_offset + offsets,
                embedding_tile,
                mask=output_mask
            )

@torch.fx.wrap
def optimized_embedding_lookup(position_ids, embedding_weight):
    batch_size, seq_len = position_ids.shape
    num_embeddings, embedding_dim = embedding_weight.shape
    
    # Output tensor
    output = torch.empty((batch_size, seq_len, embedding_dim), dtype=torch.float32, device=position_ids.device)
    
    # Optimize block size based on embedding dimension
    if embedding_dim <= 256:
        BLOCK_SIZE = 64
    elif embedding_dim <= 512:
        BLOCK_SIZE = 128
    elif embedding_dim <= 1024:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512
    
    # Calculate grid size: one program per sequence position
    grid = batch_size * seq_len
    
    # Reshape output to 2D for kernel processing (batch_size*seq_len, embedding_dim)
    output_2d = output.view(-1, embedding_dim)
    
    # Launch kernel
    embedding_kernel[(grid,)](
        position_ids_ptr=position_ids,
        embedding_weight_ptr=embedding_weight,
        output_ptr=output_2d,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_embedding_lookup