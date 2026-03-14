import torch
import triton
import triton.language as tl

def pattern(input_ids, weight):
    # Enhanced pattern: optimized standalone embedding operation
    emb = torch.nn.functional.embedding(input_ids, weight, 0, None, 2.0, False, False)
    return emb

def replacement_args(input_ids, weight):
    return (input_ids, weight)

@triton.jit
def enhanced_embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    batch_size,
    seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Enhanced kernel with better memory coalescing and parallelism
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Load input ID
    input_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)
    
    # Process embedding dimensions in blocks for memory coalescing
    for k in range(0, embedding_dim, BLOCK_SIZE_N):
        offsets = k + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < embedding_dim
        
        # Calculate weight offset and load embedding vector efficiently
        weight_offset = input_id * embedding_dim + offsets
        embedding_vector = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        # Calculate output position
        output_pos = (batch_idx * seq_len + seq_idx) * embedding_dim + offsets
        tl.store(output_ptr + output_pos, embedding_vector, mask=mask)

@torch.fx.wrap
def enhanced_embedding(input_ids, weight):
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = weight.shape
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, embedding_dim), dtype=weight.dtype, device=weight.device)
    
    if batch_size == 0 or seq_len == 0:
        return output
    
    # Calculate grid dimensions
    grid = (batch_size, seq_len)
    
    # Optimal block sizes for better memory coalescing
    BLOCK_SIZE_M = 1  # Each program handles one batch element
    BLOCK_SIZE_N = 32  # Process 32 embedding dimensions at a time for better cache utilization
    
    # Launch kernel with enhanced configuration
    enhanced_embedding_kernel[grid](
        input_ids,
        weight,
        output,
        num_embeddings,
        embedding_dim,
        batch_size,
        seq_len,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return enhanced_embedding