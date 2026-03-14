import torch
import triton
import triton.language as tl

def pattern(input_ids, weight):
    # General case: optimized standalone embedding operation
    emb = torch.nn.functional.embedding(input_ids, weight, 0, None, 2.0, False, False)
    return emb

def replacement_args(input_ids, weight):
    return (input_ids, weight)

@triton.jit
def optimized_embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, seq) position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate output position
    output_pos = (batch_idx * seq_len + seq_idx) * embedding_dim
    
    # Load input ID
    input_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)
    
    # Load embedding vector for this input ID
    for k in range(0, embedding_dim, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < embedding_dim
        
        # Calculate weight offset for this input ID and embedding dimension
        weight_offset = input_id * embedding_dim + offsets
        embedding_vector = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        # Store to output
        output_offset = output_pos + offsets
        tl.store(output_ptr + output_offset, embedding_vector, mask=mask)

@torch.fx.wrap  
def optimized_embedding(input_ids, weight):
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = weight.shape
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, embedding_dim), dtype=weight.dtype, device=weight.device)
    
    if batch_size == 0 or seq_len == 0:
        return output
    
    # Calculate grid dimensions
    grid = (batch_size, seq_len)
    
    # Optimal block size for embedding vector loading (balances memory coalescing and occupancy)
    BLOCK_SIZE = 64
    
    # Launch kernel
    optimized_embedding_kernel[grid](
        input_ids,
        weight,
        output,
        num_embeddings,
        embedding_dim,
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_embedding