import torch
import triton
import triton.language as tl

def pattern(input_ids, weight):
    return torch.nn.functional.embedding(input_ids, weight, 1, None, 2.0, False, False)

def replacement_args(input_ids, weight):
    return (input_ids, weight)

@triton.jit
def optimized_embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    input_ids_shape0,
    input_ids_shape1,
    vocab_size,
    embedding_dim,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one token position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Bounds checking
    if batch_idx >= input_ids_shape0 or seq_idx >= input_ids_shape1:
        return
    
    # Calculate output offset for this position
    output_offset = (batch_idx * input_ids_shape1 + seq_idx) * embedding_dim
    
    # Load input ID
    input_id = tl.load(input_ids_ptr + batch_idx * input_ids_shape1 + seq_idx)
    
    # If it's a padding token (ID 2), zero out the output
    if input_id == 2:
        for i in range(0, embedding_dim, BLOCK_SIZE_N):
            idx = i + tl.arange(0, BLOCK_SIZE_N)
            mask = idx < embedding_dim
            tl.store(output_ptr + output_offset + idx, 0.0, mask=mask)
        return
    
    # Load embedding for non-padding tokens
    emb_offset = input_id * embedding_dim
    
    # Load embedding vector efficiently
    for i in range(0, embedding_dim, BLOCK_SIZE_N):
        idx = i + tl.arange(0, BLOCK_SIZE_N)
        mask = idx < embedding_dim
        emb_val = tl.load(weight_ptr + emb_offset + idx, mask=mask, other=0.0)
        tl.store(output_ptr + output_offset + idx, emb_val, mask=mask)

@torch.fx.wrap  
def fused_embedding_with_mask(input_ids, weight):
    batch_size, seq_len = input_ids.shape
    vocab_size, embedding_dim = weight.shape
    
    # Create output tensor
    output = torch.empty(batch_size, seq_len, embedding_dim, dtype=torch.float32, device=input_ids.device)
    
    # Calculate grid dimensions
    grid = lambda meta: (batch_size, seq_len)
    
    # Launch optimized kernel
    optimized_embedding_kernel[grid](
        input_ids,
        weight,
        output,
        batch_size,
        seq_len,
        vocab_size,
        embedding_dim,
        128  # BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return fused_embedding_with_mask