import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern: Simple embedding lookup
    """
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embed_dim: tl.constexpr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Process one element per program for simplicity and correctness
    idx = pid
    if idx >= batch_size * seq_len:
        return
    
    # Convert linear index to batch and sequence indices
    batch_idx = idx // seq_len
    seq_idx = idx % seq_len
    
    # Load token ID
    token_id = tl.load(input_ids_ptr + idx)
    
    # Load embedding
    if token_id < vocab_size and token_id >= 0:
        embed_offset = token_id * embed_dim
        embeddings = tl.load(weight_ptr + embed_offset + tl.arange(0, embed_dim))
    else:
        embeddings = tl.zeros((embed_dim,), dtype=tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else tl.float16)
    
    # Store result
    output_offset = idx * embed_dim
    tl.store(output_ptr + output_offset + tl.arange(0, embed_dim), embeddings)

@torch.fx.wrap
def optimized_embedding_lookup(input_ids, weight):
    batch_size, seq_len = input_ids.shape
    embed_dim = weight.shape[1]
    vocab_size = weight.shape[0]
    
    output = torch.empty((batch_size, seq_len, embed_dim), dtype=weight.dtype, device=weight.device)
    
    # Use optimal block size for batch_size * seq_len elements
    total_elements = batch_size * seq_len
    
    if embed_dim == 128:
        BLOCK_SIZE = 2048  # Larger blocks for better GPU utilization
    elif embed_dim >= 64:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 512
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_embedding_kernel[(num_programs,)](
        input_ids_ptr=input_ids,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_embedding_lookup