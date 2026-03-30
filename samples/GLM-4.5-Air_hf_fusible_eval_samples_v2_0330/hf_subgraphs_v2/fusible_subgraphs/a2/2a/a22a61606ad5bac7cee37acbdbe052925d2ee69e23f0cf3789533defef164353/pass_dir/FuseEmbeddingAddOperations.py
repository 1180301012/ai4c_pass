import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_3, in_6, in_2, in_7, in_1):
    # Three embedding operations followed by additions
    tmp_7 = torch.nn.functional.embedding(in_0, in_3, 0, None, 2.0, False, False)
    tmp_8 = torch.nn.functional.embedding(in_6, in_2, None, None, 2.0, False, False)
    tmp_9 = tmp_7 + tmp_8
    tmp_10 = torch.nn.functional.embedding(in_7, in_1, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_11 = tmp_9
    return tmp_11

def replacement_args(in_0, in_3, in_6, in_2, in_7, in_1):
    return (in_0, in_3, in_6, in_2, in_7, in_1)

@triton.jit
def fused_embedding_kernel(
    input_ids1_ptr, input_ids2_ptr, input_ids3_ptr,
    weight1_ptr, weight2_ptr, weight3_ptr,
    output_ptr,
    batch_size, seq_len, embedding_dim,
    vocab_size1, vocab_size2, vocab_size3,
    BLOCK_SIZE: tl.constexpr, EMBEDDING_BLOCK: tl.constexpr,
):
    # Each program handles one position in the sequence
    program_id = tl.program_id(0)
    batch_id = program_id // seq_len
    seq_id = program_id % seq_len
    
    if batch_id >= batch_size or seq_id >= seq_len:
        return
    
    # Load input IDs for this position across all three embeddings
    input_ids1 = tl.load(input_ids1_ptr + batch_id * seq_len + seq_id)
    input_ids2 = tl.load(input_ids2_ptr + batch_id * seq_len + seq_id)
    input_ids3 = tl.load(input_ids3_ptr + batch_id * seq_len + seq_id)
    
    # Convert to zero-based indices if needed
    if input_ids1 >= vocab_size1:
        input_ids1 = 0
    if input_ids2 >= vocab_size2:
        input_ids2 = 0
    if input_ids3 >= vocab_size3:
        input_ids3 = 0
    
    # Calculate base indices for each embedding
    emb1_offset = input_ids1 * embedding_dim
    emb2_offset = input_ids2 * embedding_dim
    emb3_offset = input_ids3 * embedding_dim
    
    # Load embedding vectors with tiling for memory efficiency
    offsets = tl.arange(0, EMBEDDING_BLOCK)
    mask = offsets < embedding_dim
    
    # Load from first embedding
    emb1_start = emb1_offset + offsets
    emb1_vec = tl.load(weight1_ptr + emb1_start, mask=mask, other=0.0)
    
    # Load from second embedding
    emb2_start = emb2_offset + offsets
    emb2_vec = tl.load(weight2_ptr + emb2_start, mask=mask, other=0.0)
    
    # Load from third embedding
    emb3_start = emb3_offset + offsets
    emb3_vec = tl.load(weight3_ptr + emb3_start, mask=mask, other=0.0)
    
    # Accumulate: emb1 + emb2 + emb3
    result_vec = emb1_vec + emb2_vec + emb3_vec
    
    # Store result
    output_offsets = (batch_id * seq_len + seq_id) * embedding_dim + offsets
    tl.store(output_ptr + output_offsets, result_vec, mask=mask)

@torch.fx.wrap
def fused_embedding_forward(input_ids1, weight1, input_ids2, weight2, input_ids3, weight3):
    batch_size, seq_len = input_ids1.shape
    embedding_dim = weight1.shape[1]
    
    # Determine vocab sizes
    vocab_size1 = weight1.shape[0]
    vocab_size2 = weight2.shape[0]
    vocab_size3 = weight3.shape[0]
    
    # Set optimal block sizes
    BLOCK_SIZE = 1024
    EMBEDDING_BLOCK = min(1024, embedding_dim)
    
    # Calculate grid size
    total_elements = batch_size * seq_len
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, embedding_dim), dtype=weight1.dtype, device=weight1.device)
    
    # Launch kernel
    fused_embedding_kernel[(num_programs,)](
        input_ids1_ptr=input_ids1,
        input_ids2_ptr=input_ids2,
        input_ids3_ptr=input_ids3,
        weight1_ptr=weight1,
        weight2_ptr=weight2,
        weight3_ptr=weight3,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        vocab_size1=vocab_size1,
        vocab_size2=vocab_size2,
        vocab_size3=vocab_size3,
        BLOCK_SIZE=BLOCK_SIZE,
        EMBEDDING_BLOCK=EMBEDDING_BLOCK,
    )
    
    return output

def replacement_func():
    return fused_embedding_forward