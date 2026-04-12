import torch
from torch import device
import triton
import triton.language as tl


def pattern(in_4, in_1):
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    return tmp_11


def replacement_args(in_4, in_1):
    return (in_4, in_1)


@triton.jit
def embedding_projection_kernel(
    input_ids_ptr,
    embedding_table_ptr,
    output_ptr,
    seq_len,
    vocab_size,
    embed_dim,
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_SIZE_EMBED: tl.constexpr,
):
    # Each program processes one token for better parallelism
    token_id = tl.program_id(0)
    
    if token_id >= seq_len:
        return
    
    # Load input ID for this token
    input_id = tl.load(input_ids_ptr + token_id)
    embed_offset = (input_id + 2).to(tl.int64)  # Add +2 offset and convert to int64
    
    # Check if embed_offset is within valid range for safety
    embed_mask = embed_offset < vocab_size
    
    # Process embedding vector for this token in chunks
    for k in range(0, embed_dim, BLOCK_SIZE_EMBED):
        embed_chunk_offset = k + tl.arange(0, BLOCK_SIZE_EMBED)
        chunk_mask = embed_chunk_offset < embed_dim
        
        # Compute global embedding pointer for this chunk
        global_embed_offset = embed_offset * embed_dim + embed_chunk_offset
        
        # Load embedding chunk with bounds checking
        embed_val = tl.load(embedding_table_ptr + global_embed_offset, 
                           mask=(embed_mask & chunk_mask), 
                           other=0.0)
        
        # Compute output offset (batch=0, token=token_id, embed=embed_chunk_offset)  
        output_offset = token_id * embed_dim + embed_chunk_offset
        
        # Store output chunk with bounds checking
        tl.store(output_ptr + output_offset, embed_val, mask=chunk_mask)


@torch.fx.wrap
def optimized_embedding_lookup(in_4, in_1):
    # Get dimensions
    seq_len = in_4.shape[0]
    embed_dim = in_1.shape[1]
    vocab_size = in_1.shape[0]
    
    # Create output tensor (1, seq_len, embed_dim) with same dtype as input embeddings
    output = torch.empty(1, seq_len, embed_dim, dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel - one program per token for better parallelism
    BLOCK_SIZE_EMBED = 128  # Process embedding in chunks of 64 elements
    
    num_programs = seq_len
    
    embedding_projection_kernel[(num_programs,)](
        input_ids_ptr=in_4,
        embedding_table_ptr=in_1,
        output_ptr=output,
        seq_len=seq_len,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        BLOCK_SIZE_TOKENS=1,  # One token per program
        BLOCK_SIZE_EMBED=BLOCK_SIZE_EMBED,
    )
    
    return output


def replacement_func():
    return optimized_embedding_lookup