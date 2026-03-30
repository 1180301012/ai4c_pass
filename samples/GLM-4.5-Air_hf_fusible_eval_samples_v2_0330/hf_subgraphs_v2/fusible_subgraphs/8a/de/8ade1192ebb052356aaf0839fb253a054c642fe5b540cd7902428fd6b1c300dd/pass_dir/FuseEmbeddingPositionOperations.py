import torch
import triton
import triton.language as tl

def pattern(in_4, in_1):
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    return tmp_11

@triton.jit
def embedding_kernel(
    positions_ptr,
    weight_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    vocab_size: tl.constexpr,
    embed_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the sequence
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # We'll process a block of embedding dimensions
    embed_idx = tl.arange(0, BLOCK_SIZE)
    mask = embed_idx < embed_dim
    
    # Load processed position (add 2 to cached position)
    pos = tl.load(positions_ptr + seq_idx) + 2
    
    # Clamp position to valid vocabulary range
    pos = tl.maximum(pos, 0)
    pos = tl.minimum(pos, vocab_size - 1)
    
    # Calculate offset in weight matrix
    weight_offset = pos * embed_dim
    
    # Load embedding weights
    weights = tl.load(weight_ptr + weight_offset + embed_idx, mask=mask)
    
    # Store output
    output_offset = (batch_idx * seq_len + seq_idx) * embed_dim
    tl.store(out_ptr + output_offset + embed_idx, weights, mask=mask)

@torch.fx.wrap
def fused_embedding_lookup(in_4, in_1):
    # Get tensor shapes
    seq_len = in_4.shape[0]
    batch_size = 1  # Always 1 for transformer models
    vocab_size = in_1.shape[0]
    embed_dim = in_1.shape[1]
    
    # Determine optimal block size and grid
    BLOCK_SIZE = 1024  # Process embedding dimensions in chunks
    num_embed_programs = (embed_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor [1, seq_len, embed_dim]
    out = torch.empty((batch_size, seq_len, embed_dim), dtype=in_1.dtype, device=in_1.device)
    
    # Reshape positions from [seq_len] to [1, seq_len] for processing
    positions_reshaped = in_4.unsqueeze(0)  # [1, seq_len]
    positions_processed = positions_reshaped + 2  # Add 2 to positions
    
    # Launch kernel 
    embedding_kernel[(batch_size, seq_len, num_embed_programs)](
        positions_ptr=positions_processed,
        weight_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_args(in_4, in_1):
    return (in_4, in_1)

def replacement_func():
    return fused_embedding_lookup