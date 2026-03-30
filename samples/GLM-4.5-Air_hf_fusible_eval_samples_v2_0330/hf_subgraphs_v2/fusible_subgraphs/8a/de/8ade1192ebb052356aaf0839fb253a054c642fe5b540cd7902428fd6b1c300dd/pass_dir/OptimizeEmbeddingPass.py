import torch
import triton
import triton.language as tl

# Pattern for embedding lookup: position + embedding
def pattern(in_4, in_1):
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    return tmp_11

@triton.jit
def optimized_embedding_kernel(
    positions_ptr,
    weight_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    vocab_size: tl.constexpr,
    embed_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one embedding dimension for one sequence element
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1) 
    embed_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Check if we're within bounds for embedding dimensions
    embed_mask = embed_idx < embed_dim
    
    # Load and process position (add 2)
    pos = tl.load(positions_ptr + seq_idx).to(tl.int32) + 2
    
    # Clamp position to valid vocabulary range exactly like PyTorch
    pos = tl.maximum(pos, 0)
    pos = tl.minimum(pos, vocab_size - 1)
    
    # Calculate weight matrix offset
    weight_offset = pos * embed_dim + embed_idx
    
    # Load embedding weights exactly like PyTorch does
    weights = tl.load(weight_ptr + weight_offset, mask=embed_mask & (embed_idx < vocab_size * embed_dim), other=0.0)
    
    # Store result in row-major order (standard for PyTorch)
    output_offset = (batch_idx * seq_len + seq_idx) * embed_dim + embed_idx
    tl.store(out_ptr + output_offset, weights, mask=embed_mask)

@torch.fx.wrap
def optimized_embedding_lookup(in_4, in_1):
    # Get tensor shapes
    seq_len = in_4.shape[0]
    batch_size = 1  # Always 1 for transformer models
    vocab_size = in_1.shape[0]
    embed_dim = in_1.shape[1]
    
    # Use optimal block size for better GPU utilization and memory access
    BLOCK_SIZE = 128  # Process 128 embedding dimensions per program
    
    # Calculate number of programs needed for embedding dimensions
    num_embed_programs = (embed_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor [1, seq_len, embed_dim] in row-major order
    out = torch.empty((batch_size, seq_len, embed_dim), dtype=in_1.dtype, device=in_1.device)
    
    # Launch optimized kernel with 3D grid
    optimized_embedding_kernel[(batch_size, seq_len, num_embed_programs)](
        positions_ptr=in_4,
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
    return optimized_embedding_lookup