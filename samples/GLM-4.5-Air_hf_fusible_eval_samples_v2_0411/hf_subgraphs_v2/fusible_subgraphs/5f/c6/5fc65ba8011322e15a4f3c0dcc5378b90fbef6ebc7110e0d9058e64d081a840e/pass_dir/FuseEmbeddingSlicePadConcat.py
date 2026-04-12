import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    # Simple pattern: just the embedding operation (that works)
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def optimized_embedding_kernel(
    input_ids_ptr,
    embedding_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    embed_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each warp/thread handles a contiguous block of features
    batch_seq_idx = tl.program_id(0)
    feat_block_idx = tl.program_id(1)
    
    # Calculate batch and sequence indices
    batch_idx = batch_seq_idx // seq_len
    seq_idx = batch_seq_idx % seq_len
    
    # Calculate feature range for this block
    feat_start = feat_block_idx * BLOCK_SIZE
    feat_end = min(feat_start + BLOCK_SIZE, embed_dim)
    
    # Early exit if this block is out of bounds
    if batch_idx >= batch_size or seq_idx >= seq_len or feat_start >= embed_dim:
        return
    
    # Get the input token ID once per thread
    token_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)
    
    # Process each feature in the block
    for feat_idx in range(feat_start, feat_end):
        # Compute global memory offset for embedding feature
        embed_offset = token_id * embed_dim + feat_idx
        
        # Load the embedding feature
        feature = tl.load(embedding_ptr + embed_offset)
        
        # Store the output
        output_offset = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + feat_idx
        tl.store(output_ptr + output_offset, feature)


@torch.fx.wrap
def optimized_embedding(input_ids, embedding_weights):
    # Get input shapes
    batch_size, seq_len = input_ids.shape
    vocab_size, embed_dim = embedding_weights.shape
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, embed_dim), 
                        dtype=embedding_weights.dtype, 
                        device=embedding_weights.device)
    
    # Choose block size based on embedding dimension for optimal GPU occupancy
    BLOCK_SIZE = 16  # Better for 128-dimensional embeddings on A30 GPU
    
    # Calculate grid dimensions
    batch_seq_total = batch_size * seq_len
    num_feature_blocks = (embed_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel with 2D grid (batch*sequence, feature_blocks)
    grid = (batch_seq_total, num_feature_blocks)
    optimized_embedding_kernel[grid](
        input_ids,
        embedding_weights,
        output,
        batch_size,
        seq_len,
        embed_dim,
        BLOCK_SIZE
    )
    
    return output


def replacement_func():
    return optimized_embedding