import torch
import triton
import triton.language as tl

# Pattern matching function for embedding + scaling + addition fusion
def pattern(embed_positions_weight, embed_tokens_weight, token_idx):
    """
    Match the computation pattern:
    1. Token embedding lookup with scaling
    2. Position generation and embedding lookup  
    3. Addition of both embeddings
    """
    # Token embedding lookup with scaling
    token_embeddings = torch.nn.functional.embedding(token_idx, embed_tokens_weight, 1, None, 2.0, False, False)
    scaled_token_embeddings = token_embeddings * 16.0
    
    # Position generation and embedding lookup
    tmp_6 = torch.arange(0, 1, dtype = torch.int64, device = 'cuda')
    tmp_7 = tmp_6.expand(1, -1)
    positions = tmp_7 + 2
    position_embeddings = torch.nn.functional.embedding(positions, embed_positions_weight, None, None, 2.0, False, False)
    
    # Add embeddings
    combined_embeddings = scaled_token_embeddings + position_embeddings
    
    return combined_embeddings

# Argument extraction function
def replacement_args(embed_positions_weight, embed_tokens_weight, token_idx):
    return (embed_positions_weight, embed_tokens_weight, token_idx)

# Optimized fused embedding kernel using Triton
@triton.jit
def fused_embedding_kernel(
    token_idx_ptr,
    embed_tokens_ptr,
    embed_positions_ptr,
    output_ptr,
    embed_tokens_dim0,
    embed_tokens_dim1,
    embed_positions_dim0, 
    embed_positions_dim1,
    num_tokens,
    embed_dim,
    scale_factor: tl.constexpr,
    pos_offset: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Token embedding lookup with scaling
    2. Position embedding lookup
    3. Addition of both embeddings
    """
    # Get program IDs for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    offsets = m_offset + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offsets_n = n_offset + tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    # Load token indices
    if pid_m < num_tokens[0] and pid_n < num_tokens[1]:
        token_idx = tl.load(token_idx_ptr + pid_m * num_tokens[1] + pid_n)
        
        # Load token embedding
        token_emb_offset = token_idx * embed_tokens_dim1
        token_embeddings = tl.load(
            embed_tokens_ptr + token_emb_offset + offsets_n,
            mask=(offsets_n < embed_tokens_dim1),
            other=0.0
        )
        
        # Scale token embeddings
        scaled_token_embeddings = token_embeddings * scale_factor
        
        # Generate position (value = 2 for this specific case)
        position_value = pos_offset
        position_emb_offset = position_value * embed_positions_dim1
        position_embeddings = tl.load(
            embed_positions_ptr + position_emb_offset + offsets_n,
            mask=(offsets_n < embed_positions_dim1),
            other=0.0
        )
        
        # Add embeddings
        combined_embeddings = scaled_token_embeddings + position_embeddings
        
        # Store result
        output_offset = (pid_m * embed_dim + pid_n * embed_dim + offsets_n)
        tl.store(output_ptr + output_offset, combined_embeddings, mask=offsets_n < embed_dim)
    else:
        # Handle out-of-bounds case
        if pid_m < num_tokens[0] and pid_n < num_tokens[1]:
            zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            tl.store(output_ptr + offsets * embed_dim + offsets_n, zeros, mask=offsets_n < embed_dim)

@torch.fx.wrap  
def fused_embedding_lookup(embed_positions_weight, embed_tokens_weight, token_idx):
    """
    Wrapper function for the fused embedding operation
    """
    # Determine output shape
    num_tokens = token_idx.shape  # [1, 1]
    embed_dim = embed_positions_weight.shape[1]  # 256
    
    # Create output tensor
    output_shape = list(num_tokens) + [embed_dim]
    output = torch.empty(output_shape, dtype=embed_positions_weight.dtype, device=embed_positions_weight.device)
    
    # Hardcoded parameters from original computation
    scale_factor = 16.0
    pos_offset = 2
    
    # Set up kernel grid configuration
    BLOCK_SIZE_M = 1  # Since we only have 1 token in batch and sequence dim
    BLOCK_SIZE_N = 64  # Process 64 embedding dimensions per work item
    
    # Calculate grid size
    grid_m = (num_tokens[0] + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (embed_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch Triton kernel
    fused_embedding_kernel[[
        grid_m,
        grid_n
    ]](
        token_idx,
        embed_tokens_weight,
        embed_positions_weight,
        output,
        embed_tokens_weight.shape[0],  # embed_tokens_dim0
        embed_tokens_weight.shape[1],  # embed_tokens_dim1
        embed_positions_weight.shape[0],  # embed_positions_dim0
        embed_positions_weight.shape[1],  # embed_positions_dim1
        num_tokens,
        embed_dim,
        scale_factor,
        pos_offset,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output

# Replacement function (returns function reference, not a call)
def replacement_func():
    return fused_embedding_lookup