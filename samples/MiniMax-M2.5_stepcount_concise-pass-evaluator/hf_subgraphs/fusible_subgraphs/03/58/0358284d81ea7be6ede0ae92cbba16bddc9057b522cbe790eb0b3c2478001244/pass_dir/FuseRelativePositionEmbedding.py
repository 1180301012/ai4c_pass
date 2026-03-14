import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the relative position embedding computation pattern with size 128.
    
    The original computation:
    1. tmp_1 = torch.arange(128) 
    2. tmp_2 = tmp_1.view(1, -1)
    3. tmp_3 = in_1 - tmp_2  (broadcast to [128, 128])
    4. tmp_4 = tmp_3 + 2048
    5. tmp_5 = tmp_4 - 1
    6. tmp_6 = embedding(tmp_5, in_0)
    7. tmp_7 = tmp_6.to(float32)
    """
    # Create arange and view - using literal 128 for matching
    tmp_1 = torch.arange(128, dtype=torch.int64, device='cuda')
    tmp_2 = tmp_1.view(1, -1)
    
    # Compute relative positions: in_1 - tmp_2
    tmp_3 = in_1 - tmp_2
    
    # Add and subtract constants to shift the range
    tmp_4 = tmp_3 + 2048
    tmp_5 = tmp_4 - 1
    
    # Embedding lookup with padding_idx=2.0
    tmp_6 = torch.nn.functional.embedding(tmp_5, in_0, None, None, 2.0, False, False)
    
    # Cast to float32
    tmp_7 = tmp_6.to(dtype=torch.float32)
    
    return tmp_7


def replacement_args(in_0, in_1):
    """Extract arguments needed for replacement"""
    return (in_0, in_1)


@triton.jit
def fused_position_kernel(
    position_ids_ptr,
    embedding_table_ptr,
    output_ptr,
    N: tl.constexpr,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes relative position embeddings:
    1. Computes relative positions: position_ids[i] - j for all j in range(N)
    2. Shifts by 2047 (2048 - 1)
    3. Embedding lookup
    """
    # Each program handles a row
    row_idx = tl.program_id(0)
    
    # Load position_id for this row
    position_id = tl.load(position_ids_ptr + row_idx)
    
    # Compute shifted position_id
    shifted_pos = position_id + 2047
    
    # For each column j in range(N)
    for j in range(0, N, BLOCK_SIZE):
        # Create column indices
        col_offsets = j + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < N
        
        # Compute relative positions: shifted_pos - col_offsets
        # This is equivalent to: (position_id + 2047) - j
        rel_pos = shifted_pos - col_offsets
        
        # Create mask for valid positions (padding_idx=2.0 means we should mask pos==2)
        # Actually, let's check the embedding call: torch.nn.functional.embedding(tmp_5, tmp_0, None, None, 2.0, False, False)
        # padding_idx=2.0 means indices equal to 2 will be masked (zeroed out)
        mask = (rel_pos != 2) & col_mask
        
        # Clamp to valid range [0, 4094] for embedding table
        # embedding table has shape [4095, 64], so valid indices are [0, 4094]
        clamped_pos = tl.where((rel_pos >= 0) & (rel_pos < 4095), rel_pos, 0)
        
        # Load embeddings for each position
        # embedding_dim = 64
        for dim in range(0, embedding_dim, 4):
            dim_offsets = dim + tl.arange(0, 4)
            dim_mask = dim_offsets < embedding_dim
            
            # Compute the embedding table offset
            # table_offset = clamped_pos * embedding_dim + dim_offsets
            table_offsets = clamped_pos[:, None] * embedding_dim + dim_offsets[None, :]
            table_offsets = tl.reshape(table_offsets, (BLOCK_SIZE, 4))
            
            # Load embeddings
            emb = tl.load(embedding_table_ptr + table_offsets, mask=dim_mask & mask[:, None], other=0.0)
            
            # Store result
            out_offsets = (row_idx * N + col_offsets[:, None]) * embedding_dim + dim_offsets[None, :]
            out_offsets = tl.reshape(out_offsets, (BLOCK_SIZE, 4))
            tl.store(output_ptr + out_offsets, emb, mask=dim_mask & mask[:, None])


@torch.fx.wrap
def fused_embedding_kernel(position_ids, embedding_table):
    """
    Wrapper for the fused position embedding kernel.
    """
    N = position_ids.shape[0]
    embedding_dim = embedding_table.shape[1]
    
    # Output shape: [N, N, embedding_dim]
    output = torch.zeros((N, N, embedding_dim), dtype=torch.float32, device=position_ids.device)
    
    BLOCK_SIZE = 16  # Smaller block for memory efficiency
    
    # Launch grid: one program per row
    grid = (N,)
    
    fused_position_kernel[grid](
        position_ids_ptr=position_ids,
        embedding_table_ptr=embedding_table,
        output_ptr=output,
        N=N,
        embedding_dim=embedding_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    """Return the replacement function"""
    return fused_embedding_kernel