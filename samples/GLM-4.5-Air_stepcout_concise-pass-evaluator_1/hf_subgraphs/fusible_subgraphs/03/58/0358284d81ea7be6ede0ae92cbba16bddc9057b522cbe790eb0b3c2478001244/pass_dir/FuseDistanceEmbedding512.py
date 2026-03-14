import torch
import triton
import triton.language as tl


# Pattern matching function - matches the distance embedding computation with 512 positions
# This includes: arange + view + subtract + add + subtract + embedding + to(float32)
def pattern(in_0, in_1):
    tmp_1 = torch.arange(512)
    tmp_2 = tmp_1.view(1, -1)
    tmp_3 = in_1 - tmp_2
    tmp_4 = tmp_3 + 2048
    tmp_5 = tmp_4 - 1
    tmp_6 = torch.nn.functional.embedding(tmp_5, in_0, None, None, 2.0, False, False)
    tmp_7 = tmp_6.to(dtype=torch.float32)
    return (tmp_7,)


def replacement_args(in_0, in_1):
    # Extract the embedding table and position_ids
    return (in_0, in_1)


# Optimized Triton kernel for distance embedding computation
@triton.jit
def distance_embedding_kernel(
    embedding_table_ptr,
    position_ids_ptr,
    output_ptr,
    n_positions,
    embedding_dim: tl.constexpr,
    num_embeddings: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused distance embedding kernel that:
    1. Computes position indices on-the-fly
    2. Applies offset (+2048-1 = +2047)
    3. Performs embedding lookup
    4. Outputs float32 [n_positions, n_positions, embedding_dim]
    """
    # Each program handles one element in the NxN index matrix
    # We use 2D grid: (row, col)
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Load position_id for this row
    position_id = tl.load(position_ids_ptr + row_idx)
    
    # Compute the shifted index: position_id + 2047 - col_idx
    shifted_idx = position_id + 2047 - col_idx
    
    # Load embedding table pointer
    embedding_table = embedding_table_ptr
    
    # Process each embedding dimension
    for dim_idx in tl.range(0, embedding_dim):
        # Check bounds - embeddings are in range [0, num_embeddings)
        is_valid = (shifted_idx >= 0) & (shifted_idx < num_embeddings)
        
        # Compute the actual memory address for valid indices
        ptr = embedding_table + shifted_idx * embedding_dim + dim_idx
        
        # Load embedding value (0 if out of bounds)
        value = tl.load(ptr, mask=is_valid, other=0.0)
        
        # Store to output [row, col, dim]
        out_ptr = output_ptr + row_idx * n_positions * embedding_dim + col_idx * embedding_dim + dim_idx
        tl.store(out_ptr, value)


@torch.fx.wrap
def triton_distance_embedding(embedding_table, position_ids):
    """
    Wrapper function for the fused distance embedding kernel.
    Replaces the entire sequence:
    - torch.arange
    - view
    - subtraction
    - addition
    - embedding lookup
    - to(float32)
    """
    n_positions = position_ids.shape[0]
    embedding_dim = embedding_table.shape[1]
    num_embeddings = embedding_table.shape[0]
    
    # Output tensor: [n_positions, n_positions, embedding_dim] in float32
    output = torch.zeros((n_positions, n_positions, embedding_dim), 
                         dtype=torch.float32, 
                         device=embedding_table.device)
    
    # Launch kernel with 2D grid: (n_positions, n_positions)
    grid = (n_positions, n_positions)
    
    distance_embedding_kernel[grid](
        embedding_table_ptr=embedding_table,
        position_ids_ptr=position_ids,
        output_ptr=output,
        n_positions=n_positions,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        BLOCK_SIZE=1,  # Not used but required
    )
    
    return output


def replacement_func():
    return triton_distance_embedding