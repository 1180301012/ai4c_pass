import torch
import triton
import triton.language as tl

# Pattern matching function - matches the two embedding operations and their addition
def pattern(in_0, in_1, in_2, in_3, in_4, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12):
    # Use inputs directly
    _ = in_0, in_1, in_2, in_3, in_4, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12
    
    # First embedding lookup and scaling
    embedding1 = torch.nn.functional.embedding(in_4, in_1, 1, None, 2.0, False, False)
    scaled1 = embedding1 * 16.0
    
    # Second embedding lookup (for position)
    embedding2 = torch.nn.functional.embedding(tmp_8, in_0, None, None, 2.0, False, False)
    
    # Addition of the results
    result = scaled1 + embedding2
    
    return result

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12):
    return (in_4, in_1, 16.0, in_0, tmp_8)

# Triton kernel for fused dual embedding with better tile sizes
@triton.jit
def fused_embedding_kernel_optimized(
    input_ids_ptr,
    pos_ids_ptr,
    weight1_ptr,      # in_1: [64044, 256] - large vocabulary
    weight2_ptr,      # in_0: [514, 256]  - small vocabulary  
    output_ptr,
    input_size,
    vocab_size1,
    vocab_size2,
    embed_dim,
    scalar1,
    BLOCK_SIZE_M: tl.constexpr,  # Number of rows to process per block
    BLOCK_SIZE_K: tl.constexpr,  # Vector size for coalescing
):
    # Each program handles BLOCK_SIZE_M consecutive tokens
    row_idx = tl.program_id(0)
    if row_idx * BLOCK_SIZE_M >= input_size:
        return
    
    offsets_row = row_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_col = tl.arange(0, BLOCK_SIZE_K)
    
    # Process in chunks of BLOCK_SIZE_K features
    for col_start in range(0, embed_dim, BLOCK_SIZE_K):
        col_end = min(col_start + BLOCK_SIZE_K, embed_dim)
        current_k = col_end - col_start
        
        # Create offset masks for this chunk
        row_mask = offsets_row < input_size
        col_mask = offsets_col[:current_k] < current_k
        
        # Load input IDs
        input_ids = tl.load(input_ids_ptr + offsets_row, mask=row_mask, other=0)
        pos_ids = tl.load(pos_ids_ptr + offsets_row, mask=row_mask, other=0)
        
        # Initialize results for this chunk
        chunk_result = tl.zeros((BLOCK_SIZE_M, current_k), dtype=tl.float16)
        
        # Process both embeddings in parallel
        for i in range(BLOCK_SIZE_M):
            if row_mask[i]:
                # First embedding lookup with bounds check
                if input_ids[i] < vocab_size1 and input_ids[i] >= 0:
                    emb_offset1 = input_ids[i] * embed_dim + col_start
                    for k in range(current_k):
                        if col_mask[k]:
                            emb1 = tl.load(weight1_ptr + emb_offset1 + k, mask=k < current_k, other=0.0)
                            chunk_result[i, k] = emb1 * scalar1
                
                # Second embedding lookup with position offset
                pos_idx = pos_ids[i] + 2  # Original logic: pos + 2
                if pos_idx < vocab_size2 and pos_idx >= 0:
                    emb_offset2 = pos_idx * embed_dim + col_start
                    for k in range(current_k):
                        if col_mask[k]:
                            emb2 = tl.load(weight2_ptr + emb_offset2 + k, mask=k < current_k, other=0.0)
                            chunk_result[i, k] += emb2
        
        # Store result chunk
        output_offset = offsets_row[:, None] * embed_dim + (col_start + offsets_col[:current_k])[None, :]
        tl.store(output_ptr + output_offset, chunk_result, mask=row_mask[:, None] and col_mask[None, :])

@torch.fx.wrap  
def fused_embedding_forward_optimized(input_ids, pos_ids, weight1, weight2):
    # Get dimensions
    batch_size = input_ids.shape[0]
    embed_dim = weight1.shape[1]
    input_size = input_ids.numel()
    
    # Create output tensor
    output = torch.empty((batch_size, embed_dim), dtype=weight1.dtype, device=input_ids.device)
    
    # Launch Triton kernel with optimized tile sizes
    BLOCK_SIZE_M = 32  # Process 32 rows simultaneously  
    BLOCK_SIZE_K = 128  # Vector size for memory coalescing
    num_programs = (input_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    fused_embedding_kernel_optimized[(num_programs,)](
        input_ids_ptr=input_ids,
        pos_ids_ptr=pos_ids,
        weight1_ptr=weight1,
        weight2_ptr=weight2,
        output_ptr=output,
        input_size=input_size,
        vocab_size1=weight1.shape[0],
        vocab_size2=weight2.shape[0],
        embed_dim=embed_dim,
        scalar1=16.0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return fused_embedding_forward_optimized