import torch
import triton
import triton.language as tl

def pattern(matmul_input, weight, divisor):
    """
    Pattern to match: matmul followed by division by constant
    This optimizes the sequence matmul(...), result / divisor
    """
    matmul_result = torch.matmul(matmul_input, weight)
    div_result = matmul_result / divisor
    return div_result

def replacement_args(matmul_input, weight, divisor):
    """
    Extract arguments: matmul_input, weight, divisor
    """
    return (matmul_input, weight, divisor)

@triton.jit
def fused_matmul_div_kernel(
    query_ptr, key_ptr, output_ptr,
    query_batch, query_heads, query_seq, query_dim,
    key_batch, key_heads, key_seq, key_dim,
    divisor,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Triton kernel that fuses matmul and division operations
    Optimized for transformer attention patterns
    """
    # Program IDs for block-level parallelism
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block ranges
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    grid_m = (query_seq + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (key_seq + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Only execute if within bounds
    if pid_m >= grid_m or pid_n >= grid_n:
        return
    
    # Row offsets for output
    row_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    # Column offsets for keys
    col_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for out-of-bounds accesses
    row_mask = row_offsets < query_seq
    col_mask = col_offsets < key_seq
    
    # Load divisor once and convert to appropriate dtype
    divisor_val = tl.load(divisor)
    
    # Initialize accumulator for the block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension
    for k in range(0, query_dim, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, query_dim)
        
        # Load current block of query
        query_ptrs = query_ptr + (
            (row_offsets[:, None] * query_heads * query_dim + 
             tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.int32)) +
            (tl.zeros((1, BLOCK_SIZE_N), dtype=tl.int32) // query_heads * query_dim +
             k)
        ).to(tl.int64)
        
        query_block = tl.load(query_ptrs, mask=row_mask[:, None], other=0.0)
        
        # Load current block of key
        key_ptrs = key_ptr + (
            (tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.int32) // key_heads * key_seq * key_dim +
             col_offsets[None, :] * key_dim +
             k)
        ).to(tl.int64)
        
        key_block = tl.load(key_ptrs, mask=col_mask[None, :], other=0.0)
        
        # Matrix multiplication for current block
        accumulator += tl.dot(query_block, key_block.to(tl.float32))
    
    # Apply division and store result
    output_ptrs = output_ptr + (
        (row_offsets[:, None] * query_heads * key_seq + 
         tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.int32) // query_heads * key_seq +
         col_offsets[None, :])
    ).to(tl.int64)
    
    result = accumulator / divisor_val
    tl.store(output_ptrs, result, mask=row_mask[:, None] & col_mask[None, :])

@torch.fx.wrap
def fused_matmul_div_forward(query, key, divisor):
    """
    Wrapper function for the fused matmul+div operation
    """
    # Get tensor shapes and metadata
    B, H, S_Q, D_Q = query.shape
    B, H, S_K, D_K = key.shape
    
    # Validate shapes for matmul
    if D_Q != D_K:
        raise ValueError(f"Dimension mismatch for matmul: query_dim={D_Q}, key_dim={D_K}")
    
    # Create output tensor
    output = torch.empty(B, H, S_Q, S_K, dtype=query.dtype, device=query.device)
    
    # Launch Triton kernel
    BLOCK_SIZE_M = 32  # Sequence dimension for query
    BLOCK_SIZE_N = 32  # Sequence dimension for key  
    BLOCK_SIZE_K = 32  # Feature dimension
    
    # Handle different data types
    if query.dtype == torch.float16:
        divisor_ptr = torch.tensor(divisor, dtype=torch.float16, device=query.device).contiguous()
    elif query.dtype == torch.bfloat16:
        divisor_ptr = torch.tensor(divisor, dtype=torch.bfloat16, device=query.device).contiguous()
    else:
        divisor_ptr = torch.tensor(divisor, dtype=torch.float32, device=query.device).contiguous()
    
    grid = (
        (S_Q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (S_K + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )
    
    fused_matmul_div_kernel[grid](
        query, key, output,
        B, H, S_Q, D_Q,
        B, H, S_K, D_K,
        divisor_ptr,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    """
    Returns the fused matmul+div kernel function
    """
    return fused_matmul_div_forward