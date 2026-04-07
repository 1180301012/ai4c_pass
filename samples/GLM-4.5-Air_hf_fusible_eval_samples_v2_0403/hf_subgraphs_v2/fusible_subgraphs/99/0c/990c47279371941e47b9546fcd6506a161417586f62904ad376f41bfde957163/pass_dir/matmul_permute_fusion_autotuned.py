import torch
import triton
import triton.language as tl

def pattern(a, b):
    # a has shape [batch, seq_len, num_queries]
    # b has shape [batch, num_queries, hidden_dim]
    matmul = torch.matmul(a, b)
    result = matmul.permute(0, 2, 1)
    return result

def replacement_args(a, b):
    return (a, b)

@triton.jit
def matmul_transpose_kernel_autotuned(
    a_ptr, b_ptr, out_ptr,
    batch, seq_len, num_queries, hidden_dim,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Optimized kernel with autotuned block sizes
    batch_id = tl.program_id(0)
    m_id = tl.program_id(1)  # hidden dimension tile index
    n_id = tl.program_id(2)  # seq dimension tile index
    
    # Tile offsets
    m_offsets = m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Mask for out-of-bound accesses
    m_mask = m_offsets < hidden_dim
    n_mask = n_offsets < seq_len
    k_mask = k_offsets < num_queries
    
    # Initialize accumulator for the tile
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute base offsets for this batch
    a_base = batch_id * seq_len * num_queries
    b_base = batch_id * num_queries * hidden_dim
    out_base = batch_id * hidden_dim * seq_len
    
    # Main loop over K dimension (queries) - unroll by 4 for better performance
    for k in range(0, num_queries, BLOCK_SIZE_K * 4):
        # Load 4 tiles at once for loop unrolling
        for unroll in range(4):
            k_start = k + unroll * BLOCK_SIZE_K + k_offsets
            if k_start >= num_queries:
                break
            k_valid = k_start < num_queries
            
            # Load matrix A: [seq_len, num_queries]
            a_ptrs = a_base + n_offsets[:, None] * num_queries + k_start[None, :]
            a_vals = tl.load(a_ptrs, mask=n_mask[:, None] & k_valid[None, :], other=0.0)
            
            # Load matrix B: [num_queries, hidden_dim]
            b_ptrs = b_base + k_start[:, None] * hidden_dim + m_offsets[None, :]
            b_vals = tl.load(b_ptrs, mask=k_valid[:, None] & m_mask[None, :], other=0.0)
            
            # Matrix multiply: gemm
            accumulator += tl.dot(a_vals, b_vals, acc_type=tl.float32)
    
    # Store result to [hidden_dim, seq_len] layout
    out_ptrs = out_base + m_offsets[:, None] * seq_len + n_offsets[None, :]
    tl.store(out_ptrs, accumulator.to(out_ptr.type.element_ty), mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def matmul_with_transpose_autotuned(a, b):
    batch, seq_len, num_queries = a.shape
    _, _, hidden_dim = b.shape
    
    # Try different block sizes based on tensor characteristics
    if hidden_dim >= 512 and seq_len >= 512:
        # Large tensors: use bigger blocks
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
    elif hidden_dim >= 256 and seq_len >= 256:
        # Medium tensors: medium blocks
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 16
    else:
        # Small tensors: smaller blocks
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16  
        BLOCK_SIZE_K = 8
    
    # Calculate grid dimensions
    grid_m = (hidden_dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size = (batch, grid_m, grid_n)
    
    # Create output tensor with transposed shape [batch, hidden_dim, seq_len]
    output = torch.empty((batch, hidden_dim, seq_len), dtype=a.dtype, device=a.device)
    
    # Launch kernel
    matmul_transpose_kernel_autotuned[grid_size](
        a, b, output,
        batch, seq_len, num_queries, hidden_dim,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return matmul_with_transpose_autotuned