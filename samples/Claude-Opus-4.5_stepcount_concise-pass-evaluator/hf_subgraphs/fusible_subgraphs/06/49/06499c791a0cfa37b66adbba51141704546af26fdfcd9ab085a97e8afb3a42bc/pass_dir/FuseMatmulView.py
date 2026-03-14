import torch
import triton
import triton.language as tl

# Pattern: torch.matmul(in_1, in_0)
# Batched matrix-vector multiplication
# in_0: [B, 1, K, 1] - batched column vectors
# in_1: [B, 1, M, K] - batched matrices
# result: [B, 1, M, 1]

def pattern(in_0, in_1):
    tmp_0 = torch.matmul(in_1, in_0)
    return tmp_0

def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def batched_matvec_v3(
    vec_ptr,    # [B * K] flattened vectors
    mat_ptr,    # [B * M * K] flattened matrices
    out_ptr,    # [B * M] output
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program handles one (batch, m_block).
    Uses vectorized loads and better memory access patterns.
    """
    # Grid: (num_m_blocks, B)
    m_block_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    
    # M indices for this block
    m_start = m_block_id * BLOCK_M
    m_offs = tl.arange(0, BLOCK_M)
    m_idx = m_start + m_offs
    m_mask = m_idx < M
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Base pointers
    vec_base = batch_id * K
    mat_base = batch_id * M * K
    
    # Process K in blocks
    for k_start in range(0, K, BLOCK_K):
        k_offs = tl.arange(0, BLOCK_K)
        k_idx = k_start + k_offs
        k_mask = k_idx < K
        
        # Load vector chunk [BLOCK_K] - broadcasted across all M rows
        v = tl.load(vec_ptr + vec_base + k_idx, mask=k_mask, other=0.0)
        
        # Load matrix block [BLOCK_M, BLOCK_K]
        # mat[batch, 0, m, k] at index: batch*M*K + m*K + k
        mat_ptrs = mat_base + m_idx[:, None] * K + k_idx[None, :]
        mat_mask = m_mask[:, None] & k_mask[None, :]
        m = tl.load(mat_ptr + mat_ptrs, mask=mat_mask, other=0.0)
        
        # Accumulate: element-wise multiply and sum over K
        acc += tl.sum(m * v[None, :], axis=1)
    
    # Store results
    out_ptrs = batch_id * M + m_idx
    tl.store(out_ptr + out_ptrs, acc, mask=m_mask)


@torch.fx.wrap
def fused_matmul(in_0, in_1):
    """
    Optimized batched matrix-vector multiplication.
    in_0: [B, 1, K, 1]
    in_1: [B, 1, M, K]
    Output: [B, 1, M, 1]
    """
    B = in_1.shape[0]
    M = in_1.shape[2]
    K = in_1.shape[3]
    
    in_0_flat = in_0.contiguous().view(-1)
    in_1_flat = in_1.contiguous().view(-1)
    
    out = torch.empty(B * M, device=in_0.device, dtype=in_0.dtype)
    
    # Choose block sizes based on problem dimensions
    # For M: use smaller blocks for smaller M
    if M <= 128:
        BLOCK_M = 32
    elif M <= 512:
        BLOCK_M = 64
    else:
        BLOCK_M = 128
    
    # For K: larger blocks for larger K to reduce loop iterations
    if K >= 2048:
        BLOCK_K = 256
    elif K >= 512:
        BLOCK_K = 128
    else:
        BLOCK_K = 64
    
    num_m_blocks = (M + BLOCK_M - 1) // BLOCK_M
    grid = (num_m_blocks, B)
    
    batched_matvec_v3[grid](
        in_0_flat,
        in_1_flat,
        out,
        M, K,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    
    return out.view(B, 1, M, 1)


def replacement_func():
    return fused_matmul