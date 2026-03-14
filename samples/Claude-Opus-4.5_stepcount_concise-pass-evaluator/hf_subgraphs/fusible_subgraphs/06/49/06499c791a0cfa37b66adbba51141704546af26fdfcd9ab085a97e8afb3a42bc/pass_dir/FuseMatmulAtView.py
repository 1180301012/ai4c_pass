import torch
import triton
import triton.language as tl

# Pattern: in_1 @ in_0
# This handles the yolo11n cases which use @ operator instead of torch.matmul
# in_0: [B, H, K, N] 
# in_1: [B, H, M, K] 
# result of matmul: [B, H, M, N]

def pattern(in_0, in_1):
    tmp_0 = in_1 @ in_0
    return tmp_0

def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def batched_matmul_kernel(
    in_0_ptr,  # [BH * K * N] flattened
    in_1_ptr,  # [BH * M * K] flattened
    out_ptr,   # [BH * M * N] flattened
    M,
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (num_m_blocks * num_n_blocks, BH)
    pid_mn = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    m_block_id = pid_mn // num_n_blocks
    n_block_id = pid_mn % num_n_blocks
    
    m_start = m_block_id * BLOCK_M
    n_start = n_block_id * BLOCK_N
    
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    
    m_indices = m_start + m_offsets
    n_indices = n_start + n_offsets
    
    m_mask = m_indices < M
    n_mask = n_indices < N
    
    # Accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Base pointers
    in_1_base = pid_bh * M * K
    in_0_base = pid_bh * K * N
    
    # Loop over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = tl.arange(0, BLOCK_K)
        k_indices = k_start + k_offsets
        k_mask = k_indices < K
        
        # Load in_1 block [BLOCK_M, BLOCK_K]
        in_1_ptrs = in_1_base + m_indices[:, None] * K + k_indices[None, :]
        in_1_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(in_1_ptr + in_1_ptrs, mask=in_1_mask, other=0.0)
        
        # Load in_0 block [BLOCK_K, BLOCK_N]
        in_0_ptrs = in_0_base + k_indices[:, None] * N + n_indices[None, :]
        in_0_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(in_0_ptr + in_0_ptrs, mask=in_0_mask, other=0.0)
        
        # Matmul
        acc += tl.dot(a, b)
    
    # Store result
    out_base = pid_bh * M * N
    out_ptrs = out_base + m_indices[:, None] * N + n_indices[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def fused_matmul_at(in_0, in_1):
    """
    Batched matrix multiplication for @ operator using Triton.
    in_0: [B, H, K, N] 
    in_1: [B, H, M, K]
    result: [B, H, M, N]
    """
    B = in_1.shape[0]
    H = in_1.shape[1]
    M = in_1.shape[2]
    K = in_1.shape[3]
    N = in_0.shape[3]
    
    BH = B * H
    
    in_0_flat = in_0.contiguous().view(-1)
    in_1_flat = in_1.contiguous().view(-1)
    
    out = torch.empty(BH * M * N, device=in_0.device, dtype=in_0.dtype)
    
    # Choose block sizes - must be powers of 2
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    num_m_blocks = (M + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    
    grid = (num_m_blocks * num_n_blocks, BH)
    
    batched_matmul_kernel[grid](
        in_0_flat,
        in_1_flat,
        out,
        M, K, N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out.view(B, H, M, N)


def replacement_func():
    return fused_matmul_at