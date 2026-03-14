import torch
import triton
import triton.language as tl

# Pattern matching function - match only the matmul operation
def pattern(in_0, in_1, in_2):
    """Pattern matches only the matrix multiplication operation"""
    tmp_0 = in_1 @ in_0
    return tmp_0

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """Extract arguments for matmul optimization"""
    return (in_0, in_1)

@triton.jit
def simple_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Simple Triton kernel for matrix multiplication"""
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    m_offset = pid // grid_n
    n_offset = pid % grid_n
    
    m_begin = m_offset * BLOCK_SIZE_M
    m_end = min((m_offset + 1) * BLOCK_SIZE_M, M)
    n_begin = n_offset * BLOCK_SIZE_N
    n_end = min((n_offset + 1) * BLOCK_SIZE_N, N)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, K)
        
        a_offset = m_begin * K + k
        b_offset = k * N + n_begin
        
        a_block = tl.load(a_ptr + a_offset, mask=(tl.arange(BLOCK_SIZE_M)[:, None] < (m_end - m_begin)) & (tl.arange(BLOCK_SIZE_K) < (k_end - k)), other=0.0)
        b_block = tl.load(b_ptr + b_offset, mask=(tl.arange(BLOCK_SIZE_K)[:, None] < (k_end - k)) & (tl.arange(BLOCK_SIZE_N) < (n_end - n_begin)), other=0.0)
        
        accumulator += tl.dot(a_block, b_block)
    
    c_offset = m_begin * N + n_begin
    c_mask = (tl.arange(BLOCK_SIZE_M)[:, None] < (m_end - m_begin)) & (tl.arange(BLOCK_SIZE_N) < (n_end - n_begin))
    tl.store(c_ptr + c_offset, accumulator, mask=c_mask)

@torch.fx.wrap
def simple_matmul(a, b):
    """Simple optimized matrix multiplication using Triton"""
    M, N = a.shape[0], b.shape[1]
    K = a.shape[1]
    
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64  
    BLOCK_SIZE_K = 32
    
    grid_size = (M * N + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    
    simple_matmul_kernel[grid_size](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return simple_matmul