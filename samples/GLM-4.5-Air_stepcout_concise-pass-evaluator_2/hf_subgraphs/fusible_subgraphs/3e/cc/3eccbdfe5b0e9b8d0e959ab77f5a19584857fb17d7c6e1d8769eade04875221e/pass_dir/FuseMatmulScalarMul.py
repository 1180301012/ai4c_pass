import torch
import triton
import triton.language as tl

def pattern(a, b):
    # Match just the matmul operation for optimal performance
    result = torch.matmul(a, b)
    return result

def replacement_args(a, b):
    return (a, b)

# Triton kernel for fused matmul + scalar multiplication
@triton.jit
def fused_matmul_scalar_kernel(
    a_ptr,      # in_2 matrix
    b_ptr,      # in_1 matrix  
    scalar,     # in_0 scalar
    out_ptr,    # result matrix
    m,
    n,
    k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix multiplication with scalar fusion: C = scalar * (A @ B)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges for each program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    m_end = min(m_start + BLOCK_SIZE_M, m)
    n_end = min(n_start + BLOCK_SIZE_N, n)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension
    for k_start in range(0, k, BLOCK_SIZE_K):
        k_end = min(k_start + BLOCK_SIZE_K, k)
        
        # Load matrices A and B with bounds checking
        a = tl.load(a_ptr + (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * k + (k_start + tl.arange(0, BLOCK_SIZE_K))[None, :],
                    mask=(m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] < m and (k_start + tl.arange(0, BLOCK_SIZE_K))[None, :] < k,
                    other=0.0)
        b = tl.load(b_ptr + (k_start + tl.arange(0, BLOCK_SIZE_K))[:, None] * n + (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :],
                    mask=(k_start + tl.arange(0, BLOCK_SIZE_K))[:, None] < k and (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] < n,
                    other=0.0)
        
        # Matrix multiplication and accumulate
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
    
    # Apply scalar multiplication
    accumulator = accumulator * scalar
    
    # Store result
    accumulator = accumulator[:m_end - m_start, :n_end - n_start]
    tl.store(out_ptr + (m_start + tl.arange(0, m_end - m_start))[:, None] * n + (n_start + tl.arange(0, n_end - n_start))[None, :],
             accumulator)

@torch.fx.wrap
def simple_matmul_optimized(a, b):
    """Matrix multiplication specifically optimized for small matrices using Triton"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Dimension mismatch for matrix multiplication"
    
    # For very small matrices, use working Triton kernel
    result = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    # For N == 1 (vector output), use simple kernel
    if N == 1:
        @triton.jit
        def working_vector_kernel(
            a_ptr, b_ptr, out_ptr, m, k
        ):
            # Simple kernel that works for small matrices
            for i in range(m):
                sum_val = 0.0
                for k_idx in range(k):
                    sum_val += tl.load(a_ptr + i * k + k_idx) * tl.load(b_ptr + k_idx)
                tl.store(out_ptr + i, sum_val)
        
        working_vector_kernel[(M,)](
            a_ptr=a, b_ptr=b, out_ptr=result, m=M, k=K
        )
    else:
        # Small matrix kernel for general case
        @triton.jit
        def small_matrix_kernel(
            a_ptr, b_ptr, out_ptr, m, n, k,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            
            m_start = pid_m * BLOCK_SIZE_M
            n_start = pid_n * BLOCK_SIZE_N
            m_end = min(m_start + BLOCK_SIZE_M, m)
            n_end = min(n_start + BLOCK_SIZE_N, n)
            
            if m_start >= m or n_start >= n:
                return
                
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            
            for k_idx in range(k):
                a_vals = tl.load(a_ptr + (m_start + tl.arange(0, m_end - m_start)) * k + k_idx)
                b_vals = tl.load(b_ptr + k_idx * n + (n_start + tl.arange(0, n_end - n_start)))
                
                for i in range(m_end - m_start):
                    for j in range(n_end - n_start):
                        accumulator[i, j] += a_vals[i] * b_vals[j]
            
            tl.store(out_ptr + (m_start + tl.arange(0, m_end - m_start)) * n + (n_start + tl.arange(0, n_end - n_start)),
                    accumulator)
        
        BLOCK_SIZE_M = min(16, M)
        BLOCK_SIZE_N = min(16, N)
        grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        
        small_matrix_kernel[(grid_m, grid_n)](
            a_ptr=a, b_ptr=b, out_ptr=result,
            m=M, n=N, k=K,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
        )
    
    return result

def replacement_func():
    return simple_matmul_optimized