import torch
import triton
import triton.language as tl

def pattern(a, b):
    """Match the matmul operation: [2, 1152] @ [1152, 1] -> [2, 1]"""
    return torch.matmul(a, b)

def replacement_args(a, b):
    """Extract arguments for the replacement kernel"""
    return (a, b)

@triton.jit
def matmul_kernel_2_1152_1(
    a_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """High-performance matmul kernel for M=2, N=1, K=1152"""
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    m = pid // grid_n
    n = pid % grid_n
    
    if m >= M or n >= N:
        return
    
    # Offset pointers
    a_ptr += m * K
    b_ptr += n * K
    out_ptr += m * N + n
    
    # Allocate shared memory
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Process in tiles of BLOCK_SIZE_K
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tile from A and B
        a_tile = tl.load(a_ptr + k + tl.arange(0, BLOCK_SIZE_K), 
                        mask=(k + tl.arange(0, BLOCK_SIZE_K)) < K,
                        other=0.0)
        b_tile = tl.load(b_ptr + k * N + tl.arange(0, BLOCK_SIZE_N) * K, 
                        mask=(tl.arange(0, BLOCK_SIZE_N) * K) < N * K and (k + tl.arange(0, BLOCK_SIZE_K)) < K,
                        other=0.0).to(tl.float32)
        
        # Accumulate matrix multiply - manually compute outer product for small matrices
        for i in range(len(a_tile)):
            accumulator[i, 0] += a_tile[i] * b_tile[0]
    
    # Store the result
    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE_N) * M, accumulator[:, 0], 
             mask=(tl.arange(0, BLOCK_SIZE_N) * M + m) < M * N)

@torch.fx.wrap
def optimized_matmul_2_1152_1(a, b):
    """Wrapper for optimized matmul with specific dimensions [2, 1152] @ [1152, 1]"""
    M, K = a.shape
    N = b.shape[1]
    
    # Use small block sizes for this specific small matrix
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 32
    
    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Special case for small dimensions - use simpler tiling
    grid_size = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    matmul_kernel_2_1152_1[grid_size](
        a_ptr=a,
        b_ptr=b,
        out_ptr=output,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return optimized_matmul_2_1152_1