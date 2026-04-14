import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 16])
    tmp_2 = in_2.transpose(-1, -2)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Create offsets for loading
        a_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)[:, None]
        b_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)[None, None]
        
        # Create masks
        a_mask = a_offsets < M
        b_mask = b_offsets < N
        k_mask = k_offsets < K
        
        # Load data
        a = tl.load(a_ptr + a_offsets * K + k_offsets, mask=(a_mask & k_mask), other=0.0)
        b = tl.load(b_ptr + k_offsets * N + b_offsets, mask=(k_mask & b_mask), other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Store result
    c_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] * N + n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
    c_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)[:, None]) < M and (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]) < N
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)

@torch.fx.wrap
def optimized_matmul_reshape(in_0, in_1, in_2):
    # Get shapes
    in_1_shape = in_1.shape
    in_0_shape = in_0.shape
    
    # Get original dimensions
    M = in_1_shape[0]
    K = in_1_shape[2] if len(in_1_shape) > 2 else in_1_shape[1]
    N = in_0_shape[2] if len(in_0_shape) > 2 else in_0_shape[1]
    
    # Handle different tensor shapes
    if len(in_1_shape) == 3 and len(in_0_shape) == 3:
        # Case 1: (M, A, K) @ (M, K, N) -> (M, A, N)
        M, A, K = in_1_shape
        _, _, N = in_0_shape
        K_actual = K
    elif len(in_1_shape) == 3 and len(in_0_shape) == 2:
        # Case 2: (M, A, K) @ (M, K) -> (M, A)
        M, A, K = in_1_shape
        K_actual = K
        N = 1
    else:
        # Default matmul
        return pattern(in_0, in_1, in_2)
    
    # Create output tensor
    if N == 1:
        matmul_out = torch.empty((M, A), dtype=in_1.dtype, device=in_1.device)
    else:
        matmul_out = torch.empty((M, A, N), dtype=in_1.dtype, device=in_1.device)
    
    # Set block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N if N > 1 else 1
    
    # Launch kernel
    matmul_kernel[(grid_m, grid_n, 1)](
        in_1,
        in_0,
        matmul_out,
        M, N, K_actual,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    # Reshape to [-1, 16]
    if N == 1:
        # (M, A) -> (M*A, 1) -> reshape to [-1, 16]
        total_elements = M * A
        if total_elements % 16 == 0:
            reshaped = matmul_out.reshape(-1, 16)
        else:
            reshaped = matmul_out.reshape(-1, 1)  # Fallback
    else:
        # (M, A, N) ->.reshape to [-1, 16]
        total_elements = M * A * N
        if total_elements % 16 == 0:
            reshaped = matmul_out.reshape(-1, 16)
        else:
            reshaped = matmul_out.reshape(-1, 1)  # Fallback
    
    # Handle transpose
    transposed = in_2.transpose(-1, -2)
    
    return (reshaped, transposed)

def replacement_func():
    return optimized_matmul_reshape