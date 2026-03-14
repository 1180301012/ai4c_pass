import torch
import triton
import triton.language as tl

@triton.jit
def optimized_transpose_kernel(
    input_ptr,
    output_ptr, 
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized kernel for transpose(-1, -2) with better memory access patterns"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1) 
    pid_k = tl.program_id(2)
    
    # Calculate block boundaries
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_K
    
    # Create 1D offsets
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
    
    # Create coordinate grids
    m_coords = m_offsets[:, None, None]  # [BLOCK_SIZE_M, 1, 1]
    n_coords = n_offsets[None, :, None]  # [1, BLOCK_SIZE_N, 1]  
    k_coords = k_offsets[None, None, :]  # [1, 1, BLOCK_SIZE_K]
    
    # Create masks
    m_mask = m_coords < M
    n_mask = n_coords < N
    k_mask = k_coords < K
    mask = m_mask & n_mask & k_mask
    
    # Calculate input linear indices: (m, k, n) -> m*K*N + k*N + n
    input_indices = (m_coords * K * N + k_coords * N + n_coords)
    input_ptrs = input_ptr + input_indices
    
    # Calculate output linear indices: (m, n, k) -> m*N*K + n*K + k (transposed)
    output_indices = (m_coords * N * K + n_coords * K + k_coords)
    output_ptrs = output_ptr + output_indices
    
    # Load input values and store to transposed positions
    input_vals = tl.load(input_ptrs, mask=mask, other=0.0)
    tl.store(output_ptrs, input_vals, mask=mask)

@torch.fx.wrap
def optimized_transpose_2d_last_dims(x):
    """Optimized transpose operation for (-1, -2) with better tiling"""
    M, K, N = x.shape
    
    # input shape: [M, K, N], output shape should be [M, N, K] for transpose(-1, -2)
    out_shape = (M, N, K)
    
    # Block sizes optimized for typical GPU architectures
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64  # We'll tile over the dimensions being transposed
    BLOCK_SIZE_K = K   # Process full K dimension per thread
    
    # Calculate grid dimensions (2D grid for M and N)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = 1  # Single block for K dimension
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel with 2D grid
    optimized_transpose_kernel[
        (grid_m, grid_n, grid_k)
    ](
        input_ptr=x,
        output_ptr=out,
        M=M, K=K, N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def pattern(x):
    """Pattern to match transpose(-1, -2) operation"""
    out = x.transpose(-1, -2)
    return out

def replacement_args(x):
    """Extract arguments for the optimized transpose"""
    return (x,)

def replacement_func():
    """Return the optimized transpose function"""
    return optimized_transpose_2d_last_dims