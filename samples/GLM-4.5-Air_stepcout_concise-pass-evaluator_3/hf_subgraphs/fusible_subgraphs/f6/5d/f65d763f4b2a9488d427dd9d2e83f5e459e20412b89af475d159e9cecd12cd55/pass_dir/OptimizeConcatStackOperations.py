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
    pid_k = tl.program_id(1) 
    pid_n = tl.program_id(2)
    
    # Calculate input and output block boundaries
    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create offset within block
    m_offset = tl.arange(0, BLOCK_SIZE_M)
    k_offset = tl.arange(0, BLOCK_SIZE_K) 
    n_offset = tl.arange(0, BLOCK_SIZE_N)
    
    # Reshape for inner loops (tiling optimization)
    m_offset = m_offset[:, None]  # Shape [BLOCK_SIZE_M, 1]
    k_offset = k_offset[None, :]  # Shape [1, BLOCK_SIZE_K]
    n_offset = n_offset[None, :]  # Shape [1, BLOCK_SIZE_N]
    
    # Calculate total offsets
    input_row = m_start + m_offset
    input_col = k_start + k_offset
    out_row = k_start + k_offset  # Transposed
    out_col = m_start + m_offset  # Transposed
    
    # Create masks
    row_mask = input_row < M
    col_mask = input_col < K
    
    # Load input block (optimized for memory coalescing)
    input_ptrs = input_ptr + (input_row * K + input_col)
    input_vals = tl.load(input_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    
    # Store output block with transposed indices
    output_ptrs = output_ptr + (out_row * N + out_col)
    tl.store(output_ptrs, input_vals, mask=row_mask[:, None] & col_mask[None, :])

@torch.fx.wrap
def optimized_transpose_2d_last_dims(x):
    """Optimized transpose operation for (-1, -2) with better tiling"""
    M, K, N = x.shape
    
    # Optimal tile sizes for typical GPU architectures
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 64
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K  
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty((K, M, N), dtype=x.dtype, device=x.device)
    
    # Launch kernel with 3D grid for optimal GPU utilization
    optimized_transpose_kernel[
        (grid_m, grid_k, grid_n)
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