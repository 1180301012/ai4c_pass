import torch
import triton
import triton.language as tl

# Pattern matching function - matches just the matmul operation
def pattern(in_2, in_3):
    """Matches the matmul operation: torch.matmul(in_2, in_3)"""
    return torch.matmul(in_2, in_3)

# Argument extraction function
def replacement_args(in_2, in_3):
    """Extract the arguments needed for the matmul operation"""
    return (in_2, in_3)

# Optimized Triton kernel for small matrix multiplication (hidden_size x 1 output)
@triton.jit
def optimized_matmul_kernel(
    x_ptr,  # [M, K] input tensor
    y_ptr,  # [K, N] input tensor  
    out_ptr,  # [M, N] output tensor
    M,  # batch size
    K,  # hidden dimension
    N,  # output dimension (usually 1 for our case)
    dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized Triton kernel for matrix multiplication with small output dimension"""
    # Each program handles one tile of the output matrix
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create offsets within the block
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for bounds checking
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    k_mask = k_offsets < K
    
    # Initialize accumulator - use float32 for precision
    if N == 1:
        accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, K)
        
        # Load x tile [BLOCK_SIZE_M, BLOCK_SIZE_K]
        x_offsets = m_offsets[:, None] * K + k_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load y tile [BLOCK_SIZE_K, BLOCK_SIZE_N]
        y_offsets = k_offsets[:, None] * N + n_offsets[None, :]
        y = tl.load(y_ptr + y_offsets, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Matrix multiplication and accumulate - use float32 for better precision
        if N == 1:
            # Specialized path for N=1 (vector output) - convert to float32 for precision
            x_f32 = x.to(tl.float32)
            y_f32 = y.to(tl.float32).reshape(BLOCK_SIZE_K)
            accumulator += tl.sum(x_f32 * y_f32[None, :], axis=1)
        else:
            # General matrix multiplication - convert to float32 for precision
            x_f32 = x.to(tl.float32)
            y_f32 = y.to(tl.float32)
            accumulator += tl.sum(x_f32 * y_f32, axis=[2])  # Sum over the K dimension
    
    # Store result
    if N == 1:
        # Store as [BLOCK_SIZE_M, 1]
        out_offsets = m_offsets[:, None] * N
        tl.store(out_ptr + out_offsets, accumulator[:, None].to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        # Store as [BLOCK_SIZE_M, BLOCK_SIZE_N]
        out_offsets = m_offsets[:, None] * N + n_offsets[None, :]
        tl.store(out_ptr + out_offsets, accumulator.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])



# Kernel wrapper function for just the matrix multiplication
def optimized_matmul(x, y):
    """
    Optimized matrix multiplication wrapper that handles different data types
    and launches the appropriate Triton kernel
    """
    # Get tensor shapes and data types
    M, K = x.shape
    N = y.shape[1]
    dtype = x.dtype
    
    # Determine optimal block sizes for small matrix multiplication
    # For M x K @ K x 1 operations, use small blocks to minimize overhead
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 64
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty((M, N), dtype=dtype, device=x.device)
    
    # Launch the kernel with the data type
    optimized_matmul_kernel[grid_m, grid_n, BLOCK_SIZE_K](
        x, y, out, M, K, N,
        dtype, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return out

# Replacement function - returns the optimized matmul function
def replacement_func():
    return optimized_matmul