import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Simple pattern for matrix multiplication"""
    return in_1 @ in_0

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for optimized matmul"""
    return (in_0, in_1)

@triton.jit
def simple_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Simple matrix multiplication kernel"""
    # Get program ID
    pid = tl.program_id(0)
    total_threads = pid * BLOCK_SIZE
    
    # Simple 1D grid for small matrices
    if total_threads >= M * N:
        return
    
    # Calculate row and column
    row = total_threads // N
    col = total_threads % N
    
    if row >= M:
        return
    
    # Initialize accumulator
    acc = 0.0
    
    # Vectorized matrix multiplication
    for k in range(K):
        a_val = tl.load(a_ptr + row * K + k, mask=(row < M) & (k < K), other=0.0)
        b_val = tl.load(b_ptr + k * N + col, mask=(k < K) & (col < N), other=0.0)
        acc += a_val * b_val
    
    # Store result
    tl.store(c_ptr + row * N + col, acc, mask=(row < M) & (col < N))

@torch.fx.wrap
def simple_optimized_matmul(a, b):
    """Simple wrapper for optimized matrix multiplication"""
    # Handle batch dimensions
    if a.dim() > 2:
        # For batched matrices, reshape to 2D and process each batch
        batch_shape = a.shape[:-2]
        a_flat = a.reshape(-1, a.shape[-2], a.shape[-1])
        b_flat = b.reshape(-1, b.shape[-2], b.shape[-1])
        c_flat = torch.empty((a_flat.shape[0], a.shape[-2], b.shape[-1]), dtype=a.dtype, device=a.device)
        
        # Process each batch
        for i in range(a_flat.shape[0]):
            M, N = a_flat.shape[1], b_flat.shape[2]
            K = a_flat.shape[2]
            BLOCK_SIZE = 256
            num_threads = M * N
            num_programs = (num_threads + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            simple_matmul_kernel[(num_programs,)](
                a_flat[i], b_flat[i], c_flat[i],
                M, N, K, BLOCK_SIZE
            )
        
        return c_flat.reshape(*batch_shape, a.shape[-2], b.shape[-1])
    else:
        # Single matrix
        M, N = a.shape[0], b.shape[1]
        K = a.shape[1]
        c = torch.empty((M, N), dtype=a.dtype, device=a.device)
        
        BLOCK_SIZE = 256
        num_threads = M * N
        num_programs = (num_threads + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        simple_matmul_kernel[(num_programs,)](
            a, b, c, M, N, K, BLOCK_SIZE
        )
        
        return c

def replacement_func():
    """Return the optimized matmul function"""
    return simple_optimized_matmul