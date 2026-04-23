import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_1, in_0):
    matmul = in_1 @ in_0
    return matmul

# Argument extraction function
def replacement_args(in_1, in_0):
    return (in_1, in_0)

# Triton kernel for batched matrix multiplication
@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    batch_size, c1, m, n, k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Calculate global ids
    batch_id = tl.program_id(0)
    m_id = tl.program_id(1)
    n_id = tl.program_id(2)
    
    # Calculate offsets for the current tile
    m_offset = m_id * BLOCK_M
    n_offset = n_id * BLOCK_N
    
    # Create indices for the current tile
    m_idx = m_offset + tl.arange(0, BLOCK_M)
    n_idx = n_offset + tl.arange(0, BLOCK_N)
    
    # Create masks for boundary conditions
    mask_m = m_idx < m
    mask_n = n_idx < n
    
    # Allocate shared memory for the tile
    a_shared = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    b_shared = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over the K dimension in blocks
    for k_start in range(0, k, BLOCK_K):
        # Load A from global memory to shared memory
        a = tl.load(
            a_ptr + batch_id * c1 * m * k + channel_id * m * k + m_idx[:, None] * k + k_start,
            mask=(mask_m[:, None] & (k_start < k)),
            other=0.0
        )
        tl.store(
            a_shared + m_idx[:, None] * BLOCK_K + tl.arange(0, BLOCK_K),
            a,
            mask=(mask_m[:, None] & (k_start < k))
        )
        
        # Load B from global memory to shared memory
        b = tl.load(
            b_ptr + batch_id * k * n + k_start * n + n_idx[None, :],
            mask=((k_start < k) & mask_n[None, :]),
            other=0.0
        )
        tl.store(
            b_shared + tl.arange(0, BLOCK_K)[:, None] * BLOCK_N + n_idx,
            b,
            mask=((k_start < k) & mask_n[None, :])
        )
        
        # Wait for the shared memory loads to complete
        tl.debug_barrier()
        
        # Calculate inner product
        acc += tl.dot(a_shared, b_shared)
        
        # Wait for the shared memory to be ready for the next iteration
        tl.debug_barrier()
    
    # Convert to float16 and store
    c = acc.to(tl.float16)
    tl.store(
        c_ptr + batch_id * m * n + m_idx[:, None] * n + n_idx[None, :],
        c,
        mask=(mask_m[:, None] & mask_n[None, :])
    )

# Batched matmul wrapper
@torch.fx.wrap
def batched_matmul(in_1, in_0):
    # Combine batch dimensions
    batch_size, c1, m, k = in_1.shape
    _, c2, _, n = in_0.shape
    
    # Verify batch dimensions match
    if c1 != c2:
        raise ValueError("Batch dimensions do not match")
    
    # Combine batch dimensions
    batch_size_combined = batch_size * c1
    
    # Reshape for 3D matrix multiplication
    a = in_1
    b = in_0
    
    # Create output tensor
    c = torch.empty((batch_size_combined, m, n), dtype=in_1.dtype, device=in_1.device)
    
    # Configure block sizes (tuned for common GPUs)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    # Calculate grid size
    grid_m = (m + BLOCK_M - 1) // BLOCK_M
    grid_n = (n + BLOCK_N - 1) // BLOCK_N
    grid = (batch_size, c1, grid_m, grid_n)
    
    # Launch kernel
    batched_matmul_kernel[grid](
        a, b, c,
        batch_size, c1, m, n, k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    # Reshape back to original shape
    c = c.view(batch_size, c1, m, n)
    return c

# Replacement function

def replacement_func():
    return batched_matmul