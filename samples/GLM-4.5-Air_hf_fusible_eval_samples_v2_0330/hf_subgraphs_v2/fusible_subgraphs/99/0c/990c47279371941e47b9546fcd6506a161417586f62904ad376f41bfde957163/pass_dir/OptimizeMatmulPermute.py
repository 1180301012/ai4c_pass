import torch
import triton
import triton.language as tl

def pattern(tmp_1, in_1):
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3

def replacement_args(tmp_1, in_1):
    return (tmp_1, in_1)

@triton.jit
def optimized_matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    m,
    k,
    n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Grid setup - each block handles M x N output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    # Create offsets
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension
    for k_idx in range(0, k, BLOCK_SIZE_K):
        # Calculate global offsets
        a_offsets = (batch_idx * m * k + 
                    (m_offsets[:, None] * k) + 
                    (k_offsets[None, :] + k_idx))
        
        b_offsets = (batch_idx * k * n + 
                    ((k_offsets[:, None] + k_idx) * n) + 
                    n_offsets[None, :])
        
        # Load data with proper masking
        a_mask = (m_offsets[:, None] < m) & (k_offsets[None, :] + k_idx < k)
        b_mask = (k_offsets[:, None] + k_idx < k) & (n_offsets[None, :] < n)
        
        a_block = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        b_block = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Matrix multiply and accumulate
        accumulator += tl.dot(a_block, b_block)
    
    # Store result
    out_offsets = (batch_idx * n * m + 
                  (n_offsets[:, None] * m) + 
                  m_offsets[None, :])
    
    out_mask = (m_offsets[None, :] < m) & (n_offsets[:, None] < n)
    tl.store(out_ptr + out_offsets, accumulator, mask=out_mask)

@torch.fx.wrap
def optimized_matmul_permute(a, b):
    batch_size, m, k = a.shape
    _, k2, n = b.shape
    
    assert k == k2, f"Dimension mismatch: {k} != {k2}"
    
    output = torch.empty((batch_size, n, m), dtype=a.dtype, device=a.device)
    
    # Block sizes for better GPU utilization
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 64 
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    optimized_matmul_kernel[(grid_m, grid_n, batch_size)](
        a_ptr=a,
        b_ptr=b,  # Use original layout for matmul
        out_ptr=output,
        batch_size=batch_size,
        m=m,
        k=k,
        n=n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return optimized_matmul_permute