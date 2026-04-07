import torch
import triton
import triton.language as tl

def pattern(a, b):
    """Pattern matching: matmul operation"""
    return torch.matmul(a, b)

def replacement_args(a, b):
    """Extract arguments for the replacement"""
    return (a, b)

@triton.jit
def triton_matmul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Matrix multiplication kernel for M x K @ K x N -> M x N
    pid = tl.program_id(0)
    i = (pid // ((N + BLOCK_SIZE - 1) // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j = (pid % ((N + BLOCK_SIZE - 1) // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE):
        x = tl.load(x_ptr + i[:, None] * K + k + tl.arange(0, BLOCK_SIZE)[None, :], 
                   mask=(i[:, None] < M) & (k + tl.arange(0, BLOCK_SIZE)[None, :] < K),
                   other=0.0)
        y = tl.load(y_ptr + (k + tl.arange(0, BLOCK_SIZE))[:, None] * N + j[None, :],
                   mask=(k + tl.arange(0, BLOCK_SIZE)[:, None] < K) & (j[None, :] < N),
                   other=0.0)
        accumulator += tl.dot(x, y)
    
    tl.store(out_ptr + (i[:, None] * N + j[None, :]), accumulator,
             mask=(i[:, None] < M) & (j[None, :] < N))

@torch.fx.wrap
def triton_matmul(a, b):
    """Optimized matrix multiplication using Triton"""
    M, K = a.shape
    K2, N = b.shape
    
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_SIZE = 32  # Optimal block size for matrix multiplication
    grid_x = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_x * grid_y,)
    
    triton_matmul_kernel[grid](
        x_ptr=a,
        y_ptr=b,
        out_ptr=out,
        M=M, N=N, K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized kernel wrapper function"""
    return triton_matmul