import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_2, in_3):
    """
    Match exactly: matmul = torch.matmul(in_2, in_3)
    """
    return torch.matmul(in_2, in_3)

# Argument extraction function

def replacement_args(in_2, in_3):
    """
    Extract inputs needed for the optimized kernel
    """
    return (in_2, in_3)

# Triton kernel optimized for small [2, 768] x [768, 1] matrices
@triton.jit
def triton_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    block_M: tl.constexpr, block_N: tl.constexpr
):
    # Matrix dimensions: A[M, K], B[K, N], C[M, N]
    # One block processes block_M x block_N elements
    row = tl.program_id(0) * block_M + tl.arange(0, block_M)
    col = tl.program_id(1) * block_N + tl.arange(0, block_N)
    
    # Create mask for valid elements
    mask = (row[:, None] < M) & (col[None, :] < N)
    
    # Initialize accumulator
    acc = tl.zeros((block_M, block_N), dtype=tl.float32)
    
    # Compute dot product using vectorized K-loop
    for k in range(0, K, 1):
        a = tl.load(a_ptr + row[:, None] * K + k, mask=mask)
        b = tl.load(b_ptr + k * N + col, mask=mask)
        acc += a * b
    
    # Convert to output dtype and store
    c = acc.to(tl.float16) if c_ptr.dtype == tl.float16 else acc
    tl.store(c_ptr + row[:, None] * N + col, c, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def matmul_kernel_wrapper(in_2, in_3):
    # Dimensions from model: [2, 768] x [768, 1] → [2, 1]
    M, K = in_2.shape
    N = in_3.shape[1]
    
    # Create output tensor
    out = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)
    
    # Block size: optimized for [2, 1] output
    block_M, block_N = 2, 1
    grid_M = (M + block_M - 1) // block_M
    grid_N = (N + block_N - 1) // block_N
    
    # Launch kernel
    triton_matmul_kernel[(grid_M, grid_N)](
        in_2, in_3, out,
        M, N, K,
        block_M, block_N
    )
    
    return out

# Replacement function

def replacement_func():
    return matmul_kernel_wrapper