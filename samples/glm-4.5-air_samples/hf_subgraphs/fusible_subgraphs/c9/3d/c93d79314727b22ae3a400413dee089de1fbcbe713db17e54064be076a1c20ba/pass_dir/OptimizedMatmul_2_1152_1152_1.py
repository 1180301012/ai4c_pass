import torch
import triton
import triton.language as tl
import math

# Pattern matching function - matches the matmul operation
def pattern(a, b):
    # This matches the matmul operation in the original computation
    result = torch.matmul(a, b)
    return result

# Argument extraction function
def replacement_args(a, b):
    return (a, b)

# Optimized Triton kernel with better memory access patterns
@triton.jit
def matmul_kernel_2_1152_1(
    a_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,  # 2
    K: tl.constexpr,  # 1152  
    N: tl.constexpr,  # 1
):
    pid_m = tl.program_id(0)
    
    if pid_m >= M:
        return
    
    accum = 0.0
    
    # Use vectorized memory access with larger chunks for better throughput
    # Process in chunks of 64 elements for better GPU utilization
    for k_base in range(0, K, 64):
        # Process up to 64 elements per chunk
        for i in range(64):
            k = k_base + i
            if k < K:
                # Direct element access - memory coherent for small chunks
                a_val = tl.load(a_ptr + pid_m * K + k)
                b_val = tl.load(b_ptr + k * N)
                accum += a_val * b_val
    
    tl.store(out_ptr + pid_m * N, accum)

# Kernel wrapper decorated with @torch.fx.wrap
@torch.fx.wrap
def optimized_matmul_2_1152_1(a, b):
    # Get tensor dimensions
    M, K = a.shape
    K2, N = b.shape
    
    assert K == K2, f"Dimension mismatch for matmul: {K} vs {K2}"
    assert M == 2, f"Expected M=2 but got {M}"
    assert N == 1, f"Expected N=1 but got {N}"
    
    # Create output tensor
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    # Launch kernel - one program per row (simplified)
    grid = (M,)  # Grid must be a tuple even for 1D
    
    matmul_kernel_2_1152_1[grid](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        M=M,
        K=K,
        N=N,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_matmul_2_1152_1