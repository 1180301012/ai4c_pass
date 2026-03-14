import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function - matches ONLY the matmul operation
def pattern(in_2, in_3):
    tmp_2 = torch.matmul(in_2, in_3)
    return tmp_2


# Argument extraction function - extracts args needed for replacement
# Must match the pattern's parameters
def replacement_args(in_2, in_3):
    return (in_2, in_3)


# Replacement function that uses optimized Triton kernel
@torch.fx.wrap
def triton_matmul_replacement(a, b):
    """Triton-based matrix multiplication replacement."""
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel with K as constexpr
    _matmul_kernel[(M,)](
        a, b, c,
        M, K, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        K,  # Pass K as constexpr
    )
    
    return c


# Optimized Triton kernel for matmul [M, K] @ [K, N] = [M, N]
# Using constexpr for K to enable full unrolling
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    K_CONST: tl.constexpr,
):
    """Triton kernel for matrix multiplication with constexpr K."""
    # Get row index
    row_idx = tl.program_id(0)
    
    # Initialize accumulator
    accumulator = 0.0
    
    # Use unrolled loop with constexpr
    for k in range(K_CONST):
        # Load element from matrix a
        a_idx = row_idx * stride_am + k * stride_ak
        a_val = tl.load(a_ptr + a_idx)
        
        # Load element from vector b
        b_idx = k * stride_bk
        b_val = tl.load(b_ptr + b_idx)
        
        # FMA
        accumulator = tl.fma(a_val, b_val, accumulator)
    
    # Store result
    c_idx = row_idx * stride_cm
    tl.store(c_ptr + c_idx, accumulator)


def replacement_func():
    return triton_matmul_replacement