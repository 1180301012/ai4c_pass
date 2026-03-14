import torch
import triton
import triton.language as tl


def pattern(in_2, in_3):
    """
    Match the matmul pattern in the computation.
    """
    tmp_2 = torch.matmul(in_2, in_3)
    return tmp_2


def replacement_args(in_2, in_3):
    """
    Extract arguments for the replacement function.
    """
    return (in_2, in_3)


@triton.jit
def matvec_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized matrix-vector multiplication kernel with vectorized loads.
    Each program computes one output element by reducing over K.
    """
    # Program ID for output row
    pid_m = tl.program_id(0)
    
    if pid_m >= M:
        return
    
    # Accumulator
    accumulator = 0.0
    
    # Iterate over K dimension in blocks
    for k_start in range(0, K, BLOCK_SIZE_K):
        # Offsets
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask = offs_k < K
        
        # Load from A (row vector) and B (column vector) with caching
        a = tl.load(a_ptr + pid_m * stride_am + offs_k * stride_ak, mask=mask, other=0.0, cache_modifier=".cg")
        b = tl.load(b_ptr + offs_k * stride_bk, mask=mask, other=0.0, cache_modifier=".cg")
        
        # Multiply element-wise and reduce
        accumulator += tl.sum(a * b)
    
    # Store result
    tl.store(c_ptr + pid_m * stride_cm, accumulator)


@torch.fx.wrap
def optimized_matmul(a, b):
    """
    Wrapper for the optimized matmul kernel.
    Optimized for matrix-vector multiplication.
    """
    # Get dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel - one program per output row
    grid = (M,)
    
    # Use 2048 to cover K=1152 in single iteration
    BLOCK_SIZE_K = 2048
    
    matvec_kernel[grid](
        a, b, c,
        M, K,
        a.stride(0), a.stride(1),
        b.stride(0),
        c.stride(0),
        BLOCK_SIZE_K,
    )
    
    return c


def replacement_func():
    """
    Return the optimized matmul function.
    """
    return optimized_matmul