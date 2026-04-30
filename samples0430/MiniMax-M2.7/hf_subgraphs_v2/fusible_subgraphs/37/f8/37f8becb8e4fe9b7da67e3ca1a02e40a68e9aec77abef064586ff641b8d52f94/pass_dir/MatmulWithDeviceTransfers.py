import torch
import triton
import triton.language as tl

# Optimized Triton GEMV kernel designed for small matrix multiplications
# Uses a 1D grid where each program computes one output element
@triton.jit
def gemv_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized GEMV kernel with vectorized loads for better memory throughput."""
    pid = tl.program_id(0)
    row_idx = pid // N
    col_idx = pid % N
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over K with fixed block size for efficiency
    for k_offset in range(0, K, BLOCK_SIZE_K):
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # Vectorized load for better memory coalescing
        offs_a = row_idx * stride_am + offs_k
        a = tl.load(a_ptr + offs_a, mask=mask_k, other=0.0)
        
        # Contiguous access pattern for matrix B
        offs_b = offs_k * stride_bk + col_idx * stride_bn
        b = tl.load(b_ptr + offs_b, mask=mask_k, other=0.0)
        
        # Accumulate
        acc += tl.sum(a * b)
    
    # Store result
    offs_c = row_idx * stride_cm + col_idx * stride_cn
    tl.store(c_ptr + offs_c, acc)


@torch.fx.wrap
def triton_gemv(a, b):
    """
    Optimized GEMV using Triton kernel.
    """
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, f"Dimension mismatch: {K} != {K_b}"
    
    # Allocate output tensor
    output = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    # Fixed block size optimized for K=768 and K=1152
    block_size_k = 256
    
    # Launch kernel with 1D grid
    num_programs = M * N
    gemv_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        c_ptr=output,
        M=M, N=N, K=K,
        stride_am=a.stride(0),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        BLOCK_SIZE_K=block_size_k,
    )
    
    return output


def pattern(in_2, in_3):
    """
    Match the matmul operation only.
    """
    result = torch.matmul(in_2, in_3)
    return result


def replacement_args(in_2, in_3):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_2, in_3)


def optimized_matmul(in_2, in_3):
    """
    Optimized matmul function using Triton GEMV kernel.
    """
    return triton_gemv(in_2, in_3)


def replacement_func():
    """
    Returns the replacement function.
    """
    return optimized_matmul