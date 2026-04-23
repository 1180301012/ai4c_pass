import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match matmul followed by squeeze(1)
    Pattern: in_0 @ in_1 -> squeeze(1)
    in_0 shape: [1, 1, 249] (att_2)
    in_1 shape: [1, 249, 64] (concat)
    Result shape: [1, 1, 64] -> squeeze(1) -> [1, 64]
    """
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def matmul_squeeze1_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused matmul + squeeze(1) kernel.
    
    Computes: out[m, n] = sum_k(a[0, 0, k] * b[0, k, n])
    This effectively treats a's first two dims as broadcast and computes the 1D dot product.
    The squeeze(1) is free since we never create the middle dimension.
    
    Args:
        a: [1, 1, K] matrix
        b: [1, K, N] matrix
        out: [1, N] matrix (squeezed result)
    """
    # Program IDs for the output grid
    pid_m = tl.program_id(0)  # Row index (should be 0 since we squeeze)
    pid_n = tl.program_id(1)  # Column index
    
    # Create offset for N dimension
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # Load a: a[0, 0, k] - scalar for this row
        # Since M=1, pid_m should be 0, so we load a[0, 0, offs_k]
        offs_a = pid_m * stride_am + offs_k * stride_ak
        a = tl.load(a_ptr + offs_a, mask=mask_k, other=0.0)
        
        # Load b: b[0, offs_k, offs_n] - vector for each column
        offs_b = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
        
        # Multiply and accumulate: acc[n] += sum_k(a[k] * b[k, n])
        acc += tl.sum(a[:, None] * b, axis=0)
    
    # Store result
    offs_out = pid_m * stride_om + offs_n * stride_on
    tl.store(out_ptr + offs_out, acc, mask=mask_n)


@torch.fx.wrap
def fused_matmul_squeeze1(a, b):
    """
    Wrapper for the fused matmul + squeeze(1) kernel.
    
    Args:
        a: [1, 1, K] tensor
        b: [1, K, N] tensor
    
    Returns:
        [1, N] tensor (squeezed result)
    """
    # Get dimensions
    # a: [1, 1, K] -> M=1, K determined by a.shape[2]
    # b: [1, K, N] -> N determined by b.shape[2]
    M = 1
    K = a.shape[2]
    N = b.shape[2]
    
    # Allocate output
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    # Configure block sizes
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    # Grid: (M, ceil(N/BLOCK_SIZE_N))
    grid = (M, triton.cdiv(N, BLOCK_SIZE_N), 1)
    
    matmul_squeeze1_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(2),
        b.stride(1), b.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


def replacement_func():
    return fused_matmul_squeeze1