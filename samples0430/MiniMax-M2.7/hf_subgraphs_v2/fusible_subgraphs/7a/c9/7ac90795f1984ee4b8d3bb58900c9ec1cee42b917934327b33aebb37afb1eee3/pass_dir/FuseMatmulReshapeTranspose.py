import torch
import triton
import triton.language as tl


@triton.jit
def triton_matmul_kernel(
    in_1_ptr, in_0_ptr, out_ptr,
    stride_in1_b, stride_in1_m, stride_in1_k,
    stride_in0_b, stride_in0_k, stride_in0_n,
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized matmul kernel for [B, M, K] @ [B, K, 1] = [B, M]
    """
    pid = tl.program_id(0)
    
    if pid >= B * M:
        return
    
    batch_idx = pid // M
    m_idx = pid % M
    
    # Load in_1 [B, M, K]
    offs = batch_idx * stride_in1_b + m_idx * stride_in1_m + tl.arange(0, K)
    mask = tl.arange(0, K) < K
    a = tl.load(in_1_ptr + offs, mask=mask, other=0.0)
    
    # Load in_0 [B, K, 1]
    offs = batch_idx * stride_in0_b + tl.arange(0, K) * stride_in0_k
    b = tl.load(in_0_ptr + offs, mask=mask, other=0.0)
    
    # Dot product
    result = tl.sum(a * b)
    
    # Store
    tl.store(out_ptr + pid, result)


@torch.fx.wrap
def triton_matmul(in_1, in_0):
    """
    Triton kernel for matmul: [B, M, K] @ [B, K, 1] = [B, M]
    """
    B, M, K = in_1.shape
    
    out = torch.empty((B, M), dtype=in_1.dtype, device=in_1.device)
    
    grid = (B * M,)
    BLOCK_SIZE = 128
    
    triton_matmul_kernel[grid](
        in_1, in_0, out,
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        B, M, K,
        BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Match the computation pattern:
    1. matmul(in_1, in_0) - [B, M, K] @ [B, K, 1] -> [B, M]
    2. reshape(matmul, [-1, 16/128/384])
    3. transpose(in_2, -1, -2)
    
    Returns (reshape_result, transpose_result)
    """
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 16])
    tmp_2 = in_2.transpose(-1, -2)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments. The reshape dimension is extracted from the pattern.
    Route string "16" identifies this specific pattern.
    """
    return (in_0, in_1, in_2, "16")


def replacement_func():
    return triton_matmul