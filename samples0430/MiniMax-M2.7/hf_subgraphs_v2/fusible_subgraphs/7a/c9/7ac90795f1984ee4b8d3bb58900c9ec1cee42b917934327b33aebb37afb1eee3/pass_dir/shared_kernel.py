import torch
import triton
import triton.language as tl


@triton.jit
def matmul_dot_kernel(
    in_1_ptr, in_0_ptr, out_ptr,
    stride_in1_b, stride_in1_m, stride_in1_k,
    stride_in0_b, stride_in0_k,
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized dot product kernel for [B, M, K] @ [B, K, 1] = [B, M]
    
    Each thread computes one output element: the dot product of one row of in_1
    with one row of in_0 (broadcast to K elements).
    """
    pid = tl.program_id(0)
    
    if pid >= B * M:
        return
    
    batch_idx = pid // M
    m_idx = pid % M
    
    # Compute offsets for in_1 [B, M, K]
    offs_in1 = batch_idx * stride_in1_b + m_idx * stride_in1_m + tl.arange(0, K)
    mask = tl.arange(0, K) < K
    a = tl.load(in_1_ptr + offs_in1, mask=mask, other=0.0)
    
    # Compute offsets for in_0 [B, K, 1]
    offs_in0 = batch_idx * stride_in0_b + tl.arange(0, K) * stride_in0_k
    b = tl.load(in_0_ptr + offs_in0, mask=mask, other=0.0)
    
    # Element-wise multiply and sum (dot product)
    result = tl.sum(a * b)
    
    # Store result
    tl.store(out_ptr + pid, result)


@torch.fx.wrap
def optimized_matmul(in_1, in_0):
    """
    High-performance matmul for [B, M, K] @ [B, K, 1] = [B, M]
    """
    B, M, K = in_1.shape
    
    # Allocate output
    out = torch.empty((B, M), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel with one program per output element
    grid = (B * M,)
    BLOCK_SIZE = 128
    
    matmul_dot_kernel[grid](
        in_1, in_0, out,
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        in_0.stride(0), in_0.stride(1),
        B, M, K,
        BLOCK_SIZE,
    )
    
    return out