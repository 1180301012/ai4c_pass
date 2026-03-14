import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_2 = torch.matmul(in_0, in_1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_permute_kernel(
    in_0_ptr,  # [B, M, K]
    in_1_ptr,  # [B, K, N]
    out_ptr,   # [B, N, M] (after permute)
    B: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_in0_b, stride_in0_m, stride_in0_k,
    stride_in1_b, stride_in1_k, stride_in1_n,
    stride_out_b, stride_out_n, stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel for: matmul -> permute
    
    in_0: [B, M, K]
    in_1: [B, K, N]
    out:  [B, N, M] (permuted from [B, M, N])
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    # Compute which M and N tile this program handles
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_m_tiles
    pid_n = pid // num_m_tiles
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Masks
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_k = offs_k < K
        
        # Load in_0[b, offs_m, offs_k] - shape [BLOCK_M, BLOCK_K]
        in0_ptrs = in_0_ptr + pid_b * stride_in0_b + offs_m[:, None] * stride_in0_m + offs_k[None, :] * stride_in0_k
        in0 = tl.load(in0_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load in_1[b, offs_k, offs_n] - shape [BLOCK_K, BLOCK_N]
        in1_ptrs = in_1_ptr + pid_b * stride_in1_b + offs_k[:, None] * stride_in1_k + offs_n[None, :] * stride_in1_n
        in1 = tl.load(in1_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate matmul
        acc += tl.dot(in0, in1)
    
    # Masks for output
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Store result with permutation: out[b, n, m]
    out_ptrs = out_ptr + pid_b * stride_out_b + offs_n[None, :] * stride_out_n + offs_m[:, None] * stride_out_m
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@torch.fx.wrap
def fused_matmul_permute(in_0, in_1):
    """
    Fused implementation of matmul -> permute
    
    in_0: [B, M, K]
    in_1: [B, K, N]
    returns: [B, N, M]
    """
    B, M, K = in_0.shape
    _, _, N = in_1.shape
    
    # Output shape is [B, N, M] after permute
    out = torch.empty((B, N, M), device=in_0.device, dtype=in_0.dtype)
    
    # Grid dimensions
    BLOCK_M = 64
    BLOCK_N = 64
    num_m_tiles = triton.cdiv(M, BLOCK_M)
    num_n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (num_m_tiles * num_n_tiles, B)
    
    # Launch kernel
    fused_matmul_permute_kernel[grid](
        in_0, in_1, out,
        B, M, K, N,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
    )
    
    return out


def replacement_func():
    return fused_matmul_permute