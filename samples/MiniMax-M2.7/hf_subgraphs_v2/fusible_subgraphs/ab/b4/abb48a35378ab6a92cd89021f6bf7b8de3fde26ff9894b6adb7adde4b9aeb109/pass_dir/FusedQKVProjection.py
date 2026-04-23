"""
Fused QKV Projection Pass
Fuses: Linear + Reshape + Permute + Unbind + Transpose
This pattern is common in attention mechanisms for ViT-like models.
"""
import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the QKV decomposition pattern:
    1. Linear projection (using matmul + transpose, avoiding blocked linear API)
    2. Reshape to (1, 197, 3, X, 48)
    3. Permute to (3, 1, X, 197, 48)
    4. Unbind to get 3 tensors
    5. Transpose the K tensor (index 1)
    
    Note: We return tmp_8 (K transposed), tmp_5 (Q), tmp_7 (V)
    """
    # Use matmul instead of linear - same computation, avoids blocked API
    tmp_1 = torch.matmul(in_1, in_0.t())
    tmp_2 = tmp_1.reshape(1, 197, 3, 4, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    tmp_4 = tmp_3.unbind(0)
    tmp_5 = tmp_4[0]
    tmp_6 = tmp_4[1]
    tmp_7 = tmp_4[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def pattern_9x(in_0, in_1):
    """Pattern for convit_small with reshape (1, 197, 3, 9, 48)"""
    tmp_1 = torch.matmul(in_1, in_0.t())
    tmp_2 = tmp_1.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    tmp_4 = tmp_3.unbind(0)
    tmp_5 = tmp_4[0]
    tmp_6 = tmp_4[1]
    tmp_7 = tmp_4[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def pattern_16x(in_0, in_1):
    """Pattern for convit_base with reshape (1, 197, 3, 16, 48)"""
    tmp_1 = torch.matmul(in_1, in_0.t())
    tmp_2 = tmp_1.reshape(1, 197, 3, 16, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    tmp_4 = tmp_3.unbind(0)
    tmp_5 = tmp_4[0]
    tmp_6 = tmp_4[1]
    tmp_7 = tmp_4[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Autotune configurations for different head sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'num_stages': 2}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def fused_qkv_kernel_4x(
    x_ptr, w_ptr, q_ptr, k_ptr, v_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wq, stride_wk, stride_wv,
    stride_qm, stride_qn,
    stride_km, stride_kn,
    stride_vm, stride_vn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, num_stages: tl.constexpr
):
    """
    Fused QKV kernel for head_size=4*48=192
    Computes: Q = X @ W[:192,:].T, K = X @ W[192:384,:].T, V = X @ W[384:,:].T
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block indices
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, K)
    
    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K
    
    # Load x: [M, K]
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Compute Q: x @ W[:N,:].T
    wq_ptrs = w_ptr + offs_k[:, None] * stride_wq + offs_n[None, :] * stride_wq
    wq = tl.load(wq_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    q = tl.dot(x, wq)
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn
    tl.store(q_ptrs, q, mask=mask_m[:, None] & mask_n[None, :])
    
    # Compute K: x @ W[N:2N,:].T
    wk_ptrs = w_ptr + (offs_k[:, None] * stride_wq) + ((offs_n[None, :] + N) * stride_wq)
    wk = tl.load(wk_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    k = tl.dot(x, wk)
    k_ptrs = k_ptr + offs_m[:, None] * stride_km + offs_n[None, :] * stride_kn
    tl.store(k_ptrs, k, mask=mask_m[:, None] & mask_n[None, :])
    
    # Compute V: x @ W[2N:,:].T
    wv_ptrs = w_ptr + (offs_k[:, None] * stride_wq) + ((offs_n[None, :] + 2 * N) * stride_wq)
    wv = tl.load(wv_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    v = tl.dot(x, wv)
    v_ptrs = v_ptr + offs_m[:, None] * stride_vm + offs_n[None, :] * stride_vn
    tl.store(v_ptrs, v, mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'num_stages': 2}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def fused_qkv_kernel_9x(
    x_ptr, w_ptr, q_ptr, k_ptr, v_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wq, stride_wk, stride_wv,
    stride_qm, stride_qn,
    stride_km, stride_kn,
    stride_vm, stride_vn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, num_stages: tl.constexpr
):
    """
    Fused QKV kernel for head_size=9*48=432
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, K)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K
    
    # Load x: [M, K]
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Compute Q
    wq_ptrs = w_ptr + offs_k[:, None] * stride_wq + offs_n[None, :] * stride_wq
    wq = tl.load(wq_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    q = tl.dot(x, wq)
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn
    tl.store(q_ptrs, q, mask=mask_m[:, None] & mask_n[None, :])
    
    # Compute K
    wk_ptrs = w_ptr + (offs_k[:, None] * stride_wq) + ((offs_n[None, :] + N) * stride_wq)
    wk = tl.load(wk_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    k = tl.dot(x, wk)
    k_ptrs = k_ptr + offs_m[:, None] * stride_km + offs_n[None, :] * stride_kn
    tl.store(k_ptrs, k, mask=mask_m[:, None] & mask_n[None, :])
    
    # Compute V
    wv_ptrs = w_ptr + (offs_k[:, None] * stride_wq) + ((offs_n[None, :] + 2 * N) * stride_wq)
    wv = tl.load(wv_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    v = tl.dot(x, wv)
    v_ptrs = v_ptr + offs_m[:, None] * stride_vm + offs_n[None, :] * stride_vn
    tl.store(v_ptrs, v, mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'num_stages': 2}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def fused_qkv_kernel_16x(
    x_ptr, w_ptr, q_ptr, k_ptr, v_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wq, stride_wk, stride_wv,
    stride_qm, stride_qn,
    stride_km, stride_kn,
    stride_vm, stride_vn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, num_stages: tl.constexpr
):
    """
    Fused QKV kernel for head_size=16*48=768
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, K)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K
    
    # Load x: [M, K]
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Compute Q
    wq_ptrs = w_ptr + offs_k[:, None] * stride_wq + offs_n[None, :] * stride_wq
    wq = tl.load(wq_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    q = tl.dot(x, wq)
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn
    tl.store(q_ptrs, q, mask=mask_m[:, None] & mask_n[None, :])
    
    # Compute K
    wk_ptrs = w_ptr + (offs_k[:, None] * stride_wq) + ((offs_n[None, :] + N) * stride_wq)
    wk = tl.load(wk_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    k = tl.dot(x, wk)
    k_ptrs = k_ptr + offs_m[:, None] * stride_km + offs_n[None, :] * stride_kn
    tl.store(k_ptrs, k, mask=mask_m[:, None] & mask_n[None, :])
    
    # Compute V
    wv_ptrs = w_ptr + (offs_k[:, None] * stride_wq) + ((offs_n[None, :] + 2 * N) * stride_wq)
    wv = tl.load(wv_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    v = tl.dot(x, wv)
    v_ptrs = v_ptr + offs_m[:, None] * stride_vm + offs_n[None, :] * stride_vn
    tl.store(v_ptrs, v, mask=mask_m[:, None] & mask_n[None, :])


@torch.fx.wrap
def fused_qkv_wrapper_4x(x, w):
    """
    Wrapper for fused QKV kernel with head_size=4*48=192
    Returns (Q, K^T, V) where K is already transposed
    """
    M, K = x.shape  # [1, 197, 192] -> [1, 192]
    N = 192  # 4 * 48
    
    # Reshape input to 2D if needed
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])
        M = x.shape[0]
    
    # Allocate output tensors
    q = torch.empty((M, N), dtype=x.dtype, device=x.device)
    k = torch.empty((M, N), dtype=x.dtype, device=x.device)
    v = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    # Grid: M blocks for rows, N/block_size for columns
    BLOCK_SIZE_N = 64
    grid = (M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    fused_qkv_kernel_4x[grid](
        x, w, q, k, v,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(0), w.stride(0),
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
    )
    
    # Reshape outputs: Q [1, 197, 192], K^T [1, 197, 48, 4], V [1, 197, 192]
    q = q.view(1, 197, 192)
    k = k.view(1, 197, 4, 48)  # [1, 197, 4, 48]
    k_t = k.transpose(-2, -1)   # [1, 197, 48, 4]
    v = v.view(1, 197, 192)
    
    return q, k_t, v


@torch.fx.wrap
def fused_qkv_wrapper_9x(x, w):
    """
    Wrapper for fused QKV kernel with head_size=9*48=432
    Returns (Q, K^T, V) where K is already transposed
    """
    M, K = x.shape
    N = 432  # 9 * 48
    
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])
        M = x.shape[0]
    
    q = torch.empty((M, N), dtype=x.dtype, device=x.device)
    k = torch.empty((M, N), dtype=x.dtype, device=x.device)
    v = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_N = 64
    grid = (M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    fused_qkv_kernel_9x[grid](
        x, w, q, k, v,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(0), w.stride(0),
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
    )
    
    # Reshape: Q [1, 197, 432], K^T [1, 197, 48, 9], V [1, 197, 432]
    q = q.view(1, 197, 432)
    k = k.view(1, 197, 9, 48)
    k_t = k.transpose(-2, -1)  # [1, 197, 48, 9]
    v = v.view(1, 197, 432)
    
    return q, k_t, v


@torch.fx.wrap
def fused_qkv_wrapper_16x(x, w):
    """
    Wrapper for fused QKV kernel with head_size=16*48=768
    Returns (Q, K^T, V) where K is already transposed
    """
    M, K = x.shape
    N = 768  # 16 * 48
    
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])
        M = x.shape[0]
    
    q = torch.empty((M, N), dtype=x.dtype, device=x.device)
    k = torch.empty((M, N), dtype=x.dtype, device=x.device)
    v = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_N = 64
    grid = (M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    fused_qkv_kernel_16x[grid](
        x, w, q, k, v,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(0), w.stride(0),
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
    )
    
    # Reshape: Q [1, 197, 768], K^T [1, 197, 48, 16], V [1, 197, 768]
    q = q.view(1, 197, 768)
    k = k.view(1, 197, 16, 48)
    k_t = k.transpose(-2, -1)  # [1, 197, 48, 16]
    v = v.view(1, 197, 768)
    
    return q, k_t, v


def replacement_func():
    return fused_qkv_wrapper_4x