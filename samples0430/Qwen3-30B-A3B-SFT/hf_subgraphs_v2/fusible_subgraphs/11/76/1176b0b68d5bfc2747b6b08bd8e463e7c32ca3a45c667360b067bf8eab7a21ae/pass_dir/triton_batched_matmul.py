"""
Shared Triton kernel for batched matrix multiplication.
Handles inputs shaped [B, H, D, N] @ [B, H, N, N] -> [B, H, D, N].
"""
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Large tiles – good for large M/BH
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        # Medium tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        # Large BLOCK_N=256 – reduces tile count for N=400 (only 2 tiles needed)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def batched_matmul_kernel(
    x_ptr, y_ptr, z_ptr,
    M, N, K,
    stride_xb, stride_xh, stride_xm, stride_xk,
    stride_yb, stride_yh, stride_yk, stride_yn,
    stride_zb, stride_zh, stride_zm, stride_zk,
    NUM_H_HEADS,          # total number of (batch, head) pairs = B * H
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Computes z[b, h, m, n] = sum_k(x[b, h, m, k] * y[b, h, k, n])
    where x is [B, H, M, K] and y is [B, H, K, N].
    BH (batch × head) index is flattened into pid_b.
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Decompose pid_b into (batch index, head index)
    b_idx = pid_b // NUM_H_HEADS
    h_idx = pid_b % NUM_H_HEADS

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for this (b, h) batch element
    x_base = x_ptr + b_idx * stride_xb + h_idx * stride_xh
    y_base = y_ptr + b_idx * stride_yb + h_idx * stride_yh
    z_base = z_ptr + b_idx * stride_zb + h_idx * stride_zh

    x_ptrs = x_base + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    y_ptrs = y_base + offs_k[:, None] * stride_yk + offs_n[None, :] * stride_yn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_remaining = K - k_start
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        y = tl.load(y_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, y)
        x_ptrs += BLOCK_K * stride_xk
        y_ptrs += BLOCK_K * stride_yk

    z_ptrs = z_base + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zk
    tl.store(z_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))