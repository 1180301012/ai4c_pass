import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Standard GEMM + bias: out[M,N] = x[M,K] @ w[N,K].T + bias[N]
    x: [M, K]  w: [N, K]  bias: [N]  out: [M, N]
    All row-major, fp16/bf16 or fp32.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BN]
    offs_k = tl.arange(0, BLOCK_K)                      # [BK]

    # x[M,K] and w[N,K] – stride (K, 1)
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]   # [BM, BK]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]   # [BN, BK]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k * BLOCK_K)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :].to(tl.float32)

    # Store with same dtype as input
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def fused_linear_bias(x, weight, bias):
    """
    Optimized linear(x, weight, bias) = x @ weight.T + bias
    Dropout(training=False) is identity → absorbed.
    x:     [B, M, K]
    weight:[N, K]
    bias:  [N]
    out:   [B, M, N]   (same dtype as x)
    """
    B, M, K = x.shape[0], x.shape[1], x.shape[2]
    N = weight.shape[0]
    M_total = B * M

    out = torch.empty((B, M, N), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M_total, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    _gemm_bias_kernel[grid](
        x.reshape(M_total, K), weight, bias,
        out.reshape(M_total, N),
        M_total, N, K,
    )

    return out