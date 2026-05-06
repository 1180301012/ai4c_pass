import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_linear_transpose_kernel(
    X_ptr, W_ptr, b_ptr, Y_ptr, Y_T_ptr,
    M, N, K,
    stride_Xm, stride_Xk,
    stride_wn, stride_wk,
    stride_yrm,    # = N  (Y is [M, N], row-major, row-stride = N)
    stride_ytym,   # = M  (Y_T is [N, M] viewed in [M, N] memory, so row-stride = M)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ── Determine which tile this program handles (L2-friendly swizzle) ──────────
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ── Tile offsets ─────────────────────────────────────────────────────────────
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # ── Accumulate in fp32 ──────────────────────────────────────────────────────
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── K-reduction loop ────────────────────────────────────────────────────────
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_idx * BLOCK_K
        k_offs = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load X tile [BLOCK_M, BLOCK_K]
        a_ptrs = X_ptr + m_offs[:, None] * stride_Xm + k_offs[None, :] * stride_Xk
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W tile [BLOCK_N, BLOCK_K]  (W layout: [N, K] row-major)
        b_ptrs = W_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk
        b_mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # acc += a @ b.T  =>  [BLOCK_M, BLOCK_N]
        acc = tl.dot(a, tl.trans(b), acc)

    # ── Bias add ─────────────────────────────────────────────────────────────────
    bias = tl.load(b_ptr + n_offs, mask=n_offs < N, other=0.0)
    c = acc.to(bias.dtype) + bias[None, :]

    # ── Store to Y [M, N], stride_yrm = N  (row-major, coalesced over n) ─────────
    out_offs_y = m_offs[:, None] * stride_yrm + n_offs[None, :]
    out_mask_y = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(Y_ptr + out_offs_y, c, mask=out_mask_y)

    # ── Store to Y_T [N, M], stride_ytym = M  (stride of n-dim = M) ─────────────
    # Store each row of c_T (size [BLOCK_N, BLOCK_M]) at fixed n, varying m
    # consecutive m → coalesced stores
    c_T = tl.trans(c)  # [BLOCK_N, BLOCK_M]
    out_offs_yt_n = n_offs[:, None] * stride_ytym   # [BLOCK_N, 1]
    out_offs_yt_m = m_offs[None, :]                  # [1, BLOCK_M]
    out_offs_yt = out_offs_yt_n + out_offs_yt_m
    out_mask_yt = (n_offs[:, None] < N) & (m_offs[None, :] < M)
    tl.store(Y_T_ptr + out_offs_yt, c_T, mask=out_mask_yt)