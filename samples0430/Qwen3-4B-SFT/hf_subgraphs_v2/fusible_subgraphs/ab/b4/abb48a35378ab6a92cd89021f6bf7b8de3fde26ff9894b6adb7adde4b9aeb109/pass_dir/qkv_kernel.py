"""
Shared Triton QKV projection kernels.
Two kernels per call:
  1. GEMM:  tmp[M, N_out] = in_1[M, K] @ in_0^T[N_out, K]  (computes linear)
  2. Scatter: reads tmp and writes to Q[B,S,N,HD], K_T[B,S,HD,N], V[B,S,N,HD]
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
#  GEMM kernel: C = A @ B^T
#   A: [M, K]  B: [N_out, K]  → C: [M, N_out]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 96,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 96,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 192, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 192, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 144, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 144, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 96,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 96,  'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 192, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 192, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N_out', 'K'],
)
@triton.jit
def qkv_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N_out, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[m,n] = sum_k A[m,k] * B[n,k]   (B stored as [N_out, K], i.e. B^T = [K,N_out])"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # A tile  [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        # B^T tile  [BLOCK_K, BLOCK_N]  – reads B[offs_n, offs_k]
        b = tl.load(
            B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk,
            mask=(offs_n[None, :] < N_out) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a, b)

    c = acc.to(A_ptr.dtype.element_ty)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N_out),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  QKV scatter kernel
#   src  [M * N_out] – flat linear output
#   Writes  Q [M, N, HD], K_T [M, HD, N], V [M, N, HD]
#   src linearised as: [M, 3*N, HD]  →  address m*(3*N*HD) + qkv*N*HD + n*HD + d
#   Offset math:
#     Q[b,s,h,d] = src[m= b*S+s,  0*N*HD + h*HD + d]
#     K_T[b,s,d,h] = src[m= b*S+s,  1*N*HD + d*HD + h]
#     V[b,s,h,d] = src[m= b*S+s,  2*N*HD + h*HD + d]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_out', 'M'],
)
@triton.jit
def qkv_split_kernel(
    src_ptr, Q_ptr, KTPtr, V_ptr,
    N_out, N, HEAD_DIM,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N_out * HEAD_DIM   # M elements per sequence; total here means N_out elements (per M row)
    mask  = offs < total

    m     = offs // N_out
    rem   = offs % N_out
    n_idx = (rem % (N * HEAD_DIM)) // HEAD_DIM
    hd_idx = rem % HEAD_DIM

    val = tl.load(src_ptr + offs, mask=mask, other=0.0)

    S2 = N * HEAD_DIM
    S3 = HEAD_DIM * N

    tl.store(Q_ptr       + m * S2 + n_idx * HEAD_DIM + hd_idx, val, mask=mask)
    tl.store(KTPtr       + m * S3 + hd_idx * N     + n_idx,   val, mask=mask)
    tl.store(V_ptr       + m * S2 + n_idx * HEAD_DIM + hd_idx, val, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
#  Python entry-point wrapper  (used by pass files as replacement_func)
# ─────────────────────────────────────────────────────────────────────────────
def fused_qkv_forward(in_0, in_1, N, HEAD_DIM: int = 48):
    """
    in_0 : weight  [3*N*HEAD_DIM, K]
    in_1 : input   [1, S, K]
    Returns (Q [1,S,N,HD], K_T [1,S,HD,N], V [1,S,N,HD])
    """
    device = in_1.device
    if in_0.device == torch.device("cpu"):
        in_0 = in_0.to(device)

    B, S, K = in_1.shape[0], in_1.shape[1], in_1.shape[2]
    N_out = 3 * N * HEAD_DIM
    M = B * S

    # ── Step 1: GEMM ──────────────────────────────────────────────────────────
    tmp = torch.empty((M, N_out), dtype=in_1.dtype, device=device)

    grid_gemm = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_out, meta['BLOCK_N']),
    )
    qkv_gemm_kernel[grid_gemm](
        in_1.reshape(M, K), in_0, tmp,
        M, N_out, K,
        in_1.reshape(M, K).stride(0), in_1.reshape(M, K).stride(1),
        in_0.stride(0), in_0.stride(1),
        tmp.stride(0), tmp.stride(1),
    )

    # ── Step 2: scatter into Q, K_T, V ───────────────────────────────────────
    Q   = torch.empty((B, S, N, HEAD_DIM), dtype=in_1.dtype, device=device)
    K_T = torch.empty((B, S, HEAD_DIM, N), dtype=in_1.dtype, device=device)
    V   = torch.empty((B, S, N, HEAD_DIM), dtype=in_1.dtype, device=device)

    grid_split = lambda meta: (triton.cdiv(M * N_out, meta['BLOCK_SIZE']),)
    qkv_split_kernel[grid_split](tmp, Q, K_T, V, N_out, N, HEAD_DIM)

    return Q, K_T, V