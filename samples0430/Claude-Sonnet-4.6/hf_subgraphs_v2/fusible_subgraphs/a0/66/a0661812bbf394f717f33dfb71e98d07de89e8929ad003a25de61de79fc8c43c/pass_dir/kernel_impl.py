"""
Shared Triton GEMM+bias kernel and dispatch wrapper used by all
FuseLinear_Dropout*_Trans12_* passes.
"""
import torch
import triton
import triton.language as tl


# ── fp16 kernel ───────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_fp16(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w_tile = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x_tile, w_tile)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + tl.expand_dims(b.to(tl.float32), 0)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ── bf16 kernel ───────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_bf16(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w_tile = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x_tile, w_tile)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + tl.expand_dims(b.to(tl.float32), 0)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ── fp32 kernel ───────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 2}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_fp32(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w_tile = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x_tile, w_tile)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + tl.expand_dims(b, 0)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))



# ── unified single kernel (no autotune; concrete-grid friendly) ──────────────
@triton.jit
def _gemm_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w_tile = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x_tile, w_tile)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :].to(tl.float32)
    if IS_FP16:
        out = acc.to(tl.float16)
    elif IS_BF16:
        out = acc.to(tl.bfloat16)
    else:
        out = acc
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


_BLOCK_M = 128
_BLOCK_N = 64
_BLOCK_K = 32


@torch.fx.wrap
def fused_linear_transpose_lt(in_0, in_1, in_2):
    """linear(in_2,in_1,in_0)+transpose(1,2). Returns (linear_out, transposed)."""
    B = in_2.shape[0]
    N_seq = in_2.shape[1]
    K = in_2.shape[2]
    N_out = in_1.shape[0]
    M = B * N_seq
    x_2d = in_2.view(M, K)
    out_2d = torch.empty((M, N_out), dtype=in_2.dtype, device=in_2.device)
    is_fp16 = in_2.dtype == torch.float16
    is_bf16 = in_2.dtype == torch.bfloat16
    gm = ((M + _BLOCK_M - 1) // _BLOCK_M, (N_out + _BLOCK_N - 1) // _BLOCK_N)
    _gemm_bias_kernel[gm](
        x_2d, in_1, in_0, out_2d,
        M, N_out, K,
        x_2d.stride(0), x_2d.stride(1),
        in_1.stride(0), in_1.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        IS_FP16=is_fp16, IS_BF16=is_bf16,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
    )
    out_3d = out_2d.view(B, N_seq, N_out)
    t = out_3d.transpose(1, 2)
    return out_3d, t


@torch.fx.wrap
def fused_linear_transpose_tl(in_0, in_1, in_2):
    """linear(in_2,in_1,in_0)+transpose(1,2). Returns (transposed, linear_out)."""
    B = in_2.shape[0]
    N_seq = in_2.shape[1]
    K = in_2.shape[2]
    N_out = in_1.shape[0]
    M = B * N_seq
    x_2d = in_2.view(M, K)
    out_2d = torch.empty((M, N_out), dtype=in_2.dtype, device=in_2.device)
    is_fp16 = in_2.dtype == torch.float16
    is_bf16 = in_2.dtype == torch.bfloat16
    gm = ((M + _BLOCK_M - 1) // _BLOCK_M, (N_out + _BLOCK_N - 1) // _BLOCK_N)
    _gemm_bias_kernel[gm](
        x_2d, in_1, in_0, out_2d,
        M, N_out, K,
        x_2d.stride(0), x_2d.stride(1),
        in_1.stride(0), in_1.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        IS_FP16=is_fp16, IS_BF16=is_bf16,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
    )
    out_3d = out_2d.view(B, N_seq, N_out)
    t = out_3d.transpose(1, 2)
    return t, out_3d


@torch.fx.wrap
def fused_linear(in_0, in_1, in_2):
    """linear(in_2, in_1, in_0).  Returns single [B, N_seq, N_out] tensor."""
    B = in_2.shape[0]
    N_seq = in_2.shape[1]
    K = in_2.shape[2]
    N_out = in_1.shape[0]
    M = B * N_seq
    # Allocate output as 3D directly — no .view() needed
    out_3d = torch.empty((B, N_seq, N_out), dtype=in_2.dtype, device=in_2.device)
    is_fp16 = in_2.dtype == torch.float16
    is_bf16 = in_2.dtype == torch.bfloat16
    gm = ((M + _BLOCK_M - 1) // _BLOCK_M, (N_out + _BLOCK_N - 1) // _BLOCK_N)
    # Use .stride() for metadata (not dispatched through ATen)
    # in_2: [B,N_seq,K] contiguous → stride(1)=K, stride(2)=1 gives [B*N_seq, K] layout
    # out_3d: [B,N_seq,N_out] contiguous → stride(1)=N_out, stride(2)=1
    _gemm_bias_kernel[gm](
        in_2, in_1, in_0, out_3d,
        M, N_out, K,
        in_2.stride(1), in_2.stride(2),
        in_1.stride(0), in_1.stride(1),
        out_3d.stride(1), out_3d.stride(2),
        IS_FP16=is_fp16, IS_BF16=is_bf16,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
    )
    return out_3d