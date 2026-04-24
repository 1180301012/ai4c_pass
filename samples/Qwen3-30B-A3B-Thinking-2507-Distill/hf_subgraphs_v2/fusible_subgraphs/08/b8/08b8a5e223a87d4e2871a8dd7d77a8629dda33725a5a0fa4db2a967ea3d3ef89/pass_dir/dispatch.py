"""
Shared dispatch function and Triton kernels for all AI4C passes.

All pass files import `dispatch` from this module so that
replacement_func() is identical across passes (same Python object),
avoiding output_pass_replacement_func_limit dropping any pass.

Routes:
  "mul"     -> elementwise multiply with broadcast scale
  "linear"  -> linear GEMM: C = A @ W^T  (W: [N,K], A: [...,K] -> [...,N])
  "fused"   -> fused: (A @ W^T) * scale
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: elementwise multiply with 1-D broadcast scale
# ---------------------------------------------------------------------------
@triton.jit
def _mul_broadcast_kernel(
    x_ptr, scale_ptr, out_ptr,
    n_elements, n_scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale_idx = offsets % n_scale
    scale = tl.load(scale_ptr + scale_idx, mask=mask, other=1.0)
    tl.store(out_ptr + offsets, x * scale, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: GEMM C = A @ B^T  (for linear / fused patterns)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C = A @ B^T  (A:[M,K], B:[N,K], C:[M,N])"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
        )
        # B transposed: load B[n,k] at position [k,n]
        b_T = tl.load(
            B_ptr + offs_k[:, None] * stride_bn + offs_n[None, :] * stride_bk,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        )
        acc += tl.dot(a, b_T)

    if OUT_DTYPE == tl.float16:
        c = acc.to(tl.float16)
    elif OUT_DTYPE == tl.bfloat16:
        c = acc.to(tl.bfloat16)
    else:
        c = acc

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Kernel 3: fused GEMM + elementwise scale
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_linear_scale_kernel(
    A_ptr, B_ptr, scale_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C = (A @ B^T) * scale  (A:[M,K], B:[N,K], scale:[N], C:[M,N])"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
        )
        b_T = tl.load(
            B_ptr + offs_k[:, None] * stride_bn + offs_n[None, :] * stride_bk,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        )
        acc += tl.dot(a, b_T)

    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = acc * scale[None, :]

    if OUT_DTYPE == tl.float16:
        c = acc.to(tl.float16)
    elif OUT_DTYPE == tl.bfloat16:
        c = acc.to(tl.bfloat16)
    else:
        c = acc

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Python wrappers (called from dispatch)
# ---------------------------------------------------------------------------
def _run_mul(a, b):
    """a=scale[N], b=input[...,N]  -> broadcast multiply"""
    N = a.shape[0]
    n_elements = b.numel()
    out = torch.empty_like(b)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _mul_broadcast_kernel[grid](b, a, out, n_elements, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


def _run_linear(weight, x):
    """
    weight=[N,K], x=([...],K) — any leading batch dims.
    Returns ([...],N) — no reshape needed; strides handle multi-dim x.
    """
    K = x.shape[-1]
    N = weight.shape[0]
    M = x.numel() // K
    # Output batch dims = x.batch_dims, last dim = N (not K)
    out = torch.empty(x.shape[:-1] + (N,), dtype=x.dtype, device=x.device)
    stride_am = K          # contiguous [..., K]: row stride = K
    stride_ak = 1
    stride_bn = K          # contiguous [N, K]
    stride_bk = 1
    stride_cm = N          # output: row stride = N
    stride_cn = 1
    if x.dtype == torch.float16:
        OUT_DTYPE = tl.float16
    elif x.dtype == torch.bfloat16:
        OUT_DTYPE = tl.bfloat16
    else:
        OUT_DTYPE = tl.float32
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    _linear_gemm_kernel[grid](
        x, weight, out, M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        OUT_DTYPE=OUT_DTYPE,
    )
    return out


def _run_fused(weight, scale, x):
    """weight=[N,K], scale=[N], x=[...,K]  -> (x @ weight.T) * scale"""
    M = x.numel() // x.shape[-1]
    N = weight.shape[0]
    # Output has same batch dims as x, but last dim = N (not K)
    out = torch.empty(x.shape[:-1] + (N,), dtype=x.dtype, device=x.device)
    K = x.shape[-1]
    stride_am = K          # contiguous [..., K]
    stride_ak = 1
    stride_bn = weight.shape[-1]   # K for contiguous [N, K]
    stride_bk = 1
    stride_cm = N
    stride_cn = 1
    if x.dtype == torch.float16:
        OUT_DTYPE = tl.float16
    elif x.dtype == torch.bfloat16:
        OUT_DTYPE = tl.bfloat16
    else:
        OUT_DTYPE = tl.float32
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    _fused_linear_scale_kernel[grid](
        x, weight, scale, out, M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        OUT_DTYPE=OUT_DTYPE,
    )
    return out


# ---------------------------------------------------------------------------
# Shared dispatch function  (ALL pass files return THIS exact object)
# Signature: dispatch(a, b, route)  – c is ignored for non-fused routes.
# All passes return exactly 3 args from replacement_args so the arity is fixed.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch(a, b, route):
    """
    route=="mul"     : a=scale[N], b=input  -> b * scale  (broadcast multiply)
    route=="linear"  : a=weight[N,K], b=input -> b @ a.T  (GEMM)
    route=="fused"   : a=weight[N,K], b=scale[N], c=input -> (c @ a.T) * b
                       (c == b is a dummy; use b as input for mul/linear routes)
    """
    if route == "mul":
        return _run_mul(a, b)
    elif route == "linear":
        return _run_linear(a, b)
    else:  # "fused"
        return _run_fused(a, b, b)  # b acts as dummy input for non-fused routes