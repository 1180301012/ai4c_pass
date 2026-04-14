"""
Shared Triton kernel implementations and universal dispatch for all passes.
All passes import and return `universal_dispatch` from this module so that
only ONE unique replacement_func object is seen by the framework.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused GEMM + element-wise gate multiply
#   out[i,j] = sum_k(x[i,k] * w[j,k]) * gate[i,j]
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_gemm_mul_kernel(
    x_ptr, w_ptr, gate_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_gm, stride_gn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k  = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k * BLOCK_K)
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    true_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    true_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (true_m[:, None] < M) & (true_n[None, :] < N)

    gate = tl.load(
        gate_ptr + true_m[:, None] * stride_gm + true_n[None, :] * stride_gn,
        mask=out_mask, other=0.0,
    )
    acc = acc * gate.to(tl.float32)

    tl.store(
        out_ptr + true_m[:, None] * stride_om + true_n[None, :] * stride_on,
        acc.to(x_ptr.dtype.element_ty),
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Kernel 2: Standalone tiled GEMM
#   out[i,j] = sum_k(x[i,k] * w[j,k])
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k  = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k * BLOCK_K)
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    true_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    true_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (true_m[:, None] < M) & (true_n[None, :] < N)

    tl.store(
        out_ptr + true_m[:, None] * stride_om + true_n[None, :] * stride_on,
        acc.to(x_ptr.dtype.element_ty),
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Kernel 3: Flat element-wise broadcast multiply (no autotune — fixed block
#   size minimises Python dispatch overhead for small RTMPose tensors).
#   out[i] = a[i] * scale[i % N]
# ---------------------------------------------------------------------------

@triton.jit
def _broadcast_mul_kernel(
    a_ptr, scale_ptr, out_ptr,
    total, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    a     = tl.load(a_ptr     + offs,       mask=mask, other=0.0)
    scale = tl.load(scale_ptr + (offs % N), mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * scale, mask=mask)


# ---------------------------------------------------------------------------
# Python-level helpers (not kernels)
# ---------------------------------------------------------------------------

def _run_fused_gemm_mul(w, x, gate):
    """out = (x @ w.T) * gate   [used by SmolLM and Gemma routes]
    Works for x of any shape [..., K]; no reshape/view is used."""
    M = x.numel() // x.shape[-1]
    K = x.shape[-1]
    N = w.shape[0]

    out = torch.empty(*x.shape[:-1], N, dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _fused_gemm_mul_kernel[grid](
        x, w, gate, out,
        M, N, K,
        x.stride(-2), x.stride(-1),
        w.stride(0),  w.stride(1),
        gate.stride(-2), gate.stride(-1),
        out.stride(-2),  out.stride(-1),
    )
    return out


def _run_gemm(w, x):
    """out = x @ w.T   [standalone GEMM for RTMPose linear]
    Works for x of any shape [..., K]; no reshape/view is used."""
    M = x.numel() // x.shape[-1]
    K = x.shape[-1]
    N = w.shape[0]

    out = torch.empty(*x.shape[:-1], N, dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _gemm_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(-2), x.stride(-1),
        w.stride(0),  w.stride(1),
        out.stride(-2), out.stride(-1),
    )
    return out


def _run_broadcast_mul(scale, a):
    """out = a * scale   [flat broadcast multiply; scale is 1-D of length N]
    Fixed BLOCK_SIZE=2048 avoids autotune overhead for small tensors."""
    total = a.numel()
    N     = scale.shape[0]
    out   = torch.empty_like(a)
    nblocks = (total + 2047) // 2048
    _broadcast_mul_kernel[(nblocks,)](a, scale, out, total, N, BLOCK_SIZE=2048)
    return out


# ---------------------------------------------------------------------------
# Universal dispatch  —  THE SINGLE shared replacement function
#
# Uses *args so each pass can pass its own number of tensor args + route str.
#
#  "smollm"       : args = (weight, input, gate,   "smollm")
#  "gemma"        : args = (weight, gate,  input,  "gemma")
#  "rtmpose_gemm" : args = (weight, input,         "rtmpose_gemm")
#  "rtmpose_mul"  : args = (scale,  tensor,        "rtmpose_mul")
# ---------------------------------------------------------------------------

@torch.fx.wrap
def dispatch_all(a0, a1, a2, route):
    if route == "smollm":
        # a0=weight [N,K], a1=input [...,K], a2=gate [...,N]
        return _run_fused_gemm_mul(a0, a1, a2)
    elif route == "gemma":
        # a0=weight [N,K], a1=gate [...,N], a2=input [...,K]
        return _run_fused_gemm_mul(a0, a2, a1)
    elif route == "rtmpose_gemm":
        # a0=weight [N,K], a1=input [...,K]   (a2 = dummy, never used)
        return _run_gemm(a0, a1)
    elif route == "rtmpose_mul":
        # a0=scale [N], a1=tensor [...,N]   (a2 = dummy, never used)
        return _run_broadcast_mul(a0, a1)
    else:
        return a0