"""
Shared Triton kernels for linear-gate fusion and broadcast-multiply.

Conventions
-----------
All GEMM kernels compute  C = A @ B^T  where:
  A  is stored as [M, K]  with strides (stride_am, stride_ak)
  B  is stored as [N, K]  with strides (stride_bn, stride_bk)  (row-major weight)
  C  is stored as [M, N]  with strides (stride_cm, stride_cn)

This matches  torch.nn.functional.linear(x, w, None)  which returns  x @ w^T.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 1.  Fused GEMM + element-wise gate multiply
#     out[m,n] = ( sum_k A[m,k]*B[n,k] ) * gate[m,n]
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Best for large M — SmolLM3 M=8192, N=11008
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        # Medium M — SmolLM3 M=2048
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        # Tiny M — SmolLM3 M=2-64, gemma M=3
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_matmul_gate_kernel(
    a_ptr, b_ptr, gate_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_gm, stride_gn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # --- program-ID swizzling for L2 cache reuse ---
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id      = pid // num_pid_in_group
    first_pid_m   = group_id * GROUP_M
    group_size_m  = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- tile offsets ---
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k  = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk

    # --- accumulate ---
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs,
                    mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_bn[None, :] < N) & (offs_k[:, None] < k_remaining),
                    other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- load gate and fuse multiply ---
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask   = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    gate_ptrs = gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    gate = tl.load(gate_ptrs, mask=mask, other=0.0)

    # multiply: keep float32 precision, then cast back to gate dtype
    result = (acc * gate.to(tl.float32)).to(gate.dtype)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result, mask=mask)


# ---------------------------------------------------------------------------
# 2.  Standalone GEMM (used for the rtmpose-l linear branch)
#     out[m,n] = sum_k A[m,k]*B[n,k]
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_om, stride_on,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id      = pid // num_pid_in_group
    first_pid_m   = group_id * GROUP_M
    group_size_m  = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k  = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs,
                    mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_bn[None, :] < N) & (offs_k[:, None] < k_remaining),
                    other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask   = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(DTYPE), mask=mask)


# ---------------------------------------------------------------------------
# 3.  Broadcast-multiply:  out[m,n] = inp[m,n] * scale[n]
#     inp is [M, N], scale is 1-D [N]
# ---------------------------------------------------------------------------

@triton.jit
def _broadcast_mul_1d_kernel(
    inp_ptr, scale_ptr, out_ptr,
    M, N,
    stride_im, stride_in,
    stride_sn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask   = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    inp   = tl.load(inp_ptr   + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in,
                    mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offs_n[None, :] * stride_sn,
                    mask=offs_n[None, :] < N, other=0.0)

    result = inp * scale
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             result, mask=mask)


# ---------------------------------------------------------------------------
# Python-level helper wrappers (internal; not FX-wrapped)
# ---------------------------------------------------------------------------

_TORCH_TO_TRITON_DTYPE = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


def _fused_linear_gate(w, gate, x):
    """Compute (x @ w^T) * gate using raw strides — no reshape."""
    N  = w.shape[0]
    K  = x.shape[-1]
    M  = x.numel() // K

    out = torch.empty(x.shape[:-1] + (N,), dtype=x.dtype, device=x.device)

    # For a contiguous [..., K] tensor, stride(-2) is the "row" stride when
    # viewed as [M, K] regardless of how many leading batch dims exist.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _fused_matmul_gate_kernel[grid](
        x, w, gate, out,
        M, N, K,
        x.stride(-2),    x.stride(-1),
        w.stride(0),     w.stride(1),
        gate.stride(-2), gate.stride(-1),
        out.stride(-2),  out.stride(-1),
    )
    return out


def _triton_linear(w, x):
    """Compute x @ w^T via Triton GEMM — no reshape."""
    N  = w.shape[0]
    K  = x.shape[-1]
    M  = x.numel() // K

    out = torch.empty(x.shape[:-1] + (N,), dtype=x.dtype, device=x.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(-2),   x.stride(-1),
        w.stride(0),    w.stride(1),
        out.stride(-2), out.stride(-1),
        DTYPE=_TORCH_TO_TRITON_DTYPE[x.dtype],
    )
    return out


def _triton_broadcast_mul_1d(inp, scale):
    """Compute inp * scale (1-D broadcast) — no reshape."""
    N = inp.shape[-1]
    M = inp.numel() // N

    out = torch.empty_like(inp)

    BLOCK_M = 32
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _broadcast_mul_1d_kernel[grid](
        inp, scale, out,
        M, N,
        inp.stride(-2), inp.stride(-1),
        scale.stride(0),
        out.stride(-2), out.stride(-1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


# ---------------------------------------------------------------------------
# Shared dispatch wrapper (THE SINGLE replacement_func target for all passes)
#
# All pass files import and return this EXACT function object so the
# framework counts only one unique replacement_func.
#
# Call convention (5 positional args):
#   a0, a1      — primary tensor arguments for all routes
#   a2          — third tensor (for fused gate routes) or None
#   a3          — fourth tensor or None
#   route       — string constant that selects the computation path
#
# Routes:
#   "smollm3"        : (a0=w, a1=gate, a2=x)   → (x @ w^T) * gate   [single tensor]
#   "gemma"          : (a0=w, a1=gate, a2=x)   → (x @ w^T) * gate   [single tensor]
#   "tritonlinear"   : (a0=w, a1=x)            → x @ w^T            [single tensor]
#   "broadcastscalemul": (a0=inp, a1=scale)    → inp * scale        [single tensor]
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _shared_dispatch(a0, a1, a2, a3, route):
    if route == "smollm3":
        # a0=weight, a1=gate, a2=x(linear_input), a3=None (unused)
        return _fused_linear_gate(a0, a1, a2)
    elif route == "gemma":
        # a0=weight, a1=gate, a2=x(linear_input), a3=None (unused)
        return _fused_linear_gate(a0, a1, a2)
    elif route == "tritonlinear":
        # a0=weight, a1=x(linear_input), a2=None, a3=None
        return _triton_linear(a0, a1)
    elif route == "broadcastscalemul":
        # a0=inp[*, N], a1=scale[N] — inlined for minimum Python overhead
        N   = a0.shape[-1]
        M   = a0.numel() // N
        out = torch.empty_like(a0)
        BLOCK_M = 32
        BLOCK_N = 128
        _broadcast_mul_1d_kernel[
            (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        ](
            a0, a1, out, M, N,
            N, 1,            # inp strides (contiguous: stride_im=N, stride_in=1)
            1,               # scale stride (contiguous 1-D)
            N, 1,            # out strides
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        return out
    # Fallback (never reached in practice)
    return _fused_linear_gate(a0, a1, a2)