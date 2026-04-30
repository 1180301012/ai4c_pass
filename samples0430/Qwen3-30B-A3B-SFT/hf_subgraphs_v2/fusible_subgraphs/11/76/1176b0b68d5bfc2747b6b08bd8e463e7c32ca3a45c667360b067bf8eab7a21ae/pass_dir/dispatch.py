"""
Shared dispatch module for all batched-matmul passes.

Two routes:
  "gemm"    – general GEMM (B,H,D,N) @ (B,H,N,N) -> (B,H,D,N)
  "gemv"    – GEMV when N_out=1: (B,H,M,K) @ (B,H,K,1) -> (B,H,M,1)
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# I  –  GEMM kernel  [B,H,M,K] @ [B,H,K,N] → [B,H,M,N]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # Large tiles (good for large M / large BH)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        # Medium tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        # BLOCK_N=256 – covers N=400 in 2 tiles (2nd tile masked) → fewer waves!
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        # Small BLOCK_M – more parallelism along M
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        # More pipeline depth
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_kernel(
    x_ptr, y_ptr, z_ptr,
    M, N, K,
    stride_xb, stride_xh, stride_xm, stride_xk,
    stride_yb, stride_yh, stride_yk, stride_yn,
    stride_zb, stride_zh, stride_zm, stride_zk,
    NUM_H_HEADS,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid    = tl.program_id(0)
    pid_b  = tl.program_id(1)

    b_idx = pid_b // NUM_H_HEADS
    h_idx = pid_b %  NUM_H_HEADS

    num_pid_m  = tl.cdiv(M, BLOCK_M)
    num_pid_n  = tl.cdiv(N, BLOCK_N)
    num_in_grp = GROUP_M * num_pid_n
    group_id   = pid // num_in_grp
    first_m    = group_id * GROUP_M
    g_size_m   = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m      = first_m + (pid % g_size_m)
    pid_n      = (pid % num_in_grp) // g_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_base = x_ptr + b_idx * stride_xb + h_idx * stride_xh
    y_base = y_ptr + b_idx * stride_yb + h_idx * stride_yh
    z_base = z_ptr + b_idx * stride_zb + h_idx * stride_zh

    x_ptrs = x_base + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    y_ptrs = y_base + offs_k[:, None] * stride_yk + offs_n[None, :] * stride_yn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_rem = K - k_start
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        y = tl.load(y_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, y)
        x_ptrs += BLOCK_K * stride_xk
        y_ptrs += BLOCK_K * stride_yk

    z_ptrs = z_base + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zk
    tl.store(z_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _run_gemm(in_0, in_1):
    """
    in_1: [B, H, M, K]   (values / projected queries)
    in_0: [B, H, K, N]   (attention / keys transposed)
    returns: [B, H, M, N]
    """
    B, H, M, K = in_1.shape[0], in_1.shape[1], in_1.shape[2], in_1.shape[3]
    N          = in_0.shape[3]
    BH         = B * H
    out        = torch.empty((B, H, M, N), dtype=in_1.dtype, device=in_1.device)
    grid       = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), BH)
    _gemm_kernel[grid](
        in_1, in_0, out,
        M, N, K,
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
        BH,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# II  –  GEMV kernel  [B,H,M,K] @ [B,H,K,1] → [B,H,M,1]
#       N_out=1 specialised; no tl.constexpr params to avoid binder conflicts.
#       All block sizes are Python integer literals inside the kernel body
#       (Triton JIT treats them as compile-time constants automatically).
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _gemv_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_ab, stride_ah, stride_am, stride_ak,
    stride_bb, stride_bh, stride_bk,
    stride_cb, stride_ch, stride_cm,
    NUM_H_HEADS,
    num_stages,        # accepted from framework's 17th positional arg
):
    """
    c[b,h,m,0] = sum_k a[b,h,m,k] * b[b,h,k,0]
    Grid: (ceil(M/128), BH)   — BLOCK_M=128 hardcoded
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    b_idx = pid_b // NUM_H_HEADS
    h_idx = pid_b %  NUM_H_HEADS

    offs_m = pid_m * 64 + tl.arange(0, 64)   # BLOCK_M = 64
    offs_k = tl.arange(0, 128)                  # BLOCK_K = 128

    a_base = a_ptr + b_idx * stride_ab + h_idx * stride_ah
    b_base = b_ptr + b_idx * stride_bb + h_idx * stride_bh
    c_base = c_ptr + b_idx * stride_cb + h_idx * stride_ch  # NEW: c_base

    acc = tl.zeros((64,), dtype=tl.float32)
    for k_start in range(0, K, 256):
        k_mask = offs_k < (K - k_start)
        a = tl.load(a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_base + offs_k * stride_bk, mask=k_mask, other=0.0)
        acc += tl.sum(a * b[None, :], axis=1)

    tl.store(c_base + offs_m * stride_cm,
             acc.to(c_ptr.dtype.element_ty),
             mask=offs_m < M)


def _run_gemv(in_0, in_1):
    """
    in_1: [B, H, M, K]  (e.g. [B,1,512,4096])
    in_0: [B, H, K, 1]  (e.g. [B,1,4096,1])
    returns: [B, H, M, 1]
    """
    B, H, M, K = in_1.shape[0], in_1.shape[1], in_1.shape[2], in_1.shape[3]
    BH  = B * H
    out = torch.empty((B, H, M, 1), dtype=in_1.dtype, device=in_1.device)
    # BLOCK_M=64 → better GPU occupancy (more CTAs for M≤512 models)
    grid = (triton.cdiv(M, 64), BH)
    _gemv_kernel[grid](
        in_1, in_0, out,
        M, K,
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        out.stride(0),  out.stride(1),  out.stride(2),
        BH,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# III – Shared dispatch wrapper (returned by replacement_func() in BOTH passes)
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def dispatch(in_0, in_1, route):
    if route == "gemv":
        return _run_gemv(in_0, in_1)
    else:  # "gemm"
        return _run_gemm(in_0, in_1)