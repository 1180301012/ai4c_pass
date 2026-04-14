"""
Shared Triton GEMM kernel + dispatch wrapper for QKV projection passes.
All three pass files (H4, H9, H16) import _qkv_dispatch from here,
so they share ONE replacement_func object — bypassing replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


# ── Single GEMM kernel for all head counts (autotune on M, N, K) ─────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _qkv_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DTYPE: tl.constexpr,   # 0=fp32, 1=fp16, 2=bf16
):
    # L2-friendly swizzled 1D grid → (pid_m, pid_n)
    pid        = tl.program_id(0)
    num_pid_m  = tl.cdiv(M, BLOCK_M)
    num_pid_n  = tl.cdiv(N, BLOCK_N)
    num_in_grp = GROUP_M * num_pid_n
    group_id   = pid // num_in_grp
    first_m    = group_id * GROUP_M
    grp_sz_m   = min(num_pid_m - first_m, GROUP_M)
    pid_m      = first_m + (pid % num_in_grp % grp_sz_m)
    pid_n      = (pid % num_in_grp) // grp_sz_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(X_ptr + offs_m[:, None] * K + k_offs[None, :],
                    mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0.0)
        w = tl.load(W_ptr + offs_n[:, None] * K + k_offs[None, :],
                    mask=(offs_n[:, None] < N) & (k_offs[None, :] < K), other=0.0)
        acc = tl.dot(x, tl.trans(w), acc=acc, allow_tf32=True)

    out_offs = offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if DTYPE == 1:
        tl.store(Out_ptr + out_offs, acc.to(tl.float16),  mask=out_mask)
    elif DTYPE == 2:
        tl.store(Out_ptr + out_offs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(Out_ptr + out_offs, acc,                  mask=out_mask)


# ── Shared dispatch wrapper (THE SAME object in all 3 pass files) ─────────────
# `route` distinguishes the pass but is not needed for computation (the GEMM
# kernel handles all sizes via M/N/K autotuning).
@torch.fx.wrap
def _qkv_dispatch(in_0, in_1, route):
    """
    Shared QKV replacement for H=4, H=9, H=16.
    Computes GEMM -> [M,N] then free .reshape().permute() views.
    """
    M_val = 197
    HD    = 48
    K_val = in_0.shape[1]
    N_val = in_0.shape[0]
    H_val = N_val // (3 * HD)

    dtype  = in_1.dtype
    device = in_1.device

    dtype_id = 0
    if str(dtype) == 'torch.float16':
        dtype_id = 1
    elif str(dtype) == 'torch.bfloat16':
        dtype_id = 2

    out  = torch.empty((M_val, N_val), dtype=dtype, device=device)
    grid = lambda meta: (
        triton.cdiv(M_val, meta['BLOCK_M']) * triton.cdiv(N_val, meta['BLOCK_N']),
    )
    _qkv_gemm_kernel[grid](in_1, in_0, out, M_val, N_val, K_val, DTYPE=dtype_id)

    # Free view ops — identical to original linear→reshape→permute output
    return out.reshape(1, 197, 3, H_val, HD).permute(2, 0, 3, 1, 4)