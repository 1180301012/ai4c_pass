"""
Fused pass: conv2d (1x1 kernel, 1x1 spatial) + hardswish + flatten

Strategy:
  - 1x1 conv with 1x1 spatial dims = linear (GEMM + bias)
  - Fused Triton GEMM + bias + hardswish with FIXED block sizes (no autotune).
  - No autotune: eliminates the autotune overhead that inflates mean GPU times
    for 5 of 6 evaluation graphs (different dtype/M combos each trigger a fresh
    autotuner run during the trial period, causing severe mean inflation).
  - Fixed config: BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, GROUP_M=4.
    For M=32, N=1280: gives 2×40=80 blocks (1.4 waves on 56-SM A30).
    For M=1,  N=1280: gives 1×40=40 blocks (<1 wave but no autotune noise).
  - Output directly as [B, N] so flatten(1,-1) is implicit.

Pattern:
    conv2d = torch.conv2d(x, W, b, (1,1), (0,0), (1,1), 1)   # [B,N,1,1]
    hs     = hardswish(conv2d, inplace=True)                    # [B,N,1,1]
    out    = hs.flatten(1, -1)                                  # [B,N]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    """
    in_0 : bias   [N]
    in_1 : weight [N, C_in, 1, 1]
    in_2 : input  [B, C_in, 1, 1]
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton GEMM + bias + hardswish kernel  (NO autotune — fixed block sizes)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_gemm_hardswish_kernel(
    A_ptr, W_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    C[m, n] = hardswish( A[m,:] @ W[n,:]^T + bias[n] )

    Fixed tiling: BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, GROUP_M=4, warps=8
    → 80 blocks for M=32, N=1280 (1.4 waves on 56-SM A30).
    8 warps per block → 8 concurrent L1 miss streams → hides memory latency.
    W is [N,K] row-major; its 2.46 MB fits in A30's 36 MB L2, so the
    non-coalesced [BLOCK_K, BLOCK_N] load is served from L2 (fast).
    EVEN_K=True eliminates K-boundary masking when K % BLOCK_K == 0.
    """
    # ── L2-cache-friendly pid swizzle ──────────────────────────────────
    pid          = tl.program_id(0)
    num_pid_m    = tl.cdiv(M, BLOCK_M)
    num_pid_n    = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    group_id     = pid // num_in_group
    first_m      = group_id * GROUP_M
    gs_m         = min(num_pid_m - first_m, GROUP_M)
    pid_m        = first_m + (pid % num_in_group) % gs_m
    pid_n        = (pid % num_in_group) // gs_m

    # ── Offsets ────────────────────────────────────────────────────────
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ── Pointers ───────────────────────────────────────────────────────
    # A[BLOCK_M, BLOCK_K]  stride_ak=1 → coalesced in K
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # W^T[BLOCK_K, BLOCK_N]  from W[N,K] row-major (L2-cached on A30 36 MB)
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    mask_m = offs_m[:, None] < M   # [BLOCK_M, 1]
    mask_n = offs_n[None, :] < N   # [1, BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        if EVEN_K:
            # K divisible by BLOCK_K: skip K-boundary masking (faster inner loop)
            a = tl.load(a_ptrs, mask=mask_m, other=0.0)
            w = tl.load(w_ptrs, mask=mask_n, other=0.0)
        else:
            k_rem    = K - k
            a = tl.load(a_ptrs, mask=mask_m & (offs_k[None, :] < k_rem), other=0.0)
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_rem) & mask_n,  other=0.0)

        acc += tl.dot(a, w, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # ── Bias + hardswish ───────────────────────────────────────────────
    bias  = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc   = acc + bias[None, :].to(tl.float32)

    x_p3   = acc + 3.0
    relu6  = tl.minimum(tl.maximum(x_p3, 0.0), 6.0)
    result = acc * relu6 * (1.0 / 6.0)

    # ── Store ──────────────────────────────────────────────────────────
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, result.to(C_ptr.dtype.element_ty),
             mask=mask_m & mask_n)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

# Fixed block sizes — chosen for best throughput on A30 for M∈{1,32}, N=1280
_BLOCK_M  = 16
_BLOCK_N  = 32
_BLOCK_K  = 64
_GROUP_M  = 4
_NUM_WRPS = 8
_NUM_STGS = 4


@torch.fx.wrap
def fused_conv1x1_hardswish(bias, weight, x):
    """
    Args
    ----
    bias   : Tensor [N]              (in_0)
    weight : Tensor [N, C_in, 1, 1] (in_1)
    x      : Tensor [B, C_in, 1, 1] (in_2)

    Returns
    -------
    Tensor [B, N]  (conv1x1 -> hardswish -> flatten fused)
    """
    B = x.shape[0]
    K = x.shape[1]       # C_in  (960)
    N = weight.shape[0]  # C_out (1280)

    A = x.reshape(B, K)       # [B, K]
    W = weight.reshape(N, K)  # [N, K]

    output = torch.empty((B, N), dtype=x.dtype, device=x.device)

    grid = (triton.cdiv(B, _BLOCK_M) * triton.cdiv(N, _BLOCK_N),)

    _fused_gemm_hardswish_kernel[grid](
        A, W, bias, output,
        B, N, K,
        A.stride(0), A.stride(1),
        W.stride(0), W.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K, GROUP_M=_GROUP_M,
        EVEN_K=(K % _BLOCK_K == 0),
        num_warps=_NUM_WRPS, num_stages=_NUM_STGS,
    )

    return output


# ---------------------------------------------------------------------------
# Replacement entry-point
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_conv1x1_hardswish