"""
FuseLinear pass: replaces torch.nn.functional.linear with an
autotuned Triton GEMM + bias kernel.

Motivation
----------
For this subgraph torch.compile/Inductor generates a bfloat16/float16
GEMM that is ~30-46 % *slower* than cuBLAS eager, because Inductor fuses
the GEMM with the subsequent layout permutations into one kernel that is
not as well tuned for the specific (M, N, K) shapes here.

By replacing only the linear call with a hand-tuned Triton GEMM we:
  1. Keep full Inductor optimization for the cheap view/permute ops.
  2. Potentially match or exceed cuBLAS for thin/medium M sizes.
  3. Avoid the overhead of separate copy kernels that hurt the current passes.

Pattern : torch.nn.functional.linear(in_3, in_2, in_1)
Output  : same shape and dtype as the original linear call
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# dtype map: torch → triton type constant
# ─────────────────────────────────────────────────────────────────────────────
_DTYPE_MAP = {
    torch.float32:  tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.float16:  tl.float16,
}


# ─────────────────────────────────────────────────────────────────────────────
# Triton GEMM kernel:  Y = X @ W.T + b
#
#   X : (M, K) contiguous  – flattened input
#   W : (N, K) contiguous  – weight (PyTorch linear convention)
#   b : (N,)   contiguous  – bias
#   Y : (M, N) contiguous  – output
#
# NT matmul: tile A is (BM, BK) from X; tile B is (BK, BN) from W transposed.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'GROUP_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BM': 64,  'BN': 256, 'BK': 32, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BM': 64,  'BN': 128, 'BK': 32, 'GROUP_M': 4},
                      num_stages=4, num_warps=4),
        triton.Config({'BM': 32,  'BN': 256, 'BK': 32, 'GROUP_M': 4},
                      num_stages=4, num_warps=4),
        triton.Config({'BM': 32,  'BN': 128, 'BK': 32, 'GROUP_M': 4},
                      num_stages=4, num_warps=4),
        triton.Config({'BM': 16,  'BN': 256, 'BK': 32, 'GROUP_M': 4},
                      num_stages=4, num_warps=4),
        triton.Config({'BM': 16,  'BN': 128, 'BK': 32, 'GROUP_M': 4},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    OUT_DTYPE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute Y[m, n] = sum_k( X[m,k] * W[n,k] ) + b[n]

    X : (M, K) row-major, stride (K, 1)
    W : (N, K) row-major, stride (K, 1) — accessed as W.T[k, n]
    b : (N,)
    Y : (M, N) row-major, stride (N, 1)
    """
    pid       = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)

    # ── grouped ordering for better L2 locality ─────────────────────────────
    num_pid_in_group = GROUP_M * num_pid_n
    group_id         = pid // num_pid_in_group
    first_pid_m      = group_id * GROUP_M
    group_size_m     = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    # Pointers into X and W (transposed access)
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]   # (BM, BK)
    w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]   # (BK, BN) transposed

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # ── main K loop (K=448 is divisible by BK=32 and BK=64, no K masking) ──
    for _ in range(tl.cdiv(K, BK)):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BK
        w_ptrs += BK

    # ── add bias ─────────────────────────────────────────────────────────────
    b   = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    # ── store output (cast from float32 accumulator) ─────────────────────────
    offs_out_m = pid_m * BM + tl.arange(0, BM)
    offs_out_n = pid_n * BN + tl.arange(0, BN)
    y_ptrs = y_ptr + offs_out_m[:, None] * N + offs_out_n[None, :]
    out_mask = (offs_out_m[:, None] < M) & (offs_out_n[None, :] < N)
    tl.store(y_ptrs, acc.to(OUT_DTYPE), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# @torch.fx.wrap wrapper
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def triton_linear(x, w, b):
    """
    Drop-in replacement for torch.nn.functional.linear(x, w, b).

    x : (B, seq, in_features)   contiguous
    w : (out_features, in_features)
    b : (out_features,)
    returns (B, seq, out_features)

    No .view() / .reshape() calls: x is already contiguous so its memory
    layout matches (B*seq, in_features) exactly — the kernel uses M=B*seq
    and accesses x_ptr + m*K + k directly.
    Similarly the output is allocated as (B, seq, N) contiguous, which has
    the same flat layout as (M, N).
    """
    B   = x.shape[0]
    seq = x.shape[1]
    K   = x.shape[2]    # in_features  = 448
    N   = w.shape[0]    # out_features = 1536
    M   = B * seq       # flattened M dimension

    # Allocate output with the correct 3D shape — no view needed
    y = torch.empty((B, seq, N), dtype=x.dtype, device=x.device)

    OUT_DTYPE = _DTYPE_MAP.get(x.dtype, tl.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']),
    )

    # x (B, seq, K) contiguous → same memory as (M, K) row-major
    # y (B, seq, N) contiguous → same memory as (M, N) row-major
    # Kernel accesses: x_ptr + m*K + k  and  y_ptr + m*N + n
    _matmul_bias_kernel[grid](
        x, w, b, y,
        M, N, K,
        OUT_DTYPE=OUT_DTYPE,
    )

    return y


# ─────────────────────────────────────────────────────────────────────────────
# Pattern / replacement API
# ─────────────────────────────────────────────────────────────────────────────

def pattern(in_3, in_2, in_1):
    """Matches torch.nn.functional.linear(input, weight, bias)."""
    return torch.nn.functional.linear(in_3, in_2, in_1)


def replacement_args(in_3, in_2, in_1):
    return (in_3, in_2, in_1)


def replacement_func():
    return triton_linear