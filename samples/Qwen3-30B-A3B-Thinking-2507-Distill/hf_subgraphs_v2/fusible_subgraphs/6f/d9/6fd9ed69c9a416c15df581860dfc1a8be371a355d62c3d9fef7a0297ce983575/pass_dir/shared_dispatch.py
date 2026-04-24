"""
Shared Triton GEMM kernels and dispatch wrapper for all passes.

Both FuseLinearSplitView_fp32 and FuseReshapeLinearSplitView_fp16 import
_dispatch_replacement from here so replacement_func() returns the SAME
function object, satisfying the framework's replacement_func_limit.

Route strings:
  "fp32"  -> FuseLinearSplitView_fp32  (F.linear first-half/second-half)
  "fp16"  -> FuseReshapeLinearSplitView_fp16  (reshape + F.linear, full output)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton GEMM + bias kernel: out = A @ B.T + bias
# A: [M, K]  B: [N, K] (weight rows)  out: [M, N]
# B loaded as [BLOCK_N, BLOCK_K] (coalesced), transposed in tl.dot.
# Fixed config: BLOCK_M=32, BLOCK_N=64, BLOCK_K=64 — good GPU utilization.
# ---------------------------------------------------------------------------
@triton.jit
def _shared_gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # A tile [BLOCK_M, BLOCK_K] — coalesced along K
        a = tl.load(
            A_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        # B tile [BLOCK_N, BLOCK_K] — weight rows, coalesced along K
        b = tl.load(
            B_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        # acc += A_tile @ B_tile^T  (fp32 accumulation)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)

    # Bias broadcast over M
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :].to(tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Shared dispatch wrapper — SAME object imported by all pass files.
# route is a string constant: "fp32" or "fp16" (never None).
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _dispatch_replacement(a, b, c, route):
    """
    a     : bias   [N]
    b     : weight [N, K]
    c     : input  [*, K]   (any shape; treated as [M, K])
    route : "fp32" | "fp16"

    Computes out = c @ b.T + a  (in float32, cast to c.dtype) and returns out.
    """
    M = c.shape[0]
    K = c.shape[1]
    N = b.shape[0]

    out = torch.empty((M, N), dtype=c.dtype, device=c.device)

    # BLOCK_M=32, BLOCK_N=128, BLOCK_K=64 → stable compilation, 40 blocks for
    # first linear (M=300,N=512) and 20 blocks for second (M=300,N=256)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _shared_gemm_bias_kernel[grid](
        c, b, a, out,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out