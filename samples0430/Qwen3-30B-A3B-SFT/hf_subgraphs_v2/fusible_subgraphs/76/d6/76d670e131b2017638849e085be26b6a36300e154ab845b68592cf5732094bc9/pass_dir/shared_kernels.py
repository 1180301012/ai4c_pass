"""Shared Triton kernels and unified dispatch wrapper for all passes."""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 1.  Linear Layer:  out = x @ w.T + bias
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # Fp16/Bf16 safe configs  (smem ≈ num_stages*(BM*BK + BN*BK)*2 bytes)
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=2, num_warps=4),
        # Fp32 safe configs  (smem ≈ num_stages*(BM*BK+BN*BK)*4 bytes)
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=1, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # x tile  [BLOCK_M, BLOCK_K]
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        # w tile  [BLOCK_N, BLOCK_K]  (weight stored as [N, K])
        w = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        )

        # acc += x @ w.T
        acc = tl.dot(x.to(tl.float32), tl.trans(w).to(tl.float32), acc=acc)

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :].to(tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ---------------------------------------------------------------------------
# 2.  Batch-Norm inference:  y = (x - mean) / sqrt(var+eps) * w + b
#     Fused form:  y = x * scale + shift   (scale, shift pre-computed)
#
# No @triton.autotune: avoids periodic re-benchmarking which causes timing
# instability.  Fixed tile sizes give stable, predictable performance.
#
#   BLOCK_M = 128 →  only 1 program needed for M ≤ 128  (minimal overhead)
#   BLOCK_N = 512 →  covers N=384 in ONE pass (no inner loop)
# ---------------------------------------------------------------------------
@triton.jit
def _bn_kernel(
    x_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)   # block of rows

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Pre-compute per-channel scale and shift once per program
    mean   = tl.load(mean_ptr   + offs_n, mask=mask_n, other=0.0)
    var    = tl.load(var_ptr    + offs_n, mask=mask_n, other=1.0)
    w      = tl.load(weight_ptr + offs_n, mask=mask_n, other=1.0)
    b      = tl.load(bias_ptr   + offs_n, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(var.to(tl.float32) + 1e-5)
    scale   = w.to(tl.float32) * inv_std
    shift   = b.to(tl.float32) - mean.to(tl.float32) * scale

    # Iterate over the BLOCK_M rows handled by this program
    for i in range(BLOCK_M):
        row_id = pid_m * BLOCK_M + i
        mask_m = row_id < M
        x = tl.load(x_ptr + row_id * N + offs_n, mask=mask_n & mask_m, other=0.0)
        y = x.to(tl.float32) * scale + shift
        tl.store(out_ptr + row_id * N + offs_n, y, mask=mask_n & mask_m)


# ---------------------------------------------------------------------------
# Shared wrapper  (returned by replacement_func() in every pass file)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _unified_dispatch(*args):
    route = args[-1]

    if route == "linear":
        x, weight, bias = args[0], args[1], args[2]
        x  = x.contiguous()
        w  = weight.contiguous()
        M, K = x.shape
        N    = w.shape[0]
        out  = torch.empty((M, N), dtype=x.dtype, device=x.device)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
        _linear_kernel[grid](x, w, bias, out, M, N, K)
        return out

    elif route == "batch_norm":
        x, running_mean, running_var, weight, bias = (
            args[0], args[1], args[2], args[3], args[4],
        )
        N   = x.shape[-1]
        M   = x.numel() // N
        out = torch.empty_like(x)

        # BLOCK_M=64, BLOCK_N=512: integer shift avoids triton.cdiv call overhead
        BLOCK_M = 64
        BLOCK_N = 512
        grid = ((M + 63) >> 6,)          # ceil(M / 64)
        _bn_kernel[grid](x, running_mean, running_var, weight, bias, out, M, N,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        return out