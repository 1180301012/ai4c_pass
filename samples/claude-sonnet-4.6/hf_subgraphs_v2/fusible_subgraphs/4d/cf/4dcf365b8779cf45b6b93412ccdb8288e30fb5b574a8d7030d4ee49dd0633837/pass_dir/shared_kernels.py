"""
Shared Triton kernels + single dispatch_wrapper used by all three pass files.
By returning the SAME function object from replacement_func(), all passes stay
within the output_pass_replacement_func_limit.

Fixed (non-autotuned) kernels chosen for the specific matrix shapes:
  BigBird : M=17,  N=3072, K=768   → BM=16,BN=64,BK=128  → 96 blocks (1.7 waves on A30-56SM)
  RECT_L  : M=128, N=128,  K=128   → BM=16,BN=16,BK=128  → 64 blocks (1.14 waves)
"""
import torch
import triton
import triton.language as tl


# ── BigBird kernel  BM=16, BN=64, BK=128 (no autotune) ───────────────────────
# Grid: (ceil(17/16)=2, ceil(3072/64)=48) = 96 blocks, 1.7 waves on 56-SM A30
# K=768 as tl.constexpr → loop trip=6 is compile-time → enables loop unrolling

@triton.jit
def _bigbird_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N,                      # runtime: M=17 (needs mask), N=3072 (stride only)
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    IS_BF16: tl.constexpr,
    K: tl.constexpr    = 768,  # always 768 for BigBird → unroll-friendly
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 128,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # K is constexpr=768, BLOCK_K=128 → trip count 6 is compile-time constant
    for k_start in range(0, K // BLOCK_K):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w = tl.load(w_ptrs)
        acc = tl.dot(x, tl.trans(w), acc)
    b = tl.load(b_ptr + offs_n)
    acc = acc + b[None, :].to(tl.float32)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = offs_m[:, None] < M
    if IS_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


# ── RECT_L kernel  BM=16, BN=16, BK=128 (no autotune) ────────────────────────
# Grid: (128/16=8, 128/16=8) = 64 blocks, 1.14 waves on 56-SM A30

@triton.jit
def _rect_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 128,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # RECT_L: M=128=8*16, N=128=8*16, K=128=1*128 → ALL exact multiples → no masks needed
    offs_k = tl.arange(0, BLOCK_K)  # single K iteration (K=128=BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x = tl.load(x_ptrs)  # no mask needed
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    w = tl.load(w_ptrs)  # no mask needed
    acc = tl.dot(x, tl.trans(w), acc)
    b = tl.load(b_ptr + offs_n)  # no mask needed
    acc = acc + b[None, :].to(tl.float32)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    if IS_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16))  # no mask needed
    else:
        tl.store(out_ptrs, acc.to(tl.float16))  # no mask needed


# ── Single shared dispatch wrapper returned by ALL pass files ─────────────────

@torch.fx.wrap
def dispatch_wrapper(in_0, in_1, in_2, route):
    """
    Route dispatch for fused dropout+cast+linear kernels.
    in_0 = bias, in_1 = weight, in_2 = input activation.
    route: "bigbird" | "rect_fp16" | "rect_bf16"
    """
    if route == "bigbird":
        K = in_2.shape[-1]
        N = in_1.shape[0]
        M = 1
        for d in in_2.shape[:-1]:
            M = M * d
        stride_xm = in_2.stride(-2)
        stride_xk = in_2.stride(-1)
        out_shape = list(in_2.shape[:-1]) + [N]
        output  = torch.empty(out_shape, dtype=in_2.dtype, device=in_2.device)
        is_bf16 = (in_2.dtype == torch.bfloat16)
        # K removed from runtime args — it's constexpr=768 by default
        _bigbird_linear_kernel[
            (triton.cdiv(M, 16), triton.cdiv(N, 64))
        ](
            in_2, in_1, in_0, output,
            M, N,                   # K is constexpr default 768
            stride_xm, stride_xk,
            in_1.stride(0), in_1.stride(1),
            IS_BF16=is_bf16,
        )
        return output

    elif route == "rect_fp16":
        M = in_2.shape[0]
        K = in_2.shape[1]
        N = in_1.shape[0]
        output = torch.empty((M, N), dtype=torch.float16, device=in_2.device)
        # Fixed grid matching BN=16: (ceil(M/16), ceil(N/16)) = 64 blocks for M=N=128
        _rect_linear_kernel[
            (triton.cdiv(M, 16), triton.cdiv(N, 16))
        ](
            in_2, in_1, in_0, output,
            M, N, K,
            in_2.stride(0), in_2.stride(1),
            in_1.stride(0), in_1.stride(1),
            IS_BF16=False,
        )
        return output

    elif route == "rect_bf16":
        M = in_2.shape[0]
        K = in_2.shape[1]
        N = in_1.shape[0]
        output = torch.empty((M, N), dtype=torch.bfloat16, device=in_2.device)
        # Fixed grid matching BN=16: (ceil(M/16), ceil(N/16)) = 64 blocks for M=N=128
        _rect_linear_kernel[
            (triton.cdiv(M, 16), triton.cdiv(N, 16))
        ](
            in_2, in_1, in_0, output,
            M, N, K,
            in_2.stride(0), in_2.stride(1),
            in_1.stride(0), in_1.stride(1),
            IS_BF16=True,
        )
        return output