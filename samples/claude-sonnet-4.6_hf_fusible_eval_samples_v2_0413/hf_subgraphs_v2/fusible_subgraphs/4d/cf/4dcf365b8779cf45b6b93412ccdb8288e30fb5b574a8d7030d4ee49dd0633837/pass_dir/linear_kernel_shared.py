import torch
import triton
import triton.language as tl


@triton.jit
def gemm_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    IS_FP16: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Computes: out = x @ W^T + bias
      x   : [M, K]  stride (stride_xm, stride_xk)
      W   : [N, K]  stride (stride_wn, stride_wk)   linear weight [N, K]
      bias: [N]
      out : [M, N]  stride (stride_om, stride_on)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Load x block: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # stride_xk=1 → coalesced access along K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W block as [BLOCK_SIZE_N, BLOCK_SIZE_K]
        # stride_wk=1 → coalesced access along K (much better than loading W^T directly)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # x_block [BLOCK_M, BLOCK_K] @ tl.trans(w_block) [BLOCK_K, BLOCK_N]
        acc += tl.dot(x_block, tl.trans(w_block))

    # Add bias [N] broadcast over M
    b_mask = offs_n < N
    bias = tl.load(b_ptr + offs_n, mask=b_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store output with appropriate dtype cast
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if IS_FP16:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@torch.fx.wrap
def fused_linear_wrapper(bias, weight, x):
    """
    Fused dropout (no-op, training=False) + optional cast + linear replacement.
    Computes: out = x @ weight.T + bias
    Handles any leading batch dimensions in x via stride arithmetic (no view/reshape).
    Manually dispatches block-size configs based on problem shape to avoid
    autotuner JIT-compilation overhead during the benchmark window.
    """
    K = x.shape[-1]
    M = x.numel() // K
    N = weight.shape[0]

    # Allocate output preserving all leading dims, replacing last dim with N
    out = torch.empty(*x.shape[:-1], N, dtype=x.dtype, device=x.device)

    is_fp16 = (x.dtype == torch.float16)

    # ---- Config dispatch ----
    # bigbird: M=17, K=768, N=3072  → large-N tiles (2×48=96 blocks > 56 SMs)
    # RECT_L:  M=128, K=128, N=128  → small tiles  (8×8=64 blocks ≈ 56 SMs)
    if N <= 256:
        # Small-N path (RECT_L-like): BLOCK16×16, single K-pass → 64 tiles for 128×128 grid
        # BK=128, K=128 → 1 loop iteration, no pipelining overhead
        BM, BN, BK = 16, 16, 128
        NUM_STAGES, NUM_WARPS = 1, 4
    else:
        # Large-N path (bigbird-like): BLOCK16×64 → 96 tiles for 17×3072 grid
        # BK=32 with stages=4: 24 K-iterations, deep pipeline hides memory latency well
        BM, BN, BK = 16, 64, 32
        NUM_STAGES, NUM_WARPS = 4, 4

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(-2), x.stride(-1),
        weight.stride(0), weight.stride(1),
        out.stride(-2), out.stride(-1),
        IS_FP16=is_fp16,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        num_stages=NUM_STAGES, num_warps=NUM_WARPS,
    )

    return out