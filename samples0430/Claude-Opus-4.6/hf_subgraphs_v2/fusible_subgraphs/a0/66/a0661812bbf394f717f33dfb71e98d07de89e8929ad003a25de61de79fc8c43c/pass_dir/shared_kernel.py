import torch
import triton
import triton.language as tl


@triton.jit
def simple_gemm_kernel(
    a_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute out[m,n] = sum_k A[m,k] * W[n,k] + bias[n]"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    wt_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        wt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        acc += tl.dot(a, wt)

        a_ptrs += BLOCK_K
        wt_ptrs += BLOCK_K
        offs_k += BLOCK_K

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


@torch.fx.wrap
def fused_linear_noop_dropout(bias, weight, input_tensor):
    """Fused linear + dropout(noop) returning single output."""
    M = input_tensor.shape[1]
    K = input_tensor.shape[2]
    N = weight.shape[0]

    out = torch.empty((1, M, N), dtype=input_tensor.dtype, device=input_tensor.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    if N <= 16:
        BLOCK_N = 16
    elif N <= 32:
        BLOCK_N = 32

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    simple_gemm_kernel[(grid_m, grid_n)](
        input_tensor, weight, bias, out,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out