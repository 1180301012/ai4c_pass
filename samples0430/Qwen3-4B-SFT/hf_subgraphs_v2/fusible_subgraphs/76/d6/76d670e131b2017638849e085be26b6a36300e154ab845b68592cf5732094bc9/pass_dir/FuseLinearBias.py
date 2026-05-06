import torch
import triton
import triton.language as tl


def pattern(input, weight, bias):
    return torch.nn.functional.linear(input, weight, bias)


def replacement_args(input, weight, bias):
    return (input, weight, bias)


@triton.jit
def _ln_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """out[m,n] = sum_k x[m,k]*w[n,k] + bias[n]   (i.e. F.linear)"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(0, K, BLOCK_K):
        offs_k = kb + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0,
        )
        b = tl.load(
            w_ptr + offs_k[:, None] + offs_n[None, :] * K,
            mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

    bias_val = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = (acc + bias_val[None, :]).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@torch.fx.wrap
def linear_triton(input, weight, bias):
    M = input.shape[0]
    K = input.shape[1]
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=input.dtype, device=input.device)
    _ln_kernel[
        (triton.cdiv(M, 32), triton.cdiv(N, 32))
    ](
        input, weight, bias, out,
        M, N, K,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=64,
        num_warps=4,
    )
    return out


def replacement_func():
    return linear_triton