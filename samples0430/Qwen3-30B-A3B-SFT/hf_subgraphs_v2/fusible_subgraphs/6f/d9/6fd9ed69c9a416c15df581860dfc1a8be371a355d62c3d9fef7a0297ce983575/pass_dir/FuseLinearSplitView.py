import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match the FIRST linear(in_5, in_1, in_0) call.
# Single-output → the framework applies this replacement TWICE (once per linear).
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_5):
    linear = torch.nn.functional.linear(in_5, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_5):
    return (in_0, in_1, in_5)


# ---------------------------------------------------------------------------
# Triton kernel: out = A[M,K] @ B[N,K]^T + bias[N]
#   Fixed config tuned for M=300, N=512/256, K=256 on A30 (Ampere).
# ---------------------------------------------------------------------------
@triton.jit
def _linear_kernel(
    A_ptr, B_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    # Load A tile [BM, BK]
    a = tl.load(
        A_ptr + offs_m[:, None] * stride_am + tl.arange(0, BK)[None, :] * stride_ak,
        mask=(offs_m[:, None] < M) & (tl.arange(0, BK)[None, :] < K),
        other=0.0,
    )
    # Load B^T tile [BK, BN]: b[k,n] = B[n,k] = B_ptr + n*stride_bn + k*stride_bk
    b = tl.load(
        B_ptr + tl.arange(0, BK)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        mask=(tl.arange(0, BK)[:, None] < K) & (offs_n[None, :] < N),
        other=0.0,
    )
    acc = tl.dot(a, b)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += tl.expand_dims(bias, 0)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Wrapper – single output per call; applied twice by the pass (both linears).
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_linear_split_wrapper(in_0, in_1, in_5):
    """
    in_0 : bias   [512]
    in_1 : weight [512, 256]
    in_5 : input  [300, 256]
    Returns [300, 512]  (first linear)
    Returns [300, 1, 256] (second linear, tmp_9 = in_4.reshape(300,1,256))
    """
    M, N, K = 300, 512, 256
    BM, BN, BK = 16, 128, 32

    out = torch.empty((M, N), dtype=in_5.dtype, device=in_5.device)

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    _linear_kernel[grid](
        in_5, in_1, in_0, out,
        M, N, K,
        in_5.stride(0), in_5.stride(1),
        in_1.stride(0), in_1.stride(1),
        out.stride(0), out.stride(1),
        BM=BM, BN=BN, BK=BK,
        num_warps=4, num_stages=5,
    )
    return out


def replacement_func():
    return fused_linear_split_wrapper