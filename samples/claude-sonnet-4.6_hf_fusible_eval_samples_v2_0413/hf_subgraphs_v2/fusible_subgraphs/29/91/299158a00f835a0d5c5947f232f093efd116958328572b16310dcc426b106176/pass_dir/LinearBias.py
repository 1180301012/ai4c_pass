import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton GEMM+bias for the tiny linear(in_2, in_1, in_0) classifier.
#   out[m, n] = sum_k( x[m,k] * w[n,k] ) + b[n]
#
# No @triton.autotune — fixed BLOCK_M / BLOCK_K removes per-call autotune
# cache-lookup overhead.  Grid is a pre-computed plain tuple (no lambda).
# ---------------------------------------------------------------------------

@triton.jit
def _linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m_blk = tl.program_id(0)
    n     = tl.program_id(1)

    m_off  = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_off < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off  = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        x = tl.load(
            x_ptr + m_off[:, None] * stride_xm + k_off[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        w = tl.load(
            w_ptr + n * stride_wn + k_off * stride_wk,
            mask=k_mask, other=0.0,
        )

        acc = acc + tl.sum(x.to(tl.float32) * w.to(tl.float32)[None, :], axis=1)

    bias   = tl.load(b_ptr + n)
    result = acc + bias.to(tl.float32)

    if IS_FP16:
        result = result.to(tl.float16)
    elif IS_BF16:
        result = result.to(tl.bfloat16)

    tl.store(out_ptr + m_off * N + n, result, mask=m_mask)


@torch.fx.wrap
def triton_linear(x, w, b):
    M = x.shape[0]
    K = x.shape[1]    # x is always 2-D [M, K]
    N = w.shape[0]    # weight is [N, K]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    IS_FP16 = x.dtype == torch.float16
    IS_BF16 = x.dtype == torch.bfloat16

    # Plain tuple grid — no lambda overhead
    grid = (triton.cdiv(M, 64), N)

    _linear_kernel[grid](
        x, w, b, out,
        M, N, K,
        K, 1,   # x strides: contiguous [M, K] → stride_xm=K, stride_xk=1
        K, 1,   # w strides: contiguous [N, K] → stride_wn=K, stride_wk=1
        IS_FP16, IS_BF16,
        64, 128,          # BLOCK_M, BLOCK_K as literal constexprs
    )

    return out


def pattern(x, w, b):
    return torch.nn.functional.linear(x, w, b)


def replacement_args(x, w, b):
    return (x, w, b)


def replacement_func():
    return triton_linear