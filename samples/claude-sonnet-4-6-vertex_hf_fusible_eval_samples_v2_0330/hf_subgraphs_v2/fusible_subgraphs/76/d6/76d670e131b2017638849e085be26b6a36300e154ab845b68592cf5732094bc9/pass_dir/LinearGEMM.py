import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches torch.nn.functional.linear(in_6, in_5, in_4) only
# ---------------------------------------------------------------------------
def pattern(in_4, in_5, in_6):
    return torch.nn.functional.linear(in_6, in_5, in_4)


def replacement_args(in_4, in_5, in_6):
    return (in_4, in_5, in_6)


# ---------------------------------------------------------------------------
# Triton GEMM + bias kernel
#   C[m, n] = sum_k A[m, k] * B[n, k]  +  bias[n]
#   A = input  [M, K]     (row-major: stride M=K, stride K=1)
#   B = weight [N, K]     (row-major: stride N=K, stride K=1)  ← needs transpose
#   bias       [N]
#   C = output [M, N]
# ---------------------------------------------------------------------------
@triton.jit
def gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]  — contiguous along K
        a = tl.load(
            a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )

        # Load B tile [BLOCK_N, BLOCK_K] from weight[N, K] — contiguous along K
        b = tl.load(
            b_ptr + n_offs[:, None] * stride_bn + k_offs[None, :] * stride_bk,
            mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
            other=0.0,
        )

        # acc += A @ B^T  →  [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, tl.trans(b))

    # Add bias
    bias = tl.load(bias_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store (Triton auto-converts float32 → pointer dtype)
    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(
        c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn,
        acc,
        mask=c_mask,
    )


# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def linear_triton(bias, weight, x):
    """
    bias   : [N]
    weight : [N, K]
    x      : [M, K]
    returns: [M, N]
    """
    M = x.shape[0]
    K = x.shape[1]
    N = weight.shape[0]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Fixed tile sizes tuned for M<=128, K=384, N=1000
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def replacement_func():
    return linear_triton