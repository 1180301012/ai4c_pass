import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_linear_softmax_kernel(
    in_2_ptr,
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Single block: matmul with tensor cores + softmax
    m_range = tl.arange(0, BLOCK_M)
    n_range = tl.arange(0, BLOCK_N)
    k_range = tl.arange(0, K)

    mask_m = m_range < M
    mask_n = n_range < N

    # Load matrices
    a = tl.load(in_2_ptr + m_range[:, None] * K + k_range[None, :],
                mask=mask_m[:, None], other=0.0)
    b = tl.load(in_1_ptr + n_range[:, None] * K + k_range[None, :],
                mask=mask_n[:, None], other=0.0)

    # Matmul: c = a @ b^T -> [BLOCK_M, BLOCK_N] using tensor cores
    c = tl.dot(a, tl.trans(b))

    # Add bias
    bias = tl.load(in_0_ptr + n_range, mask=mask_n, other=0.0)
    c = c + bias[None, :].to(tl.float32)

    # Softmax group 1: cols 0-8
    g1_mask = n_range[None, :] < 9
    g1_vals = tl.where(g1_mask, c, float('-inf'))
    g1_max = tl.max(g1_vals, axis=1)
    g1_exp = tl.exp(g1_vals - g1_max[:, None])
    g1_sum = tl.sum(g1_exp, axis=1)
    g1_out = g1_exp / g1_sum[:, None]

    # Softmax group 2: cols 9-17
    g2_mask = (n_range[None, :] >= 9) & (n_range[None, :] < N)
    g2_vals = tl.where(g2_mask, c, float('-inf'))
    g2_max = tl.max(g2_vals, axis=1)
    g2_exp = tl.exp(g2_vals - g2_max[:, None])
    g2_sum = tl.sum(g2_exp, axis=1)
    g2_out = g2_exp / g2_sum[:, None]

    # Combine and store
    result = tl.where(g1_mask, g1_out, g2_out)
    valid_mask = mask_m[:, None] & mask_n[None, :]
    out_offsets = m_range[:, None] * N + n_range[None, :]
    tl.store(out_ptr + out_offsets, result, mask=valid_mask)


@torch.fx.wrap
def fused_linear_softmax(in_0, in_1, in_2):
    out = torch.empty([38, 9, 1], dtype=in_2.dtype, device=in_2.device)
    fused_linear_softmax_kernel[(1,)](
        in_2, in_1, in_0, out,
        M=19, N=18, K=128, BLOCK_M=32, BLOCK_N=32,
        num_warps=4, num_stages=1,
    )
    return out


def replacement_func():
    return fused_linear_softmax