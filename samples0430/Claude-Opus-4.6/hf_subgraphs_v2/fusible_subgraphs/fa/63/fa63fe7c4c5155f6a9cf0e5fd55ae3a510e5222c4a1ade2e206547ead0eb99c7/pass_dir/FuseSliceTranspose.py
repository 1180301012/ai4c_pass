import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    result = in_1 @ in_0
    return result


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(1)
    pid_m = tl.program_id(0)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Accumulate over K in chunks of BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        a = tl.load(a_ptr + batch * stride_a_batch + m_offs[:, None] * stride_a_m + k_offs[None, :] * stride_a_k,
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptr + batch * stride_b_batch + k_offs[:, None] * stride_b_k + n_offs[None, :] * stride_b_n,
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(a, b, acc)

    # Store result
    c_ptrs = c_ptr + batch * stride_c_batch + m_offs[:, None] * stride_c_m + n_offs[None, :] * stride_c_n
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@torch.fx.wrap
def triton_batched_matmul(in_0, in_1):
    # in_1 @ in_0: [1,H,M,K] @ [1,H,K,N] = [1,H,M,N]
    out = torch.empty_like(in_1)
    M = in_1.shape[2]
    K = in_1.shape[3]
    N = in_0.shape[3]
    batch = in_1.shape[1]

    grid = ((M + 31) // 32, batch)

    batched_matmul_kernel[grid](
        in_1, in_0, out,
        M, K, N,
        in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(1), in_0.stride(2), in_0.stride(3),
        out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=32, BLOCK_K=32, BLOCK_N=64,
        num_warps=4, num_stages=2,
    )

    return out


def replacement_func():
    return triton_batched_matmul