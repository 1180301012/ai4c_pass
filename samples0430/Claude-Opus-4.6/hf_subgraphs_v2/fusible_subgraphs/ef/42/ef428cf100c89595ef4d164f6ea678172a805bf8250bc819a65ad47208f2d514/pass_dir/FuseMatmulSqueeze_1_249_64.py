import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def vecmat_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Tensor-core vector-matrix multiply for exact correctness
    # Pad M=1 to BLOCK_M=16 to use tl.dot (tensor cores)
    n_offsets = tl.arange(0, N)
    m_offsets = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A: [BLOCK_M, BLOCK_K], only row 0 has real data
        a_mask = (m_offsets[:, None] == 0) & k_mask[None, :]
        a = tl.load(a_ptr + k_offsets[None, :], mask=a_mask, other=0.0)

        # Load B: [BLOCK_K, N]
        b_offsets = k_offsets[:, None] * N + n_offsets[None, :]
        b = tl.load(b_ptr + b_offsets, mask=k_mask[:, None], other=0.0)

        # Tensor core matmul: [BLOCK_M, BLOCK_K] @ [BLOCK_K, N]
        acc += tl.dot(a, b)

    # Only row 0 of A is non-zero so only row 0 of acc is non-zero
    # Sum all rows to extract the result
    result = tl.sum(acc, axis=0)
    tl.store(out_ptr + n_offsets, result.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_matmul_squeeze(in_0, in_1):
    K = in_0.shape[-1]
    N = in_1.shape[-1]
    out = torch.empty((1, N), dtype=in_0.dtype, device=in_0.device)

    vecmat_kernel[(1,)](
        in_0,
        in_1,
        out,
        K=K,
        N=N,
        BLOCK_K=32,
        BLOCK_M=16,
        num_warps=4,
        num_stages=2,
    )

    return out


def replacement_func():
    return fused_matmul_squeeze