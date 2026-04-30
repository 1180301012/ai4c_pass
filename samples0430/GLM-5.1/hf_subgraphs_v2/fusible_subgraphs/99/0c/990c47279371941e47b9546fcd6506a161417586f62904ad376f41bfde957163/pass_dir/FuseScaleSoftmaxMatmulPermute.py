import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return (tmp_3,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['N', 'K'],
)
@triton.jit
def fused_scale_softmax_matmul_permute_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, N, M, K,
    scale,
    stride_0_b, stride_0_n, stride_0_m,
    stride_1_b, stride_1_m, stride_1_k,
    stride_out_b, stride_out_k, stride_out_n,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    total_blocks_per_batch = num_n_blocks * num_k_blocks

    b = pid // total_blocks_per_batch
    nk_id = pid % total_blocks_per_batch
    n_block = nk_id % num_n_blocks
    k_block = nk_id // num_n_blocks

    n_start = n_block * BLOCK_N
    k_start = k_block * BLOCK_K

    n_offsets = n_start + tl.arange(0, BLOCK_N)
    k_offsets = k_start + tl.arange(0, BLOCK_K)

    n_mask = n_offsets < N
    k_mask = k_offsets < K

    # Online softmax + matmul in float32 for numerical stability
    m_i = tl.full([BLOCK_N], float('-inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)               # running sum of exponentials
    o_ik = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)     # running matmul accumulator

    for j in range(M):
        # Load in_0[b, n_offsets, j] -> [BLOCK_N]
        ptrs_0 = in_0_ptr + b * stride_0_b + n_offsets * stride_0_n + j * stride_0_m
        x_ij = tl.load(ptrs_0, mask=n_mask, other=0.0).to(tl.float32)
        x_ij = x_ij * scale

        # Load in_1[b, j, k_offsets] -> [BLOCK_K]
        ptrs_1 = in_1_ptr + b * stride_1_b + j * stride_1_m + k_offsets * stride_1_k
        v_jk = tl.load(ptrs_1, mask=k_mask, other=0.0).to(tl.float32)

        # Online softmax update: compute new max and correction factor
        m_new = tl.maximum(m_i, x_ij)
        alpha = tl.exp(m_i - m_new)

        # Update running sum
        l_i = l_i * alpha + tl.exp(x_ij - m_new)

        # Update matmul accumulator: rescale previous accumulation and add new contribution
        o_ik = o_ik * alpha[:, None] + tl.exp(x_ij - m_new)[:, None] * v_jk[None, :]

        m_i = m_new

    # Normalize by softmax denominator
    o_ik = o_ik / l_i[:, None]

    # Store output[b, k, n] = o_ik[n_local, k_local]
    # output layout: [B, K, N] with n being the contiguous dimension
    for ki in range(BLOCK_K):
        k_idx = k_start + ki
        if k_idx < K:
            ptrs_out = out_ptr + b * stride_out_b + k_idx * stride_out_k + n_offsets * stride_out_n
            tl.store(ptrs_out, o_ik[:, ki], mask=n_mask)


@torch.fx.wrap
def fused_scale_softmax_matmul_permute(in_0, in_1):
    B = in_0.shape[0]
    N = in_0.shape[1]
    M = in_0.shape[2]
    K = in_1.shape[2]

    # Output shape: [B, K, N] (transposed from matmul result [B, N, K])
    out = torch.empty((B, K, N), dtype=in_0.dtype, device=in_0.device)

    scale = 0.0625

    def grid_fn(meta):
        num_n_blocks = triton.cdiv(N, meta['BLOCK_N'])
        num_k_blocks = triton.cdiv(K, meta['BLOCK_K'])
        return (B * num_n_blocks * num_k_blocks,)

    fused_scale_softmax_matmul_permute_kernel[grid_fn](
        in_0_ptr=in_0, in_1_ptr=in_1, out_ptr=out,
        B=B, N=N, M=M, K=K,
        scale=scale,
        stride_0_b=in_0.stride(0), stride_0_n=in_0.stride(1), stride_0_m=in_0.stride(2),
        stride_1_b=in_1.stride(0), stride_1_m=in_1.stride(1), stride_1_k=in_1.stride(2),
        stride_out_b=out.stride(0), stride_out_k=out.stride(1), stride_out_n=out.stride(2),
    )

    return (out,)


def replacement_func():
    return fused_scale_softmax_matmul_permute