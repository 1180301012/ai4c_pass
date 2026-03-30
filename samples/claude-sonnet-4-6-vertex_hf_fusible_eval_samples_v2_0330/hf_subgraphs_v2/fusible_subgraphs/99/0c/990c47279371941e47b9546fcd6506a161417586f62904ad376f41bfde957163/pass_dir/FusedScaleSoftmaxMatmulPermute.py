import torch
import triton
import triton.language as tl


# ─── Pattern: match the full fused subgraph ──────────────────────────────────
# scale * in_0 -> softmax(dim=-1) -> matmul(in_1) -> permute(0,2,1)
def pattern(in_0, in_1):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ─── Triton kernel: fused scale+softmax+matmul+permute ───────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
    ],
    key=['B', 'M', 'K', 'N'],
)
@triton.jit
def fused_scale_softmax_matmul_permute_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, M, K, N,
    stride_in0_b, stride_in0_m, stride_in0_k,
    stride_in1_b, stride_in1_k, stride_in1_n,
    stride_out_b, stride_out_n, stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_PADDED: tl.constexpr,
):
    """
    in0:  [B, M, K]  – raw similarity map
    in1:  [B, K, N]  – value tensor
    out:  [B, N, M]  – permuted attention output

    Computes: out[b, n, m] = sum_k softmax(0.0625 * in0[b, m, :])[k] * in1[b, k, n]
    Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    m_offs  = m_start + tl.arange(0, BLOCK_M)
    k_offs  = tl.arange(0, K_PADDED)
    n_start = pid_n * BLOCK_N
    n_offs  = n_start + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    k_mask = k_offs < K
    n_mask = n_offs < N

    # Load in_0[b, m, k]  shape [BLOCK_M, K_PADDED]
    in0_offs = (pid_b * stride_in0_b
                + m_offs[:, None] * stride_in0_m
                + k_offs[None, :] * stride_in0_k)
    in0 = tl.load(in0_ptr + in0_offs,
                  mask=m_mask[:, None] & k_mask[None, :],
                  other=float('-inf'))
    in0 = in0.to(tl.float32)

    # Scale
    in0 = in0 * 0.0625

    # Numerically-stable softmax over K (last dim)
    in0_max = tl.max(in0, axis=1)
    in0_exp = tl.exp(in0 - in0_max[:, None])
    in0_exp = tl.where(k_mask[None, :], in0_exp, 0.0)
    in0_sum = tl.sum(in0_exp, axis=1)
    softmax  = in0_exp / in0_sum[:, None]          # [BLOCK_M, K_PADDED]

    # Load in_1[b, k, n]  shape [K_PADDED, BLOCK_N]
    in1_offs = (pid_b * stride_in1_b
                + k_offs[:, None] * stride_in1_k
                + n_offs[None, :] * stride_in1_n)
    in1 = tl.load(in1_ptr + in1_offs,
                  mask=k_mask[:, None] & n_mask[None, :],
                  other=0.0)
    in1 = in1.to(tl.float32)

    # Matmul  [BLOCK_M, K_PADDED] @ [K_PADDED, BLOCK_N] → [BLOCK_M, BLOCK_N]
    result = tl.dot(softmax, in1, allow_tf32=True)

    # Store transposed: out[b, n, m] = result[m, n]
    result_t = tl.trans(result)                    # [BLOCK_N, BLOCK_M]
    out_offs = (pid_b * stride_out_b
                + n_offs[:, None] * stride_out_n
                + m_offs[None, :] * stride_out_m)
    tl.store(out_ptr + out_offs, result_t,
             mask=n_mask[:, None] & m_mask[None, :])


@torch.fx.wrap
def fused_scale_softmax_matmul_permute(in_0, in_1):
    B, M, K = in_0.shape
    _,  _, N = in_1.shape

    # K_PADDED: next power-of-2 >= max(16, K)  → 32 for K=19
    K_PADDED = max(16, 1 << (max(K, 1) - 1).bit_length())

    out = torch.empty((B, N, M), dtype=in_0.dtype, device=in_0.device)

    def grid(meta):
        return (B,
                triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(N, meta['BLOCK_N']))

    fused_scale_softmax_matmul_permute_kernel[grid](
        in_0, in_1, out,
        B, M, K, N,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        K_PADDED=K_PADDED,
    )

    return out


def replacement_func():
    return fused_scale_softmax_matmul_permute