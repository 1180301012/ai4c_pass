import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matmul(in_0, in_1)  →  permute(0, 2, 1)
#
#   in_0 : [B, M, K]  (e.g. softmax output [B, 8192, 19])
#   in_1 : [B, K, N]  (e.g. value matrix   [B, 19,  256])
#   out  : [B, N, M]  (permuted result)
#
# The Triton kernel writes the batched-matmul result directly in the
# transposed [B, N, M] layout, saving a subsequent permute memory pass.
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_3  = matmul.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: batched matmul with transposed output write
#   in0_ptr : [B, M, K]
#   in1_ptr : [B, K, N]
#   out_ptr : [B, N, M]  ← result written directly in [B, N, M] layout
#
# Each CTA computes a [BLOCK_M, BLOCK_N] tile of the matmul result and
# stores it at the transposed addresses out[b, n, m].
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8,  num_stages=4),
    ],
    key=['B', 'M', 'K', 'N'],
)
@triton.jit
def _matmul_transposed_kernel(
    in0_ptr,            # [B, M, K]
    in1_ptr,            # [B, K, N]
    out_ptr,            # [B, N, M]
    B, M, K, N,
    stride_in0_b, stride_in0_m, stride_in0_k,
    stride_in1_b, stride_in1_k, stride_in1_n,
    stride_out_b, stride_out_n, stride_out_m,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_range = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    m_mask  = m_range < M
    n_mask  = n_range < N

    # acc[BLOCK_M, BLOCK_N] in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    in0_base = pid_b * stride_in0_b
    in1_base = pid_b * stride_in1_b

    # Tile over K (K=19, fits in one BLOCK_K=32 tile; loop handles general case)
    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask  = k_range < K

        # Load A tile [BLOCK_M, BLOCK_K] — coalesced in K direction
        a_off  = in0_base + m_range[:, None] * stride_in0_m + k_range[None, :] * stride_in0_k
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(in0_ptr + a_off, mask=a_mask, other=0.0).to(tl.float32)

        # Load B tile [BLOCK_K, BLOCK_N] — coalesced in N direction
        b_off  = in1_base + k_range[:, None] * stride_in1_k + n_range[None, :] * stride_in1_n
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(in1_ptr + b_off, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
        acc = tl.dot(a, b, acc)

    # Cast to output dtype and store in transposed [B, N, M] layout
    if IS_FP16:
        acc = acc.to(tl.float16)
    elif IS_BF16:
        acc = acc.to(tl.bfloat16)

    out_base    = pid_b * stride_out_b
    # out[b, n, m]: offsets[m_i, n_j] = n_j*stride_n + m_i*stride_m
    out_offsets = (out_base
                   + n_range[None, :] * stride_out_n   # [1, BLOCK_N]
                   + m_range[:, None] * stride_out_m)  # [BLOCK_M, 1]
    out_mask    = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper  (@torch.fx.wrap required for FX graph substitution)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def matmul_transposed(in_0, in_1):
    B, M, K = in_0.shape
    _, _,  N = in_1.shape

    # Output shape: [B, N, M]  (permuted from the matmul result [B, M, N])
    out = torch.empty((B, N, M), dtype=in_0.dtype, device=in_0.device)

    IS_FP16 = (in_0.dtype == torch.float16)
    IS_BF16 = (in_0.dtype == torch.bfloat16)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
            B,
        )

    _matmul_transposed_kernel[grid](
        in_0, in_1, out,
        B, M, K, N,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
    )

    return out


# ---------------------------------------------------------------------------
# Required by the AI4C framework
# ---------------------------------------------------------------------------

def replacement_func():
    return matmul_transposed