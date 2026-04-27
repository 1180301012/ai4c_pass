import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
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
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: scale(0.0625) + softmax(dim=-1) + matmul + permute(0,2,1)

    in0 : [B, M, K]   (e.g. [B, 8192, 19])
    in1 : [B, K, N]   (e.g. [B, 19,  256])
    out : [B, N, M]   (e.g. [B, 256, 8192])  -- permuted result

    Each program handles a [BLOCK_M, BLOCK_N] tile of the (M, N) output for
    one batch element.  Because K=19 fits in BLOCK_K=32, the entire softmax
    row is loaded once and stays in registers for the matmul.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    k_offs = tl.arange(0, BLOCK_K)                      # [BLOCK_K]

    m_mask = m_offs < M
    n_mask = n_offs < N
    k_mask = k_offs < K

    # ------------------------------------------------------------------
    # 1. Load in0 [BLOCK_M, BLOCK_K]
    #    Use -inf for out-of-range k entries so they are neutralised by
    #    the softmax max-subtract step.
    # ------------------------------------------------------------------
    in0_raw = tl.load(
        in0_ptr
        + pid_b * stride_in0_b
        + m_offs[:, None] * stride_in0_m
        + k_offs[None, :] * stride_in0_k,
        mask=m_mask[:, None] & k_mask[None, :],
        other=float('-inf'),
    )

    # ------------------------------------------------------------------
    # 2. Scale + softmax (computed in float32 for numerical stability)
    # ------------------------------------------------------------------
    in0_f32 = in0_raw.to(tl.float32)
    in0_scaled = 0.0625 * in0_f32                       # [BLOCK_M, BLOCK_K]

    # Subtract row-max for numerical stability
    row_max = tl.max(in0_scaled, axis=1)                # [BLOCK_M]
    in0_centered = in0_scaled - row_max[:, None]
    in0_exp = tl.exp(in0_centered)                      # [BLOCK_M, BLOCK_K]

    # Zero out padded k positions before summing
    in0_exp = tl.where(k_mask[None, :], in0_exp, 0.0)
    row_sum = tl.sum(in0_exp, axis=1)                   # [BLOCK_M]
    # Guard against divide-by-zero for fully out-of-bounds m rows
    row_sum = tl.where(m_mask, row_sum, 1.0)
    in0_softmax = in0_exp / row_sum[:, None]            # [BLOCK_M, BLOCK_K] float32

    # ------------------------------------------------------------------
    # 3. Load in1 [BLOCK_K, BLOCK_N] in native dtype
    # ------------------------------------------------------------------
    in1 = tl.load(
        in1_ptr
        + pid_b * stride_in1_b
        + k_offs[:, None] * stride_in1_k
        + n_offs[None, :] * stride_in1_n,
        mask=k_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    # ------------------------------------------------------------------
    # 4. MatMul  [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
    #    Cast softmax to in1's native dtype for tl.dot
    # ------------------------------------------------------------------
    in0_for_dot = in0_softmax.to(in1.dtype)
    result = tl.dot(in0_for_dot, in1, allow_tf32=True)  # float32 accumulation

    # Cast back to native dtype
    result_native = result.to(in1.dtype)

    # ------------------------------------------------------------------
    # 5. Store with permute(0,2,1):  out[b, n, m] = result[m, n]
    #    Transpose the [BLOCK_M, BLOCK_N] tile to [BLOCK_N, BLOCK_M] and
    #    store at the corresponding [n, m] addresses.
    # ------------------------------------------------------------------
    result_T = tl.trans(result_native)   # [BLOCK_N, BLOCK_M]

    tl.store(
        out_ptr
        + pid_b * stride_out_b
        + n_offs[:, None] * stride_out_n
        + m_offs[None, :] * stride_out_m,
        result_T,
        mask=n_mask[:, None] & m_mask[None, :],
    )


@torch.fx.wrap
def fused_scale_softmax_matmul_permute(in_0, in_1):
    B, M, K = in_0.shape
    _, _, N = in_1.shape

    # Output shape after permute(0, 2, 1): [B, N, M]
    out = torch.empty((B, N, M), dtype=in_0.dtype, device=in_0.device)

    def grid(meta):
        return (
            B,
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    fused_scale_softmax_matmul_permute_kernel[grid](
        in_0, in_1, out,
        B, M, K, N,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
    )

    return out


def replacement_func():
    return fused_scale_softmax_matmul_permute