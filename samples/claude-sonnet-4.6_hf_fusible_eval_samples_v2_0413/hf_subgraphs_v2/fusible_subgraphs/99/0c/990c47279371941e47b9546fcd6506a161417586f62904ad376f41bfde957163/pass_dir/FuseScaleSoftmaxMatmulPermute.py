import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused Triton kernel: scale + softmax + matmul + implicit-permute
#
# Computation:
#   attn[b,m,k] = softmax_k(0.0625 * in0[b,m,k])
#   out[b,m,n]  = sum_k(attn[b,m,k] * in1[b,k,n])
# The wrapper calls .permute(0,2,1) on the [B,M,N] result to get [B,N,M].
#
# in0 : [B, M, K]   M=8192, K=19
# in1 : [B, K, N]   N=256
# out : [B, M, N]   (then viewed as [B, N, M] via permute)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128},num_warps=8, num_stages=2),
    ],
    key=['B', 'M', 'N'],
)
@triton.jit
def _fused_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, M, K, N,
    s_b0, s_m0, s_k0,   # strides for in0
    s_b1, s_k1, s_n1,   # strides for in1
    s_bo, s_mo, s_no,   # strides for out
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,   # fixed = 32 (next power-of-2 >= 19, also >= 16 for tl.dot)
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_off = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    k_off = tl.arange(0, BLOCK_K)              # [BLOCK_K]  (padded to 32)
    n_off = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # ------------------------------------------------------------------
    # 1. Load in0[b, m:m+BM, 0:K] and scale + softmax in fp32
    # ------------------------------------------------------------------
    in0_mask = (m_off[:, None] < M) & (k_off[None, :] < K)
    in0_raw = tl.load(
        in0_ptr + pid_b * s_b0 + m_off[:, None] * s_m0 + k_off[None, :] * s_k0,
        mask=in0_mask, other=0.0,
    )                                                # native dtype, [BM, BK]

    # Promote to fp32 for numerically stable softmax
    in0_f32 = in0_raw.to(tl.float32)

    # Scale
    in0_f32 = in0_f32 * 0.0625

    # Mask padding positions before softmax (k >= K → -inf so exp → 0)
    in0_f32 = tl.where(k_off[None, :] < K, in0_f32, float('-inf'))

    # Numerically-stable softmax along K axis
    row_max = tl.max(in0_f32, axis=1)[:, None]     # [BM, 1]
    in0_f32 = in0_f32 - row_max
    exp_vals = tl.exp(in0_f32)
    row_sum = tl.sum(exp_vals, axis=1)[:, None]    # [BM, 1]
    attn_f32 = exp_vals / row_sum                  # [BM, BK], fp32

    # Cast attention weights back to native dtype for tensor-core matmul
    attn = attn_f32.to(in0_raw.dtype)             # [BM, BK]

    # ------------------------------------------------------------------
    # 2. Load in1[b, 0:K, n:n+BN]
    # ------------------------------------------------------------------
    in1_mask = (k_off[:, None] < K) & (n_off[None, :] < N)
    in1 = tl.load(
        in1_ptr + pid_b * s_b1 + k_off[:, None] * s_k1 + n_off[None, :] * s_n1,
        mask=in1_mask, other=0.0,
    )                                              # native dtype, [BK, BN]

    # ------------------------------------------------------------------
    # 3. Batched matmul via tl.dot (uses tensor cores for fp16/bf16)
    #    result = attn [BM, BK] @ in1 [BK, BN]  → [BM, BN]  (fp32 accum)
    # ------------------------------------------------------------------
    result = tl.dot(attn, in1, allow_tf32=True)   # [BM, BN], fp32

    # ------------------------------------------------------------------
    # 4. Store back in native dtype to out[b, m:m+BM, n:n+BN]
    # ------------------------------------------------------------------
    out_mask = (m_off[:, None] < M) & (n_off[None, :] < N)
    tl.store(
        out_ptr + pid_b * s_bo + m_off[:, None] * s_mo + n_off[None, :] * s_no,
        result.to(in0_raw.dtype),
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return (tmp_3,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_scale_softmax_matmul_permute(in_0, in_1):
    B = in_0.shape[0]
    M = in_0.shape[1]
    K = in_0.shape[2]
    N = in_1.shape[2]

    # Allocate contiguous output [B, M, N]
    out = torch.empty((B, M, N), dtype=in_0.dtype, device=in_0.device)

    # BLOCK_K = 32: next power-of-2 ≥ K=19, satisfies tl.dot inner-dim ≥ 16
    BLOCK_K = 32

    grid = lambda meta: (B,
                         triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))

    _fused_kernel[grid](
        in_0, in_1, out,
        B, M, K, N,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        BLOCK_K=BLOCK_K,
    )

    # permute(0, 2, 1): [B, M, N] → [B, N, M] as a zero-copy view
    return out.permute(0, 2, 1)


def replacement_func():
    return fused_scale_softmax_matmul_permute