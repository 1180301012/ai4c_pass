import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # ── Small-block: many programs → good SM coverage for B=1 ──
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=4),
        # ── Medium-block: FP16/BF16 tensor cores (BLOCK_K=64 feeds MMA pipeline) ──
        # BLOCK_N=128 divides N=768 exactly; BLOCK_K=64 gives 4 K-iters (6.25% waste)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        # ── Large-block FP16/BF16: BLOCK_N=256 divides N=768 exactly ──
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        # ── Large-block FP32: BLOCK_K=32 is optimal with TF32 tensor cores ──
        # For FP32, 96KB shared: 3×(128×32+128×32)×4 = 96KB ← exactly at A30 limit
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32},  num_stages=2, num_warps=8),
    ],
    key=['B', 'M', 'N', 'K', 'DTYPE_SIZE'],
)
@triton.jit
def _fused_linear_transpose_mul_kernel(
    in2_ptr,   # [B, N, K]  — input to linear
    in1_ptr,   # [M, K]     — weight matrix
    in0_ptr,   # [M]        — bias
    in3_ptr,   # [B, M, N]  — scale tensor
    out_ptr,   # [B, M, N]  — output
    B, M, N, K,
    DTYPE_SIZE,   # element size in bytes — used ONLY as autotune key discriminator
    s_in2_b, s_in2_n, s_in2_k,
    s_in1_m, s_in1_k,
    s_in3_b, s_in3_m, s_in3_n,
    s_out_b,  s_out_m,  s_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel computing:
      out[b, m, n] = in3[b, m, n] * (sum_k(in1[m, k] * in2[b, n, k]) + in0[m])

    Corresponds to:  out = in3 * F.linear(in2, in1, in0).transpose(-1, -2)

    Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    Each program owns a [BLOCK_M, BLOCK_N] output tile.

    KEY: loads keep the native dtype so tl.dot can dispatch to the correct
    tensor-core mode (FP16/BF16 → 8x throughput vs FP32 on A30).
    Accumulation is always in FP32 via out_dtype=tl.float32.
    """
    bid   = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # FP32 accumulator — tl.dot uses tensor cores for fp16/bf16 and
    # accumulates into fp32 via out_dtype=tl.float32.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-reduction loop — iterate over the shared K dimension
    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # Load in1[m, k]: [BLOCK_M, BLOCK_K] in native dtype
        in1_ptrs = in1_ptr + m_off[:, None] * s_in1_m + k_off[None, :] * s_in1_k
        in1_mask = (m_off[:, None] < M) & (k_off[None, :] < K)
        in1 = tl.load(in1_ptrs, mask=in1_mask, other=0.0)

        # Load in2[b, n, k]: [BLOCK_N, BLOCK_K] in native dtype
        in2_ptrs = (in2_ptr
                    + bid * s_in2_b
                    + n_off[:, None] * s_in2_n
                    + k_off[None, :] * s_in2_k)
        in2_mask = (n_off[:, None] < N) & (k_off[None, :] < K)
        in2 = tl.load(in2_ptrs, mask=in2_mask, other=0.0)

        # acc[m, n] += sum_k(in1[m, k] * in2[n, k])
        #            = in1 @ in2.T   [BLOCK_M,K] @ [K,BLOCK_N] → [BLOCK_M,BLOCK_N]
        # out_dtype=tl.float32 ensures FP32 accumulation for all input dtypes;
        # tl.dot selects FP16/BF16 tensor cores when inputs are half-precision.
        acc += tl.dot(in1, tl.trans(in2), out_dtype=tl.float32, allow_tf32=True)

    # Add bias in0[m] (broadcast over N)
    bias = tl.load(in0_ptr + m_off, mask=m_off < M, other=0.0).to(tl.float32)
    acc += bias[:, None]

    # Load scale in3[b, m, n]: [BLOCK_M, BLOCK_N] in native dtype
    in3_ptrs = (in3_ptr
                + bid * s_in3_b
                + m_off[:, None] * s_in3_m
                + n_off[None, :] * s_in3_n)
    in3_mask = (m_off[:, None] < M) & (n_off[None, :] < N)
    in3 = tl.load(in3_ptrs, mask=in3_mask, other=0.0)

    # Element-wise multiply — cast acc back to input dtype before multiply
    result = in3 * acc.to(in3.dtype)

    out_ptrs = (out_ptr
                + bid * s_out_b
                + m_off[:, None] * s_out_m
                + n_off[None, :] * s_out_n)
    tl.store(out_ptrs, result, mask=in3_mask)


@torch.fx.wrap
def fused_linear_transpose_mul(in_0, in_1, in_2, in_3):
    """
    in_0 : [M]       bias
    in_1 : [M, K]    weight
    in_2 : [B, N, K] input to linear
    in_3 : [B, M, N] scale tensor
    Returns [B, M, N]
    """
    B = in_2.shape[0]
    N = in_2.shape[1]
    K = in_2.shape[2]
    M = in_1.shape[0]
    # element size: 4 for fp32, 2 for fp16/bf16 — used to give each dtype
    # its own autotune cache entry so fp32 and fp16/bf16 don't share configs.
    dtype_size = in_2.element_size()

    out = torch.empty((B, M, N), dtype=in_3.dtype, device=in_3.device)

    grid = lambda meta: (
        B,
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    _fused_linear_transpose_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, M, N, K,
        dtype_size,
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        in_1.stride(0), in_1.stride(1),
        in_3.stride(0), in_3.stride(1), in_3.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    """Matches: linear(in_2, in_1, in_0) -> transpose(-1,-2) -> mul(in_3, ...)"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_linear_transpose_mul