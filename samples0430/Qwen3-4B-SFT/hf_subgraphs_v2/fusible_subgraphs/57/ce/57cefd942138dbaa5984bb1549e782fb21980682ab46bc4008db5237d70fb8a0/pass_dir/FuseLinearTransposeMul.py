import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_2, in_1, in_0, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_2, in_1, in_0, in_3):
    return (in_2, in_1, in_0, in_3)


# ── Triton kernel ─────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['B', 'K', 'N'],
)
@triton.jit
def _fused_linear_transpose_mul_kernel(
    in2_ptr, in1_ptr, bias_ptr, in3_ptr, out_ptr,
    B, K, N,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused: out = in3 * (in2 @ in1.T + bias).transpose(-1, -2)

    Grid: (ceil(B*K / BLOCK_M),  ceil(N / BLOCK_N))

    Each program covers:
      - BLOCK_M "tokens" from (b, k) space  →  b = token // K,  k = token % K
      - BLOCK_N output-feature positions    →  i = n_offset + [0..BLOCK_N)
      - accumulates over K in BLOCK_K steps

    After the matmul+add-bias, each output element is multiplied by in3[b, i, j]
    and written contiguously.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ── Offset arrays ────────────────────────────────────────────────────────
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # shape [BLOCK_M]
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # shape [BLOCK_N]

    m_mask = m_offs < B * K
    n_mask = n_offs < N

    # Decode (b, k) from the flattened m index
    b_idx = m_offs // K    # [BLOCK_M]
    k_idx = m_offs % K     # [BLOCK_M]

    # ── Accumulator ─────────────────────────────────────────────────────────
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── Main reduction loop over K ───────────────────────────────────────────
    for k_start in range(0, K, BLOCK_K):
        k_keep = k_start + tl.arange(0, BLOCK_K)       # [BLOCK_K]
        k_mask = k_keep < K

        # in2[b, k, j]  shape [B, K, N]
        # Stride: [K*N, N, 1]
        in2_idx = (b_idx[:, None] * K * N
                   + k_keep[None, :] * N
                   + n_offs[None, :])                   # [BLOCK_M, BLOCK_K], [BLOCK_K, BLOCK_N]
        in2_tile = tl.load(in2_ptr + in2_idx,
                           mask=m_mask[:, None] & k_mask[None, :] & n_mask[None, :],
                           other=0.0).to(tl.float16)

        # in1[i, k]  shape [N, K]
        # We need shape [BLOCK_K, BLOCK_N]  (i outer, k inner → transpose w.r.t. storage)
        in1_idx = (k_keep[:, None] * N
                   + n_offs[None, :])                   # [BLOCK_K, BLOCK_N]
        in1_tile = tl.load(in1_ptr + in1_idx,
                           mask=k_mask[:, None] & n_mask[None, :],
                           other=0.0).to(tl.float16)

        # acc += in2_tile @ in1_tile.T   →  [BLOCK_M, BLOCK_N]
        acc += tl.dot(in2_tile, tl.trans(in1_tile))

    # ── Add bias ─────────────────────────────────────────────────────────────
    bias_vals = tl.load(
        bias_ptr + n_offs,
        mask=n_mask, other=0.0
    ).to(tl.float32)
    acc += bias_vals[None, :]

    # ── Multiply by in3[b, i, j] and contiguous out[b, i, j] ─────────────────
    # in3 is [B, N, K]  →  stride [N*K, K, 1]
    # out  is [B, N, K]  contiguous
    i_idx = n_offs   # n_offs == i because we tile over N = output features
    in3_idx = (b_idx[:, None] * N * K
               + i_idx[None, :] * K
               + k_keep[:, None])                       # [BLOCK_M, BLOCK_K]
    in3_vals = tl.load(
        in3_ptr + in3_idx,
        mask=m_mask[:, None] & k_mask[:, None],
        other=0.0
    ).to(tl.float32)

    result = acc[:, None] * in3_vals                    # [BLOCK_M, BLOCK_K, 1] → broadcast

    # ── Store ────────────────────────────────────────────────────────────────
    out_idx = (b_idx[:, None] * N * K
               + i_idx[None, :] * K
               + k_keep[:, None])                       # [BLOCK_M, BLOCK_K]
    out_mask = m_mask[:, None] & k_mask[:, None] & n_mask[None, :]

    if IS_FP16:
        tl.store(out_ptr + out_idx, result.to(tl.float16),  mask=out_mask)
    elif IS_BF16:
        tl.store(out_ptr + out_idx, result.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptr + out_idx, result.to(tl.float32),  mask=out_mask)


# ── Wrapper ───────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _fused_linear_transpose_mul(in_2, in_1, in_0, in_3):
    """
    Replacement for:  F.linear(in_2, in_1, in_0).transpose(-1,-2) * in_3

    in_2 : [B, K, N]   (BN = 196)
    in_1 : [N, K]       (weight)
    in_0 : [N]          (bias)
    in_3 : [B, N, K]
    out  : [B, N, K]
    """
    B = in_2.shape[0]
    K = in_2.shape[1]
    N = in_2.shape[2]

    out = torch.empty_like(in_3)

    IS_FP16 = (in_2.dtype == torch.float16)
    IS_BF16 = (in_2.dtype == torch.bfloat16)

    grid = lambda meta: (
        triton.cdiv(B * K, meta['BLOCK_M']),
        triton.cdiv(N,     meta['BLOCK_N']),
    )

    _fused_linear_transpose_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, K, N,
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
    )
    return out


# ── Replacement function ──────────────────────────────────────────────────────
def replacement_func():
    return _fused_linear_transpose_mul