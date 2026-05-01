"""
Optimization pass for BotNet relative-position attention with H=8, N=8 (S=64).

Same mathematical structure as the H=16/N=16 variant:
  logit[b, s1, s2] = in_0[b, s1, s2]
                   + in_2[b, i, h, j, n]
                   + sum_d(in_1[b, h, i, d] * in_3[d, rel_idx])
  where i = s1 // H, h = s1 % H, j = s2 // N, n = s2 % N,
        rel_idx = (N-1) + j - i   (in [0, 2N-2])
  with H=8, N=8, S=64, D=128, L=15.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match (must mirror model.py exactly)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4):
    matmul = in_1 @ in_3
    tmp_1 = matmul.reshape(-1, 8, 15)
    tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 7], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 9, 15)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 8, None), slice(7, None, None))]
    tmp_7 = tmp_6.reshape(4, 8, 1, 8, 8)
    tmp_8 = tmp_7.expand(-1, -1, 8, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    tmp_11 = tmp_10.reshape(4, 64, 64)
    tmp_12 = in_0 + tmp_11
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15 = matmul_1.transpose(-1, -2)
    return (tmp_15,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ---------------------------------------------------------------------------
# Triton kernel: fuse skew + in_2 + in_0 → logit[B, S, S]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=2),
    ],
    key=['B', 'S'],
)
@triton.jit
def rel_pos_logit_kernel_8(
    in0_ptr,   # [B, S, S]
    in1_ptr,   # [B, H, N, D]
    in2_ptr,   # [B, N, H, N, N]  (indices: b, i, h, j, n)
    in3_ptr,   # [D, L]  L = 2N-1
    out_ptr,   # [B, S, S]
    B, S,
    H: tl.constexpr,   # = 8
    N: tl.constexpr,   # = 8
    D: tl.constexpr,   # = 128
    L: tl.constexpr,   # = 15  (= 2*N-1)
    IS_FP16: tl.constexpr,
):
    """
    Grid: (B, S)
    Each program computes one full row of the output logit matrix.
    """
    b  = tl.program_id(0)
    s1 = tl.program_id(1)

    # Decompose s1 = i * H + h
    i = s1 // H
    h = s1 % H

    # Load in1[b, h, i, 0:D]  – the query vector for this head/row
    d_off = tl.arange(0, D)                          # [D]
    in1_base = b * (H * N * D) + h * (N * D) + i * D
    q_row = tl.load(in1_ptr + in1_base + d_off).to(tl.float32)   # [D]

    # Base offsets for this (b, s1) / (b, i, h)
    in0_row_off = b * (S * S) + s1 * S
    in2_base    = b * (N * H * N * N) + i * (H * N * N) + h * (N * N)
    out_row_off = b * (S * S) + s1 * S

    # Iterate over the N column-groups
    for j in range(N):
        rel_idx = (N - 1) - i + j           # scalar, in [0, 2N-2]

        # Load in3[0:D, rel_idx]
        k_col = tl.load(in3_ptr + d_off * L + rel_idx).to(tl.float32)  # [D]

        # Dot product (shared across all n for this j)
        dot = tl.sum(q_row * k_col, axis=0)   # scalar

        # Load N elements of in0 and in2
        n_off = tl.arange(0, N)                         # [N]
        in0_vals = tl.load(in0_ptr + in0_row_off + j * N + n_off).to(tl.float32)
        in2_vals = tl.load(in2_ptr + in2_base  + j * N + n_off).to(tl.float32)

        result = in0_vals + in2_vals + dot   # [N]

        if IS_FP16:
            tl.store(out_ptr + out_row_off + j * N + n_off, result.to(tl.float16))
        else:
            tl.store(out_ptr + out_row_off + j * N + n_off, result.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def botnet_attn_h8_n8(in_0, in_1, in_2, in_3, in_4):
    B, S, _ = in_0.shape   # B=4, S=64
    H = 8
    N = 8
    D = 128
    L = 15

    IS_FP16 = 1 if in_0.dtype == torch.float16 else 0

    logits = torch.empty_like(in_0)

    grid = (B, S)
    rel_pos_logit_kernel_8[grid](
        in_0, in_1, in_2, in_3, logits,
        B, S,
        H=H, N=N, D=D, L=L,
        IS_FP16=IS_FP16,
    )

    # Softmax over last dimension
    attn = logits.softmax(dim=-1)

    # Attention output + transpose
    out = (attn @ in_4).transpose(-1, -2)

    return (out,)


def replacement_func():
    return botnet_attn_h8_n8