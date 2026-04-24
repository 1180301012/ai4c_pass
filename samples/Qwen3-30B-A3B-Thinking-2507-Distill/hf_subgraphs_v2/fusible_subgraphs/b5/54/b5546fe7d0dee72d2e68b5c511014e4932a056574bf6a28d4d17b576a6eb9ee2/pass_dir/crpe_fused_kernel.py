"""
Shared Triton kernel for fused CRPE (Cross-stage Token Reduction with Positional Embedding) computation.

The computation fuses:
  cat([in2, in3, conv], dim=1) -> reshape(1,8,H,W) -> transpose(-1,-2)
  -> multiply(in6) -> pad((0,0,1,0,0,0))
  -> scale * in4 + padded
  -> transpose(1,2) -> reshape(1, N+1, 8*D)

Inputs:
  in2:   [1, C2, H1, W1]   -- first branch feature map
  in3:   [1, C3, H2, W2]   -- second branch feature map
  conv:  [1, C4, H3, W3]   -- conv2d output (third branch)
  in4:   [1, 8, N+1, D]    -- scale tensor (includes class token)
  in6:   [1, 8, N,   D]    -- gate tensor
  out:   [1, N+1, 8*D]     -- output

Where:
  S0 = C2 * H1 * W1  (branch-1 valid range)
  S1 = C3 * H2 * W2  (branch-2 valid range)
  N  = S0 + S1 + C4*H3*W3  (total spatial tokens before padding)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1}, num_warps=1),
        triton.Config({'BLOCK_N': 2}, num_warps=1),
        triton.Config({'BLOCK_N': 4}, num_warps=2),
        triton.Config({'BLOCK_N': 8}, num_warps=4),
        triton.Config({'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_N': 32}, num_warps=8),
        triton.Config({'BLOCK_N': 64}, num_warps=8),
    ],
    key=['N', 'D', 'S0'],
)
@triton.jit
def _crpe_fused_kernel(
    in2_ptr, in3_ptr, conv_ptr,
    in4_ptr, in6_ptr,
    out_ptr,
    N,       # total spatial tokens (before pad) — runtime int
    D,       # head_dim — constexpr
    S0,      # C2 * H1 * W1  — runtime int
    S1,      # C3 * H2 * W2  — runtime int
    scale,   # scalar multiplier — float
    BLOCK_N: tl.constexpr,   # tile size over j dimension (power-of-2)
    BLOCK_D: tl.constexpr,   # must be >= D, power-of-2 (passed by wrapper)
):
    """
    Grid: (ceil(N / BLOCK_N),)
    Each program handles BLOCK_N consecutive spatial token positions j = 0..N-1.

    For each position j:
      i = j + 1  (row in the padded [1,8,N+1,D] tensor, i.e. spatial row i)
      branch_idx = h * D + d  (selects which branch to read from)
      b = in2 if branch_idx < S0
          = in3 if S0 <= branch_idx < S0+S1
          = conv otherwise
      gate = in6[0,h,j,d] * b           (zero when j >= N via masked load)
      out[0, i, h*D+d] = scale * in4[0,h,i,d] + gate
    """
    pid = tl.program_id(0)
    j_base = pid * BLOCK_N

    # ── index vectors ────────────────────────────────────────────────────
    j_off = tl.arange(0, BLOCK_N)   # [BLOCK_N]
    d_off = tl.arange(0, BLOCK_D)   # [BLOCK_D]
    j = j_base + j_off              # [BLOCK_N]  spatial-position offsets
    d_mask = d_off < D              # [BLOCK_D]  valid element mask

    # ── head & branch indices (broadcast across j dimension) ─────────────
    h = d_off // D                  # [BLOCK_D]  head index (0..7)
    branch_idx = h * D + d_off      # [BLOCK_D]  linear branch index

    in_b0 = branch_idx < S0                      # [BLOCK_D]
    in_b1 = (branch_idx >= S0) & (branch_idx < S0 + S1)  # [BLOCK_D]
    # branch 2: no upper bound needed — grid ensures j < N covers all valid tokens

    # ── load from branches (masked) ──────────────────────────────────────
    # in2 / in3 / conv are each stored as a flat [S_k] contiguous tensor
    # (the batch=1 is folded away; the spatial dimensions are linearised).
    b0 = tl.load(
        in2_ptr + branch_idx[:, None] + d_off[None, :],
        mask=in_b0[:, None] & d_mask[None, :],
        other=0.0,
    )  # [BLOCK_N, BLOCK_D]

    b1 = tl.load(
        in3_ptr + (branch_idx - S0)[:, None] + d_off[None, :],
        mask=in_b1[:, None] & d_mask[None, :],
        other=0.0,
    )  # [BLOCK_N, BLOCK_D]

    b2 = tl.load(
        conv_ptr + (branch_idx - S0 - S1)[:, None] + d_off[None, :],
        mask=(branch_idx >= S0 + S1)[:, None] & d_mask[None, :],
        other=0.0,
    )  # [BLOCK_N, BLOCK_D]

    b = tl.where(in_b0, b0, tl.where(in_b1, b1, b2))  # [BLOCK_N, BLOCK_D]

    # ── load gate in6[0, h, j, d] ────────────────────────────────────────
    # in6 shape: [1, 8, N, D]
    in6_base = h[None, :] * (N * D) + j[:, None] * D + d_off[None, :]  # [BLOCK_N, BLOCK_D]
    valid = (j < N)[:, None] & d_mask[None, :]
    in6 = tl.load(in6_ptr + in6_base, mask=valid, other=0.0)

    # ── gated value: in6 * b  (b=0 where in6=0 via d_mask) ───────────────
    gate = in6 * b  # [BLOCK_N, BLOCK_D]

    # ── load in4[0, h, i, d] where i = j+1  ──────────────────────────────
    # in4 shape: [1, 8, N+1, D]
    i = j + 1
    x4_base = h[None, :] * ((N + 1) * D) + i[:, None] * D + d_off[None, :]  # [BLOCK_N, BLOCK_D]
    x4 = tl.load(in4_ptr + x4_base, mask=valid, other=0.0)

    # ── out = gate + scale * in4 ─────────────────────────────────────────
    out_val = gate + scale * x4  # [BLOCK_N, BLOCK_D]

    # ── store to out[0, i, h*D+d]  ───────────────────────────────────────
    # out shape: [1, N+1, 8*D]
    # Element [0, i, h*D+d]  →  flat index = i*(8*D) + h*D + d
    out_base = i[:, None] * (8 * D) + h[None, :] * D + d_off[None, :]  # [BLOCK_N, BLOCK_D]
    tl.store(out_ptr + out_base, out_val, mask=valid)