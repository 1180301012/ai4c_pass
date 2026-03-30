"""
Optimization pass: Fuse split([32,32,128], dim=3) + permute(0,2,1,3) x3 + transpose(-2,-1)
into a single Triton kernel.

Input:  x [B, 49, 8, 192]  (contiguous, output of reshape after linear)
Output: q [B, 8, 49, 32]   (permuted Q)
        k [B, 8, 32, 49]   (permuted + transposed K)
        v [B, 8, 49, 128]  (permuted V)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_S': 64}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_S': 64}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_S': 64}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_S': 64}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 64}, num_warps=16, num_stages=3),
    ],
    key=['B', 'S', 'H'],
    warmup=25,
    rep=100,
)
@triton.jit
def _fused_qkv_split_permute_kernel(
    x_ptr,   # [B, S, H, D_total]  contiguous
    q_ptr,   # [B, H, S, DQ]
    k_ptr,   # [B, H, DK, S]
    v_ptr,   # [B, H, S, DV]
    B, S, H,
    DQ: tl.constexpr,       # 32
    DK: tl.constexpr,       # 32
    DV: tl.constexpr,       # 128
    DV_HALF: tl.constexpr,  # 64
    D_TOTAL: tl.constexpr,  # 192
    BLOCK_S: tl.constexpr,  # padded S dimension (power-of-2 >= S)
):
    # One program per (b, h) pair
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H

    # Stride for x:  x[b, s, h, d] offset = b*S*H*D_TOTAL + s*H*D_TOTAL + h*D_TOTAL + d
    x_bh_base = b * S * H * D_TOTAL + h * D_TOTAL

    s = tl.arange(0, BLOCK_S)      # padded S range
    mask_s = s < S

    # ------------------------------------------------------------------ Q
    # Read x[b, s, h, 0:DQ]  →  write q[b, h, s, 0:DQ]
    # q layout: [B, H, S, DQ]  →  q[b, h, s, dq] = b*H*S*DQ + h*S*DQ + s*DQ + dq
    dq = tl.arange(0, DQ)
    x_q_off = x_bh_base + s[:, None] * (H * D_TOTAL) + dq[None, :]
    q_vals   = tl.load(x_ptr + x_q_off, mask=mask_s[:, None], other=0.0)
    q_off    = b * H * S * DQ + h * S * DQ + s[:, None] * DQ + dq[None, :]
    tl.store(q_ptr + q_off, q_vals, mask=mask_s[:, None])

    # ------------------------------------------------------------------ K  (transpose!)
    # Read x[b, s, h, DQ:DQ+DK]  →  write k[b, h, dk, s]  (transposed)
    # k layout: [B, H, DK, S]  →  k[b, h, dk, s] = b*H*DK*S + h*DK*S + dk*S + s
    dk = tl.arange(0, DK)
    # Access pattern: s varies across columns, dk varies across rows → DK × BLOCK_S tile
    x_k_off = x_bh_base + s[None, :] * (H * D_TOTAL) + (DQ + dk[:, None])
    k_vals   = tl.load(x_ptr + x_k_off, mask=mask_s[None, :], other=0.0)
    k_off    = b * H * DK * S + h * DK * S + dk[:, None] * S + s[None, :]
    tl.store(k_ptr + k_off, k_vals, mask=mask_s[None, :])

    # ------------------------------------------------------------------ V  (split across two 64-wide sub-blocks)
    # Read x[b, s, h, DQ+DK : DQ+DK+DV]  →  write v[b, h, s, 0:DV]
    # v layout: [B, H, S, DV]  →  v[b, h, s, dv] = b*H*S*DV + h*S*DV + s*DV + dv

    # First half: dv in [0, 64)
    dv0 = tl.arange(0, DV_HALF)
    x_v0_off = x_bh_base + s[:, None] * (H * D_TOTAL) + (DQ + DK + dv0[None, :])
    v0_vals  = tl.load(x_ptr + x_v0_off, mask=mask_s[:, None], other=0.0)
    v0_off   = b * H * S * DV + h * S * DV + s[:, None] * DV + dv0[None, :]
    tl.store(v_ptr + v0_off, v0_vals, mask=mask_s[:, None])

    # Second half: dv in [64, 128)
    dv1 = tl.arange(0, DV_HALF) + DV_HALF
    x_v1_off = x_bh_base + s[:, None] * (H * D_TOTAL) + (DQ + DK + dv1[None, :])
    v1_vals  = tl.load(x_ptr + x_v1_off, mask=mask_s[:, None], other=0.0)
    v1_off   = b * H * S * DV + h * S * DV + s[:, None] * DV + dv1[None, :]
    tl.store(v_ptr + v1_off, v1_vals, mask=mask_s[:, None])


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_qkv_split_permute(x):
    """
    x : [B, S, H, D_total]  contiguous  (D_total = DQ + DK + DV = 192)
    returns q [B, H, S, DQ], k [B, H, DK, S], v [B, H, S, DV]

    For small B, Triton kernel launch overhead exceeds any bandwidth savings
    (permute/split are zero-copy views in eager PyTorch).  Fall back to
    PyTorch view-ops so we match eager performance in those cases.
    """
    B, S, H, D_total = x.shape
    DQ, DK, DV = 32, 32, 128
    BH = B * H

    # Use Triton for all large batches (float32, float16, bfloat16):
    #   - torch.compile optimises GEMMs (fp32 most, fp16 moderately)
    #     and the Triton kernel overhead is less than the framework
    #     dispatch overhead for PyTorch views at large BH.
    #   - For small BH the kernel launch cost dominates; use views.
    if BH >= 256:
        # ── Triton materialisation kernel ─────────────────────────────
        if not x.is_contiguous():
            x = x.contiguous()
        q = torch.empty((B, H, S, DQ),  dtype=x.dtype, device=x.device)
        k = torch.empty((B, H, DK, S), dtype=x.dtype, device=x.device)
        v = torch.empty((B, H, S, DV),  dtype=x.dtype, device=x.device)
        grid = (BH,)
        _fused_qkv_split_permute_kernel[grid](
            x, q, k, v,
            B, S, H,
            DQ=DQ, DK=DK, DV=DV,
            DV_HALF=64,
            D_TOTAL=D_total,
        )
        return q, k, v
    else:
        # ── Zero-copy view path (matches eager behaviour exactly) ─────
        split = x.split([32, 32, 128], dim=3)
        q = split[0].permute(0, 2, 1, 3)
        k = split[1].permute(0, 2, 1, 3).transpose(-2, -1)
        v = split[2].permute(0, 2, 1, 3)
        return q, k, v


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(x):
    """
    Match:
        split([32, 32, 128], dim=3)
        → getitem[0], getitem[1], getitem[2]
        → permute(0,2,1,3) × 3
        → transpose(-2,-1) on K
    Returns (q_out, k_out, v_out) which map to (tmp_9, tmp_13, tmp_11)
    """
    split   = x.split([32, 32, 128], dim=3)
    q_raw   = split[0]
    k_raw   = split[1]
    v_raw   = split[2]
    q_out   = q_raw.permute(0, 2, 1, 3)
    k_perm  = k_raw.permute(0, 2, 1, 3)
    v_out   = v_raw.permute(0, 2, 1, 3)
    k_out   = k_perm.transpose(-2, -1)
    return (q_out, k_out, v_out)


def replacement_args(x):
    return (x,)


# The replacement body is NOT wrapped so FX can trace the getitem unpacking.
# This gives 3 distinct returning nodes (q, k, v) matching the pattern's 3 outputs.
def _replacement_body(x):
    result = fused_qkv_split_permute(x)
    q = result[0]
    k = result[1]
    v = result[2]
    return (q, k, v)


def replacement_func():
    return _replacement_body