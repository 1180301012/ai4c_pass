"""
Shared Triton kernel for fused softmax + weighted sum pattern.

Pattern: softmax(in1, dim=1) reshaped to [B, 2, C, 1, 1], multiplied with
         in0 [B, 2, C, H, W], summed over dim=1 -> [B, C, H, W]

Uses a 2D grid (hw_blocks x B*C) to avoid expensive integer divisions
in the hot path. Weights are scalar-loaded once per (b,c) block.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16, num_stages=3),
    ],
    key=['HW'],
)
@triton.jit
def _fused_softmax_weighted_sum_kernel(
    in0_ptr,            # [B, 2, C, H, W]  – main data tensor (contiguous)
    in1_ptr,            # [B, 2, 1, C]     – weight logits (contiguous)
    out_ptr,            # [B, C, H, W]     – output (contiguous)
    C,                  # number of channels (128 in all test cases)
    HW,                 # H * W
    BLOCK_HW: tl.constexpr,
):
    """
    Grid: (ceil(HW / BLOCK_HW),  B*C)
      pid0 = hw block index
      pid1 = b*C + c  (bc_idx)
    """
    # ── program indices ──────────────────────────────────────────────────────
    hw_pid  = tl.program_id(0)
    bc_idx  = tl.program_id(1)          # scalar

    b_idx   = bc_idx // C               # scalar integer division (once/program)
    c_idx   = bc_idx % C                # scalar modulo            (once/program)

    # ── HW offsets and mask ─────────────────────────────────────────────────
    hw_offs = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask    = hw_offs < HW

    # ── in_0 base pointers ──────────────────────────────────────────────────
    # in_0[b, k, c, hw] = b*2*C*HW + k*C*HW + c*HW + hw
    in0_k0_base = b_idx * 2 * C * HW + c_idx * HW
    in0_k1_base = in0_k0_base + C * HW

    x0 = tl.load(in0_ptr + in0_k0_base + hw_offs, mask=mask).to(tl.float32)
    x1 = tl.load(in0_ptr + in0_k1_base + hw_offs, mask=mask).to(tl.float32)

    # ── in_1 scalar weight loads ─────────────────────────────────────────────
    # in_1[b, k, 0, c] = b*2*C + k*C + c   (shape [B,2,1,C] contiguous)
    in1_k0 = b_idx * (2 * C) + c_idx
    in1_k1 = in1_k0 + C

    w0_raw = tl.load(in1_ptr + in1_k0).to(tl.float32)   # scalar
    w1_raw = tl.load(in1_ptr + in1_k1).to(tl.float32)   # scalar

    # ── numerically-stable softmax over K=2 ─────────────────────────────────
    max_w   = tl.maximum(w0_raw, w1_raw)
    e0      = tl.exp(w0_raw - max_w)
    e1      = tl.exp(w1_raw - max_w)
    inv_sum = 1.0 / (e0 + e1)
    w0      = e0 * inv_sum
    w1      = e1 * inv_sum

    # ── weighted sum ─────────────────────────────────────────────────────────
    out_f32 = w0 * x0 + w1 * x1

    # ── store (Triton auto-casts float32 → output tensor dtype) ──────────────
    out_base = bc_idx * HW
    tl.store(out_ptr + out_base + hw_offs, out_f32, mask=mask)


@torch.fx.wrap
def fused_softmax_weighted_sum(in0, in1):
    """
    in0 : [B, 2, C, H, W]
    in1 : [B, 2, 1, C]
    out : [B, C, H, W]   (same dtype as in0)

    Computes the fully-fused:
      weights = softmax(in1, dim=1)          # [B,2,1,C] -> [B,2,C,1,1] after reshape
      out     = (weights * in0).sum(dim=1)   # [B,C,H,W]
    """
    B  = in0.shape[0]
    C  = in0.shape[2]
    H  = in0.shape[3]
    W  = in0.shape[4]
    HW = H * W

    out = torch.empty(B, C, H, W, dtype=in0.dtype, device=in0.device)

    grid = lambda meta: (
        (HW + meta['BLOCK_HW'] - 1) // meta['BLOCK_HW'],
        B * C,
    )

    _fused_softmax_weighted_sum_kernel[grid](
        in0, in1, out,
        C, HW,
    )

    return out