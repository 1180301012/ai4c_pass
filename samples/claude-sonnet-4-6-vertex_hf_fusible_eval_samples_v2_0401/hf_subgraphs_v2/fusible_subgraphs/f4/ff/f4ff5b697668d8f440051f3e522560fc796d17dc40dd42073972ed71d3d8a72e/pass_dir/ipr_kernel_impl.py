"""
Shared Triton kernel for IPR (Integral Pose Regression) fused computation.

Fuses: softmax(in_2, dim=2) -> reshape -> mul(linspace_x) -> sum
                                        -> mul(linspace_y) -> sum
                                        -> cat

into a single kernel that:
  1. Computes softmax over HW=4096 elements
  2. Simultaneously computes sum_x and sum_y (weighted sums)
  3. Outputs heatmap [B,K,H,W] and coords [B,K,2]

NOTE: Only the Triton kernel lives here. The @torch.fx.wrap wrapper must be
defined in each individual pass file so that the FX patcher can intercept it
correctly (the patcher patches functions in their defining module's globals,
and replacement_func() returns the function via a same-module globals lookup).
"""

import triton
import triton.language as tl


@triton.jit
def ipr_softmax_weighted_sum_kernel(
    in2_ptr,          # Input logits [B*K, HW]
    in0_ptr,          # linspace_x   [W]  (64 elements)
    in1_ptr,          # linspace_y   [H]  (64 elements)
    out_heatmap_ptr,  # Output heatmap [B*K, HW] (viewed as [B,K,H,W])
    out_coords_ptr,   # Output coords  [B*K, 2]  (viewed as [B,K,2])
    HW: tl.constexpr,          # H*W = 4096
    W:  tl.constexpr,          # W   = 64
    BLOCK_SIZE: tl.constexpr,  # Must equal HW = 4096
):
    """One Triton program per (batch, keypoint) row."""
    row_id = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)

    # ── Load row of logits ──────────────────────────────────────────────────
    x = tl.load(in2_ptr + row_id * HW + offs).to(tl.float32)

    # ── Numerically-stable softmax ──────────────────────────────────────────
    x_max  = tl.max(x, axis=0)
    x_exp  = tl.exp(x - x_max)
    x_sum  = tl.sum(x_exp, axis=0)
    soft   = x_exp / x_sum

    # ── Store heatmap (auto-cast to pointer dtype) ──────────────────────────
    tl.store(out_heatmap_ptr + row_id * HW + offs, soft)

    # ── Weighted sums ───────────────────────────────────────────────────────
    j = offs % W          # column index → linspace_x[j]
    i = offs // W         # row    index → linspace_y[i]

    lx = tl.load(in0_ptr + j).to(tl.float32)
    ly = tl.load(in1_ptr + i).to(tl.float32)

    sx = tl.sum(soft * lx, axis=0)
    sy = tl.sum(soft * ly, axis=0)

    # ── Store [sum_x, sum_y] ────────────────────────────────────────────────
    tl.store(out_coords_ptr + row_id * 2,     sx)
    tl.store(out_coords_ptr + row_id * 2 + 1, sy)