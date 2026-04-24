"""
Fused pass: matches  tmp_3 * in_0 -> sum(dim=1) -> contiguous()
and replaces with a single Triton kernel.

This is a weighted-sum pattern where:
  tmp_3  shape: [B, 2, H, 1, 1]  (softmax weights, already computed)
  in_0   shape: [B, 2, H, C, W]
  output  shape: [B, H, C, W]

The pattern is batch-size agnostic and matches ALL six graphs.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64},   num_warps=2),
        triton.Config({'BLOCK_W': 64},   num_warps=4),
        triton.Config({'BLOCK_W': 32},   num_warps=1),  # optimal for W=48 (1 iter, 100%)
        triton.Config({'BLOCK_W': 128},  num_warps=4),
        triton.Config({'BLOCK_W': 128},  num_warps=8),
        triton.Config({'BLOCK_W': 64},   num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 64},   num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 256},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 512},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_W': 1024}, num_warps=16, num_stages=2),
    ],
    key=['H', 'C', 'W'],
)
@triton.jit
def _ws_kernel(
    w_ptr,    # [B, 2, H, 1, 1]  — w[b,g,h,0,0]
    x_ptr,    # [B, 2, H, C, W]
    out_ptr,  # [B, H, C, W]
    H, C, W,
    BLOCK_W: tl.constexpr,
):
    """
    One program per (b, h, c).  Iterates over W in tiles of BLOCK_W.

    w layout [B, 2, H, 1, 1]:
      w[b, g, h, 0, 0] = w_ptr + b*(2*H) + g*H + h

    x layout [B, 2, H, C, W]:
      x[b, g, h, c, w] = x_ptr + b*(2*H*C*W) + g*(H*C*W) + h*(C*W) + c*W + w

    out layout [B, H, C, W]:
      out[b, h, c, w] = out_ptr + b*(H*C*W) + h*(C*W) + c*W + w
    """
    pid   = tl.program_id(0)
    c_idx = pid % C
    h_idx = (pid // C) % H
    b_idx = pid // (H * C)

    # Load pre-computed softmax weights (already normalized, NO re-softmax needed)
    w0 = tl.load(w_ptr + b_idx * (2 * H) + h_idx).to(tl.float32)
    w1 = tl.load(w_ptr + b_idx * (2 * H) + H + h_idx).to(tl.float32)

    # Base offsets into in_0 (group 0 and group 1) and output
    xb0 = b_idx * (2 * H * C * W) + 0 * (H * C * W) + h_idx * (C * W) + c_idx * W
    xb1 = b_idx * (2 * H * C * W) + 1 * (H * C * W) + h_idx * (C * W) + c_idx * W
    ob  = b_idx * (H * C * W) + h_idx * (C * W) + c_idx * W

    for w_start in tl.range(0, W, BLOCK_W):
        offs = w_start + tl.arange(0, BLOCK_W)
        mask = offs < W

        v0 = tl.load(x_ptr + xb0 + offs, mask=mask, other=0.0).to(tl.float32)
        v1 = tl.load(x_ptr + xb1 + offs, mask=mask, other=0.0).to(tl.float32)

        # Direct weighted sum — NO softmax (tmp_3 is already normalized weights)
        result = w0 * v0 + w1 * v1

        tl.store(out_ptr + ob + offs, result, mask=mask)


@torch.fx.wrap
def fused_weighted_sum(tmp_3, in_0):
    """
    tmp_3 : [B, 2, H, 1, 1]  (softmax weights, already normalized)
    in_0  : [B, 2, H, C, W]
    returns [B, H, C, W]
    """
    B, G, H = tmp_3.shape[0], tmp_3.shape[1], tmp_3.shape[2]
    C = in_0.shape[3]
    W = in_0.shape[4]

    out = torch.empty((B, H, C, W), dtype=in_0.dtype, device=in_0.device)
    # Autotune selects optimal BLOCK_W per (H, C, W) shape
    _ws_kernel[(B * H * C,)](tmp_3, in_0, out, H, C, W)
    return out


# ─── Pattern / replacement API ────────────────────────────────────────────────

def pattern(tmp_3, in_0):
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(tmp_3, in_0):
    return (tmp_3, in_0)


def replacement_func():
    return fused_weighted_sum