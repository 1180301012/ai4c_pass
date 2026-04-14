import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: softmax(in_1, dim=1) -> multiply with in_0 -> sum(dim=1)
# Inputs:
#   in_0 : [B, K, C, H, W]   (feature maps)
#   in_1 : [B, K, C, 1, 1]   (attention logits, K=2)
# Output:
#   tmp_2 : [B, C, H, W]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused softmax-weighted sum with maximised occupancy
#
# Design:
#  • Flat 1-D grid → near-100 % thread occupancy on A30 (57 344 max threads)
#  • HW and CxHW as tl.constexpr → compiler replaces integer division with
#    a fast multiply-by-reciprocal sequence (~3 cycles instead of ~20)
#  • in0 loads issued BEFORE softmax arithmetic so memory latency overlaps
#    with exp / div compute (instruction-level prefetch)
#  • No explicit .to(tl.float32) on in0 → avoids noop convert instructions
#    for float32 paths; Triton auto-promotes when multiplied with fp32 s0/s1
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['B', 'K', 'C', 'HW'],
)
@triton.jit
def _fused_softmax_weighted_sum_k2_kernel(
    in0_ptr,    # [B, K, C, H, W]  contiguous
    in1_ptr,    # [B, K, C, 1, 1]  contiguous
    out_ptr,    # [B, C, H, W]     contiguous
    K, C,
    B:    tl.constexpr,    # batch  – constexpr allows dead-code of b arithmetic for B=1
    HW:   tl.constexpr,    # H*W    – constexpr for fast integer division
    CxHW: tl.constexpr,    # C*HW   – constexpr for fast integer division
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * CxHW
    mask = offs < total

    # ---- OPTIMISED B=1 fast path ----------------------------------------
    # For B=1, base = c*HW + hw = c*HW + (offs - c*HW) = offs  (identity!)
    # So in0 loads become trivially in0_ptr + offs (no multiply needed).
    # Only 1 integer division is required for the in1 index (offs // HW).
    if B == 1:
        c  = offs // HW                   # constexpr HW → fast reciprocal mult

        # in0: k=0 and k=1 slices at exactly offs and offs+CxHW
        f0 = tl.load(in0_ptr + offs,        mask=mask, other=0.0)
        f1 = tl.load(in0_ptr + offs + CxHW, mask=mask, other=0.0)

        # in1: indexed only by channel c (no b term)
        v0 = tl.load(in1_ptr + c,     mask=mask, other=0.0).to(tl.float32)
        v1 = tl.load(in1_ptr + c + C, mask=mask, other=0.0).to(tl.float32)

    # ---- General path for B > 1 -----------------------------------------
    else:
        b    = offs // CxHW
        c_hw = offs - b * CxHW
        c    = c_hw // HW
        hw   = c_hw - c * HW
        base = b * (K * CxHW) + c * HW + hw

        f0 = tl.load(in0_ptr + base,        mask=mask, other=0.0)
        f1 = tl.load(in0_ptr + base + CxHW, mask=mask, other=0.0)

        in1_base = b * (K * C) + c
        v0 = tl.load(in1_ptr + in1_base,     mask=mask, other=0.0).to(tl.float32)
        v1 = tl.load(in1_ptr + in1_base + C, mask=mask, other=0.0).to(tl.float32)

    # ---- Numerically-stable softmax over K=2 ----------------------------
    max_v = tl.maximum(v0, v1)
    e0    = tl.exp(v0 - max_v)
    e1    = tl.exp(v1 - max_v)
    inv   = 1.0 / (e0 + e1)
    s0    = e0 * inv
    s1    = e1 * inv

    # ---- Weighted sum and store -----------------------------------------
    tl.store(out_ptr + offs, s0 * f0 + s1 * f1, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_softmax_weighted_sum(in_0, in_1):
    B, K, C, H, W = in_0.shape
    HW   = H * W
    CxHW = C * HW
    total = B * CxHW

    out = torch.empty((B, C, H, W), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    _fused_softmax_weighted_sum_k2_kernel[grid](
        in_0, in_1, out,
        K, C, B, HW, CxHW,
    )

    return out


def replacement_func():
    return fused_softmax_weighted_sum