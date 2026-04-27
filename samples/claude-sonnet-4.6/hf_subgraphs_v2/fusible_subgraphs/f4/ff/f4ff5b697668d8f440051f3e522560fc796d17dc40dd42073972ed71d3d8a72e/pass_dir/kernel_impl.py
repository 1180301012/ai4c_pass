"""
Shared Triton kernel for fused softmax + weighted coordinate sum.

This fuses:
  1. softmax(in_2, dim=2)  [B, 17, 4096]
  2. reshape -> [B, 17, 64, 64]
  3. sum(softmax * in_0)   -> x-coordinate [B, 17, 1]
  4. sum(softmax * in_1)   -> y-coordinate [B, 17, 1]
  5. cat([x, y], dim=-1)   -> [B, 17, 2]

into a single GPU kernel pass over data.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 4096}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK': 4096}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK': 4096}, num_warps=32, num_stages=1),
    ],
    key=['BK'],
)
@triton.jit
def _fused_softmax_weighted_sum_kernel(
    in2_ptr,    # [B, K, 4096]  input heatmap logits (flat: BK x 4096)
    in0_ptr,    # [64]          x-linspace  (from in_0 [1,1,1,64])
    in1_ptr,    # [64]          y-linspace  (from in_1 [1,1,64,1])
    out3_ptr,   # [B, K, 4096]  softmax heatmap (flat)
    out10_ptr,  # [B, K, 2]     expected (x, y) coords (flat: BK x 2)
    BK,         # B * K  (used as autotune key)
    BLOCK: tl.constexpr,   # 4096
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)  # [0..4095]

    # ── Load the 4096 logits for this (b, k) pair, cast to fp32 ──
    x = tl.load(in2_ptr + pid * BLOCK + offsets).to(tl.float32)

    # ── Numerically-stable softmax ──
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, axis=0)
    softmax = x_exp / x_sum  # float32, sums to 1

    # ── Store softmax output (Triton auto-casts fp32 → pointer dtype) ──
    tl.store(out3_ptr + pid * BLOCK + offsets, softmax)

    # ── Column index j (0..63) and row index i (0..63) ──
    #    Viewing the 4096-vector as a 64x64 matrix: element n → row=n//64, col=n%64
    j = offsets % 64    # column  → x-coordinate index
    i = offsets // 64   # row     → y-coordinate index

    # ── Load linspace vectors (64 values each, cast to fp32) ──
    # in_0 shape [1,1,1,64]: contiguous, element j is at flat offset j
    in0 = tl.load(in0_ptr + j).to(tl.float32)
    # in_1 shape [1,1,64,1]: contiguous, element i is at flat offset i
    in1 = tl.load(in1_ptr + i).to(tl.float32)

    # ── Compute expected x and y coordinates ──
    sum_x = tl.sum(softmax * in0, axis=0)
    sum_y = tl.sum(softmax * in1, axis=0)

    # ── Store [x, y] pair ──
    tl.store(out10_ptr + pid * 2,     sum_x)
    tl.store(out10_ptr + pid * 2 + 1, sum_y)


@torch.fx.wrap
def fused_softmax_weighted_sum(in_0, in_1, in_2):
    """
    Wrapper that launches the fused kernel.

    in_0 : [1, 1, 1, 64]  – x linspace
    in_1 : [1, 1, 64, 1]  – y linspace
    in_2 : [B, 17, 4096]  – heatmap logits

    Returns:
      out3  : [B, 17, 64, 64]  – softmax heatmap
      out10 : [B, 17, 2]       – expected (x, y) coordinates
    """
    B    = in_2.shape[0]
    K    = in_2.shape[1]   # 17 joints
    BK   = B * K
    BLOCK = 4096

    # Allocate outputs (dtype matches input)
    out3  = torch.empty((B, K, 64, 64), dtype=in_2.dtype, device=in_2.device)
    out10 = torch.empty((B, K, 2),      dtype=in_2.dtype, device=in_2.device)

    # One program per (batch, joint) pair
    _fused_softmax_weighted_sum_kernel[(BK,)](
        in_2, in_0, in_1,
        out3, out10,
        BK=BK,
        BLOCK=BLOCK,
    )

    return out3, out10