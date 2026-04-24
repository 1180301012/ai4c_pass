"""
Shared Triton kernels for fused SE-attention post-conv operations.

Pattern:  (conv_out + SCALE_BIAS) / SCALE_DIV  clamp_(0,1)  * x2

conv_out: [B, C, 1, 1]  – element (b,c,0,0) at flat index b*C + c.
x2:       [B, C, H, W]  – fully contiguous NCHW.
out:      [B, C, H, W]

Grid: (B*C,)  – one program per (b,c) pair.
Each program:
  1. Loads scale once as a scalar (no per-element division).
  2. Loops over HW in BLOCK_HW-element chunks (BLOCK_HW=256, single JIT version).
  3. BLOCK_HW=256 covers all HW sizes with 1-5 iterations (no recompilation needed
     for different HW values, since JIT compiles one version and reuses it).
  4. Grid passed as a plain tuple to avoid per-call Python overhead.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused kernel  (BLOCK_HW=256 is the ONLY JIT-compiled variant – single cache
# entry reused for every HW value; loop handles the rest)
# ---------------------------------------------------------------------------
@triton.jit
def scale_broadcast_2d_kernel(
    scale_ptr,   # [B, C, 1, 1] – flat index bc = b*C+c
    x2_ptr,      # [B, C, H, W] – contiguous
    out_ptr,     # [B, C, H, W] – contiguous
    HW,          # H*W
    SCALE_BIAS: tl.constexpr,
    SCALE_DIV:  tl.constexpr,
    BLOCK_HW:   tl.constexpr,
):
    bc_idx = tl.program_id(0)

    # 1. Load scale once (scalar)
    scale_v = tl.load(scale_ptr + bc_idx).to(tl.float32)
    scale_f = (scale_v + SCALE_BIAS) / SCALE_DIV
    scale_f = tl.maximum(scale_f, 0.0)
    scale_f = tl.minimum(scale_f, 1.0)

    # 2. Loop over HW in BLOCK_HW chunks (no per-element division)
    base = bc_idx * HW
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offs = hw_start + tl.arange(0, BLOCK_HW)
        mask    = hw_offs < HW
        x2_v    = tl.load(x2_ptr + base + hw_offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + base + hw_offs, (x2_v * scale_f).to(x2_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper – IDENTICAL across both pass files.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_se_dispatch(bias_or_conv_out, x2, scale_bias, scale_div):
    """
    Dispatch wrapper used by both FuseSEAttnScaleBias passes.
    Returns a single tensor (the model's return wraps it in a tuple).
    """
    B, C, H, W = x2.shape
    HW = H * W
    BC = B * C
    out = torch.empty_like(x2)

    # BLOCK_HW=256 → ONE JIT-compiled kernel used for ALL HW values.
    # No recompilation during timing trials (unlike larger BLOCK_HW values
    # that would trigger fresh JIT for each unique size).
    scale_broadcast_2d_kernel[(BC,)](
        bias_or_conv_out, x2, out, HW,
        scale_bias, scale_div,
        BLOCK_HW=256,
        num_warps=4,
    )

    return out