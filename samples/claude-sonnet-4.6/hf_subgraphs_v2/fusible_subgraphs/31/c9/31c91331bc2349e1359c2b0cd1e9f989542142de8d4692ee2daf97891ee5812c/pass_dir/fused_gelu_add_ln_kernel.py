"""
Shared Triton kernel for fusing:
  GELU(in_2[1,C,H,W]) → flatten/transpose → add in_3[1,N,C] → LayerNorm

in_2 layout [1, C, N] contiguous  →  element [0, c, n]  at  c*N + n
in_3 layout [1, N, C] contiguous  →  element [0, n, c]  at  n*C + c

One Triton program per spatial row n ∈ [0, N).

BLOCK_C == C (power-of-2 feature size); EPS is the LayerNorm epsilon.
Both are tl.constexpr to avoid runtime type-promotion issues.
"""

import triton
import triton.language as tl


@triton.jit
def fused_gelu_add_layernorm_kernel(
    in2_ptr,                  # [1, C, N]  strided along C
    in3_ptr,                  # [1, N, C]  contiguous
    weight_ptr,               # [C]  LN scale
    bias_ptr,                 # [C]  LN bias
    out_ptr,                  # [1, N, C]  tmp_10 (pre-LN)
    ln_out_ptr,               # [1, N, C]  tmp_11 (post-LN)
    N,                        # H * W  (runtime)
    BLOCK_C: tl.constexpr,   # == C, power-of-2
    EPS: tl.constexpr,       # layer-norm epsilon
):
    row = tl.program_id(0)
    c_offs = tl.arange(0, BLOCK_C)

    # ---- GELU(in_2[0, :, row]) – stride N between channels ----------------
    x2 = tl.load(in2_ptr + c_offs * N + row).to(tl.float32)
    x2_gelu = x2 * 0.5 * (1.0 + tl.math.erf(x2 * 0.7071067811865476))

    # ---- in_3[0, row, :] contiguous ----------------------------------------
    x3 = tl.load(in3_ptr + row * BLOCK_C + c_offs).to(tl.float32)

    # ---- element-wise add → tmp_10 -----------------------------------------
    x = x2_gelu + x3
    tl.store(out_ptr + row * BLOCK_C + c_offs, x)   # auto-cast to dest dtype

    # ---- layer norm -----------------------------------------------------------
    mean    = tl.sum(x, axis=0) / BLOCK_C
    xc      = x - mean
    var     = tl.sum(xc * xc, axis=0) / BLOCK_C
    rstd    = 1.0 / tl.sqrt(var + EPS)
    x_norm  = xc * rstd

    ww = tl.load(weight_ptr + c_offs).to(tl.float32)
    bb = tl.load(bias_ptr   + c_offs).to(tl.float32)
    tl.store(ln_out_ptr + row * BLOCK_C + c_offs, x_norm * ww + bb)