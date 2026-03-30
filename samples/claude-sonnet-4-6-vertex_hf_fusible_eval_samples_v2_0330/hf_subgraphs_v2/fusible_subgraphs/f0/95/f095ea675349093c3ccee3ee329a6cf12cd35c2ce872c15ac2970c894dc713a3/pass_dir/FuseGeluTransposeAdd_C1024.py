"""
Fused pass for: GELU + transpose(1,2) + residual add

Pattern (no dropout, no layer_norm):
  tmp_5 = gelu(tmp_4)            # tmp_4: [B, C, T]
  tmp_6 = tmp_5.transpose(1, 2)  # [B, T, C]
  tmp_7 = in_3 + tmp_6           # [B, T, C]
  return tmp_7

This pattern is dtype-agnostic and dropout-rate-agnostic, so it matches
ALL THREE TARGET GRAPHS (bfloat16/float16/float32).

The Triton kernel fuses the three ops into a single pass, saving:
  - 2 intermediate tensor allocations
  - ~3.5 MB → 1.5 MB of HBM traffic (for [1,1024,249] shapes)
  - 2 extra kernel launches
"""

import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------ #
# Triton kernel
# ------------------------------------------------------------------ #

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["C", "T"],
)
@triton.jit
def _gelu_transpose_add_kernel(
    # Inputs
    tmp4_ptr,   # [B, C, T_slice]  – may be non-contiguous
    in3_ptr,    # [B, T, C]        – contiguous
    # Output
    out_ptr,    # [B, T, C]
    # Dims
    B, T, C,
    # Strides for tmp4
    tmp4_s0, tmp4_s1, tmp4_s2,
    # Strides for in3
    in3_s0, in3_s1, in3_s2,
    # Strides for out
    out_s0, out_s1, out_s2,
    # Compile-time block
    BLOCK_C: tl.constexpr,
):
    """One program per (b, t) row — processes C=1024 channels."""
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T

    c_offsets = tl.arange(0, BLOCK_C)

    # ---- 1. Load tmp4[b, :, t]  (stride over C dimension) ----------
    x_raw = tl.load(tmp4_ptr + b * tmp4_s0 + c_offsets * tmp4_s1 + t * tmp4_s2)
    x_f32 = x_raw.to(tl.float32)

    # ---- 2. Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) ----------
    SQRT2_INV: tl.constexpr = 0.7071067811865476
    gelu_f32 = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * SQRT2_INV))

    # Round GELU result back to the native dtype, then expand to float32.
    # This exactly replicates PyTorch eager: gelu() returns a tensor in the
    # input dtype (not float32), so the subsequent add operates on dtype-rounded
    # values.  Without this step, BF16 / FP16 outputs would differ numerically.
    gelu_rounded = gelu_f32.to(x_raw.dtype).to(tl.float32)

    # ---- 3. Load in3[b, t, :]  (contiguous over C) -----------------
    y_raw = tl.load(in3_ptr + b * in3_s0 + t * in3_s1 + c_offsets * in3_s2)
    y_f32 = y_raw.to(tl.float32)

    # ---- 4. Residual add  (y + gelu, both are dtype-rounded values) ---
    result = y_f32 + gelu_rounded

    # ---- 5. Store in original dtype --------------------------------
    out_base = b * out_s0 + t * out_s1
    tl.store(out_ptr + out_base + c_offsets * out_s2, result.to(x_raw.dtype))


# ------------------------------------------------------------------ #
# @torch.fx.wrap kernel launcher
# ------------------------------------------------------------------ #

@torch.fx.wrap
def _gelu_transpose_add_launch(tmp_4, in_3):
    """
    Opaque FX leaf: runs the Triton kernel.
    tmp_4 : [B, C, T]   – gelu input (possibly non-contiguous)
    in_3  : [B, T, C]   – residual
    Returns tmp_7 : [B, T, C]
    """
    B, C, T = tmp_4.shape
    out = torch.empty(B, T, C, dtype=tmp_4.dtype, device=tmp_4.device)

    _gelu_transpose_add_kernel[(B * T,)](
        tmp_4, in_3, out,
        B, T, C,
        tmp_4.stride(0), tmp_4.stride(1), tmp_4.stride(2),
        in_3.stride(0),  in_3.stride(1),  in_3.stride(2),
        out.stride(0),   out.stride(1),   out.stride(2),
        BLOCK_C=1024,
    )
    return out


# Outer function: NOT @torch.fx.wrap so FX traces into it.
# The single returning node is the direct output of _gelu_transpose_add_launch,
# matching the single returning node (tmp_7) expected by the pattern.
def fused_gelu_transpose_add(tmp_4, in_3):
    return _gelu_transpose_add_launch(tmp_4, in_3)


# ------------------------------------------------------------------ #
# Pattern / replacement API
# ------------------------------------------------------------------ #

def pattern(tmp_4, in_3):
    tmp_5 = torch.nn.functional.gelu(tmp_4)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = in_3 + tmp_6
    return tmp_7


def replacement_args(tmp_4, in_3):
    return (tmp_4, in_3)


def replacement_func():
    return fused_gelu_transpose_add