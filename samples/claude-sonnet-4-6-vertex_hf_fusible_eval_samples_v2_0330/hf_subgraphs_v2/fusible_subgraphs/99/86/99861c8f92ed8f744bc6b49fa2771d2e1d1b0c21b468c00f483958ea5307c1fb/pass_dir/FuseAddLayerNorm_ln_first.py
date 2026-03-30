"""
Fused Add + LayerNorm pass — returns (ln_result, sum).
Matches: galsenai_hubert-large graphs (float16 & bfloat16).

Computation:
  tmp_2 = in_2 + in_3
  tmp_4 = layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
  return (tmp_4, tmp_2)

Fusion: single kernel reads in_2/in_3 once, writes ln_result and sum.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused elementwise-add + layer-norm
# One Triton program per row; each row has C=1024 elements.
#
# Optimizations:
#  • No autotune overhead — fixed config (num_warps=8).
#  • Mean + variance computed via sum(x) and sum(x²) simultaneously.
#    Var(x) = E[x²] - E[x]²
#  • Both outputs stored after all register computation.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_add_ln_kernel_lnfirst(
    in2_ptr, in3_ptr,          # inputs to add  [rows, C]
    weight_ptr, bias_ptr,      # LN weight/bias  [C]
    sum_ptr, ln_ptr,           # outputs         [rows, C]
    C: tl.constexpr,           # normalized dim (1024)
    eps: tl.constexpr,         # layer-norm epsilon
    BLOCK_C: tl.constexpr,     # tile size along C (== C)
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)          # [BLOCK_C]
    base = row * C

    # ---- Load inputs and compute sum (stays in registers) -----------------
    x2 = tl.load(in2_ptr + base + offs)   # fp16 / bf16
    x3 = tl.load(in3_ptr + base + offs)
    x  = x2 + x3                          # still fp16/bf16

    # ---- Upcast and compute mean + variance in one pass -------------------
    xf    = x.to(tl.float32)
    xf2   = xf * xf

    sum_x  = tl.sum(xf,  axis=0)
    sum_x2 = tl.sum(xf2, axis=0)

    mean   = sum_x  / C
    var    = sum_x2 / C - mean * mean
    rstd   = tl.rsqrt(var + eps)

    # ---- Load LN parameters and normalise ---------------------------------
    w   = tl.load(weight_ptr + offs).to(tl.float32)
    b   = tl.load(bias_ptr   + offs).to(tl.float32)
    out = (xf - mean) * rstd * w + b

    # ---- Store BOTH outputs after all register computation ----------------
    tl.store(sum_ptr + base + offs, x)
    tl.store(ln_ptr  + base + offs, out.to(x.dtype))


# ---------------------------------------------------------------------------
# Kernel launcher — @torch.fx.wrap makes it opaque to FX tracing
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_add_ln_impl_lf(in_0, in_1, in_2, in_3):
    """in_0=bias, in_1=weight, in_2/in_3=tensors to add.
    Returns a 2-tuple: (ln_out, sum_out)  [ln first, matching galsenai return order]."""
    C   = 1024
    eps = 1e-5
    rows = in_2.numel() // C

    sum_out = torch.empty_like(in_2)
    ln_out  = torch.empty_like(in_2)

    _fused_add_ln_kernel_lnfirst[(rows,)](
        in_2, in_3,
        in_1, in_0,       # weight, bias
        sum_out, ln_out,
        C=C,
        eps=eps,
        BLOCK_C=C,
        num_warps=16,
    )
    # galsenai graphs return (layer_norm_result, sum)
    return (ln_out, sum_out)


# ---------------------------------------------------------------------------
# Replacement function — NOT @torch.fx.wrap so FX traces through it,
# creating explicit getitem nodes for each tuple element.
# This makes copied_returning_nodes length == 2, matching pattern's length.
# ---------------------------------------------------------------------------
def _fused_add_ln_ln_first(in_0, in_1, in_2, in_3):
    """Replacement traced by FX: unpacks the 2-tuple so each output is a
    separate FX node, matching the 2 returning_nodes of the pattern."""
    result  = _fused_add_ln_impl_lf(in_0, in_1, in_2, in_3)
    ln_out  = result[0]
    sum_out = result[1]
    return (ln_out, sum_out)


# ---------------------------------------------------------------------------
# Pattern / replacement API required by the AI4C pass framework
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    """Matches: add → layer_norm → return (ln, sum)."""
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return (tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return _fused_add_ln_ln_first