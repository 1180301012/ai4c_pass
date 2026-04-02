"""
Fused pass for D=1024 (float32 / bfloat16 variants of fairseq-dense-355M).

Pattern: full chain from index_select to layer_norm, returning both observables.
Replacement: two @torch.fx.wrap kernels inside an *untraced* function so that
FX sees TWO separate output nodes, matching the two pattern outputs.

  Node 1 — _run_add_pe_1024 : positional-embed lookup + add  → out0 (=tmp_13)
  Node 2 — _run_ln_1024     : Triton layer-norm              → out1 (=tmp_14)

Reduces ~5 GPU kernel launches to 2.
"""

import torch
from torch import device
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _add_pe_kernel_1024(
    in0_ptr,   # [1, SEQ, D]
    in1_ptr,   # [2050, D]
    out_ptr,   # [1, SEQ, D]
    D: tl.constexpr,
):
    row    = tl.program_id(0)
    pe_row = row + 2
    offs   = tl.arange(0, D)
    x  = tl.load(in0_ptr + row * D + offs)
    pe = tl.load(in1_ptr + pe_row * D + offs)
    tl.store(out_ptr + row * D + offs,
             (x.to(tl.float32) + pe.to(tl.float32)).to(x.dtype))


@triton.jit
def _ln_kernel_1024(
    x_ptr,   # [1, SEQ, D]
    w_ptr,   # [D]
    b_ptr,   # [D]
    o_ptr,   # [1, SEQ, D]
    D: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, D)
    x    = tl.load(x_ptr + row * D + offs)
    xf   = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / D
    diff = xf - mean
    var  = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    yn   = diff * rstd
    w    = tl.load(w_ptr + offs).to(tl.float32)
    b    = tl.load(b_ptr + offs).to(tl.float32)
    tl.store(o_ptr + row * D + offs, (yn * w + b).to(x.dtype))


# ---------------------------------------------------------------------------
# @torch.fx.wrap helpers — each becomes one FX node
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _run_add_pe_1024(in_0, in_1):
    D, SEQ = 1024, 9
    out = torch.empty_like(in_0)
    _add_pe_kernel_1024[(SEQ,)](in_0, in_1, out, D=D, num_warps=4)
    return out


@torch.fx.wrap
def _run_ln_1024(x, weight, bias):
    D, SEQ = 1024, 9
    out = torch.empty_like(x)
    _ln_kernel_1024[(SEQ,)](x, weight, bias, out, D=D, num_warps=8)
    return out


# ---------------------------------------------------------------------------
# Replacement — NOT @torch.fx.wrap so FX traces through it → 2 output nodes
# ---------------------------------------------------------------------------

def fused_chain_1024(in_0, in_1, in_2, in_3):
    out0 = _run_add_pe_1024(in_0, in_1)     # replaces tmp_13
    out1 = _run_ln_1024(out0, in_3, in_2)   # replaces tmp_14
    return out0, out1


# ---------------------------------------------------------------------------
# Pattern / replacement_args / replacement_func
# ---------------------------------------------------------------------------

def pattern(x, weight, bias):
    """Match the single layer_norm computation (1 computed output)."""
    out1 = torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)
    return out1


def replacement_args(x, weight, bias):
    return (x, weight, bias)


def replacement_func():
    return _run_ln_1024