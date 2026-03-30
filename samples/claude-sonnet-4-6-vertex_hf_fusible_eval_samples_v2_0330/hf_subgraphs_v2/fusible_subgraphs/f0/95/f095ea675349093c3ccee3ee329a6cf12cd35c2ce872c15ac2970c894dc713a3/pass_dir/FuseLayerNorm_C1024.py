"""
Second-pass optimization: LayerNorm over the last dimension (C=1024).

After FuseGeluTransposeAdd_C1024 runs, the graph becomes:
  my_gelu_add_result → dropout(identity) → tmp_8 → layer_norm → tmp_10
  return (tmp_8, tmp_10)

This pass matches just the layer_norm node.  The pattern is dtype-agnostic
(no dropout rate constant), so it applies to all three graphs.

Returning-node analysis:
  - tmp_8 is an INPUT to the matched subgraph (produced outside), not a returning node.
  - tmp_10 is produced inside and used outside  →  1 returning node  →  replacement returns 1 tensor.
"""

import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------ #
# Triton kernel for LayerNorm over C=1024
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
def _layernorm_kernel(
    inp_ptr,    # [B, T, C]  – input (tmp_8)
    w_ptr,      # [C]        – layer_norm weight  (in_1)
    b_ptr,      # [C]        – layer_norm bias    (in_0)
    out_ptr,    # [B, T, C]  – output (tmp_10)
    B, T, C,
    inp_s0, inp_s1, inp_s2,
    out_s0, out_s1, out_s2,
    eps,
    BLOCK_C: tl.constexpr,
):
    """One program per (b, t) row – normalises C=1024 values."""
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T
    c_offsets = tl.arange(0, BLOCK_C)

    # Load input row
    x_raw = tl.load(inp_ptr + b * inp_s0 + t * inp_s1 + c_offsets * inp_s2)
    x = x_raw.to(tl.float32)

    # Mean
    mean = tl.sum(x, axis=0) / C
    diff = x - mean

    # Variance (biased)
    var = tl.sum(diff * diff, axis=0) / C
    inv_std = 1.0 / tl.sqrt(var + eps)
    norm = diff * inv_std

    # Affine transform
    w = tl.load(w_ptr + c_offsets).to(tl.float32)
    b_ln = tl.load(b_ptr + c_offsets).to(tl.float32)
    out = norm * w + b_ln

    # Store
    tl.store(out_ptr + b * out_s0 + t * out_s1 + c_offsets * out_s2, out.to(x_raw.dtype))


# ------------------------------------------------------------------ #
# Wrappers
# ------------------------------------------------------------------ #

@torch.fx.wrap
def _layernorm_launch(tmp_8, in_1, in_0):
    """
    Opaque FX leaf.
    tmp_8 : [B, T, C]   – input
    in_1  : [C]         – weight
    in_0  : [C]         – bias
    Returns tmp_10 : [B, T, C]
    """
    B, T, C = tmp_8.shape
    out = torch.empty_like(tmp_8)

    _layernorm_kernel[(B * T,)](
        tmp_8, in_1, in_0, out,
        B, T, C,
        tmp_8.stride(0), tmp_8.stride(1), tmp_8.stride(2),
        out.stride(0),   out.stride(1),   out.stride(2),
        1e-5,
        BLOCK_C=1024,
    )
    return out


def fused_layernorm(tmp_8, in_1, in_0):
    """Traced by FX; calls the opaque kernel leaf."""
    return _layernorm_launch(tmp_8, in_1, in_0)


# ------------------------------------------------------------------ #
# Pattern / replacement API
# ------------------------------------------------------------------ #

def pattern(tmp_8, in_1, in_0):
    tmp_10 = torch.nn.functional.layer_norm(tmp_8, (1024,), in_1, in_0, 1e-05)
    return tmp_10


def replacement_args(tmp_8, in_1, in_0):
    return (tmp_8, in_1, in_0)


def replacement_func():
    return fused_layernorm