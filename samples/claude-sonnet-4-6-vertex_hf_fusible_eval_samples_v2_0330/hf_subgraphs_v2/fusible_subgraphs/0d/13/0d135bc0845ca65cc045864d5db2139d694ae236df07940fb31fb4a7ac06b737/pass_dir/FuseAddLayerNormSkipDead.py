"""
FuseAddLayerNormSkipDead.py

Pattern: add + layer_norm + getitem + linear + tanh
         The getitem → linear → tanh chain is dead code (result not returned).

Strategy:
  Use the same Triton fused add+layer_norm kernel as FuseAddLayerNorm.py
  but match the *larger* subgraph so the dead linear and tanh kernel launches
  are eliminated from the compiled graph, saving those extra GPU operations.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(in_5, in_6, in_1, in_2, in_4, in_3):
    tmp_5  = in_6 + in_5
    tmp_6  = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    tmp_7  = tmp_6[(slice(None, None, None), 0)]
    linear = torch.nn.functional.linear(tmp_7, in_4, in_3)
    tmp_9  = torch.tanh(linear)
    return tmp_6


def replacement_args(in_5, in_6, in_1, in_2, in_4, in_3):
    return (in_5, in_6, in_1, in_2, in_4, in_3)


# ---------------------------------------------------------------------------
# Triton kernel: fused add + layer-norm (identical to FuseAddLayerNorm.py)
#   BLOCK_SIZE=512, num_warps=4  → 128 threads, 4 elements/thread
#   Masked loads return 0.0, so xz is already 0 at invalid positions —
#   no redundant tl.where needed before the reductions.
# ---------------------------------------------------------------------------

@triton.jit
def fused_add_layernorm_skip_dead_kernel(
    x_ptr, z_ptr, w_ptr, b_ptr, out_ptr,
    D, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row     = tl.program_id(0)
    row_off = row * D
    cols    = tl.arange(0, BLOCK_SIZE)
    mask    = cols < D

    x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols,           mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols,           mask=mask, other=0.0).to(tl.float32)

    xz = x + z   # masked positions are 0.0 (loaded with other=0.0)

    # Parallel variance — both sums from the same in-register data
    s1   = tl.sum(xz,      axis=0)    # Σxᵢ  (zeros at invalid cols → correct)
    s2   = tl.sum(xz * xz, axis=0)   # Σxᵢ²
    mean = s1 / D
    var  = s2 / D - mean * mean
    rstd = tl.rsqrt(var + eps)

    out = (xz - mean) * rstd * w + b
    tl.store(out_ptr + row_off + cols, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper — accepts 6 args (in_4, in_3 are for the dead linear, ignored)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_layernorm(in_5, in_6, in_1, in_2, in_4, in_3):
    D   = in_5.shape[-1]
    N   = in_5.numel() // D
    out = torch.empty_like(in_5)

    fused_add_layernorm_skip_dead_kernel[(N,)](
        in_5, in_6, in_2, in_1, out,
        D,
        1e-12,
        BLOCK_SIZE=512,
        num_warps=4,
    )

    return out.view(in_5.shape)


def replacement_func():
    return fused_add_layernorm