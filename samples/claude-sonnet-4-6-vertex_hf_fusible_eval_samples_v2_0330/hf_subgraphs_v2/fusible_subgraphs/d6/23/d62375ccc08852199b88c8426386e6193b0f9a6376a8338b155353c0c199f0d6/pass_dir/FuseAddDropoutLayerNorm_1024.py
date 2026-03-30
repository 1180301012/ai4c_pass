"""
LayerNorm pass for hidden_dim=1024 (float32 / bfloat16).

Pattern: single layer_norm replacement (framework supports only single-return).
Replacement uses Triton kernel for the layer_norm computation.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern — single layer_norm, single return value
# ---------------------------------------------------------------------------

def pattern(x, weight, bias):
    norm_result = torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)
    return norm_result


def replacement_args(x, weight, bias):
    return (x, weight, bias)


# ---------------------------------------------------------------------------
# Triton kernel — layer-norm for 1024-wide rows, no autotune overhead
# ---------------------------------------------------------------------------

@triton.jit
def _ln_1024_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N:   tl.constexpr,
    eps: tl.constexpr,
    BS:  tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BS)

    x = tl.load(x_ptr + row * N + offs)
    a = x.to(tl.float32)

    mean = tl.sum(a, axis=0) * (1.0 / N)
    diff = a - mean
    var  = tl.sum(diff * diff, axis=0) * (1.0 / N)
    rstd = 1.0 / tl.sqrt(var + eps)
    norm = diff * rstd

    w = tl.load(w_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)

    out = (norm * w + b).to(x.dtype)
    tl.store(out_ptr + row * N + offs, out)


# ---------------------------------------------------------------------------
# Python wrapper — minimal overhead
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_ln_1024(x, weight, bias):
    out    = torch.empty_like(x)
    n_rows = x.numel() >> 10     # // 1024

    _ln_1024_kernel[(n_rows,)](
        x, weight, bias, out,
        N=1024, eps=1e-05, BS=1024,
        num_warps=8,
    )

    return out


# ---------------------------------------------------------------------------
# Required by the pass framework
# ---------------------------------------------------------------------------

def replacement_func():
    return triton_ln_1024