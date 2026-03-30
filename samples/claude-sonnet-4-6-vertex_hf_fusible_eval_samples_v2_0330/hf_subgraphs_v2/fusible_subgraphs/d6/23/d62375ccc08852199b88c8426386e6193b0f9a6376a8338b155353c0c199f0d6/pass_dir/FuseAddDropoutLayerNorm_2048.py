"""
Minimal pattern: match ONLY layer_norm with normalized_shape=(2048,).
Single return value, no dropout in pattern.
The dropout node remains unchanged in the graph; only layer_norm is replaced.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern — single layer_norm op, single return
# ---------------------------------------------------------------------------

def pattern(x, weight, bias):
    norm_result = torch.nn.functional.layer_norm(x, (2048,), weight, bias, 1e-05)
    return norm_result


def replacement_args(x, weight, bias):
    return (x, weight, bias)


# ---------------------------------------------------------------------------
# Triton kernel — one CUDA block per row (2048 elements/row)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _ln_2048(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + row * N + offs)
    a = x.to(tl.float32)

    mean = tl.sum(a, axis=0) / N
    diff = a - mean
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    norm = diff * rstd

    w = tl.load(weight_ptr + offs).to(tl.float32)
    b = tl.load(bias_ptr   + offs).to(tl.float32)

    out = (norm * w + b).to(x.dtype)
    tl.store(out_ptr + row * N + offs, out)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_ln_2048(x, weight, bias):
    orig_shape = x.shape
    x_flat  = x.reshape(-1, 2048)
    n_rows  = x_flat.shape[0]
    out     = torch.empty_like(x_flat)

    _ln_2048[(n_rows,)](
        x_flat, weight, bias, out,
        N=2048, eps=1e-05,
    )

    return out.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Required by the pass framework
# ---------------------------------------------------------------------------

def replacement_func():
    return triton_ln_2048