import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton layer-norm kernel: one program per row, BLOCK_D >= D
# Autotuned for D=768, eps=1e-6 (normalized_shape=(768,))
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 1024}, num_warps=4),
        triton.Config({'BLOCK_D': 1024}, num_warps=8),
        triton.Config({'BLOCK_D': 1024}, num_warps=16),
    ],
    key=['D'],
)
@triton.jit
def _ln_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    D,
    BLOCK_D: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    base = row * D

    x    = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    diff = x - mean
    var  = tl.sum(diff * diff, axis=0) / D
    rstd = tl.rsqrt(var + 1e-6)
    xhat = diff * rstd

    w   = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b   = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = xhat * w + b
    tl.store(out_ptr + base + offs, out.to(x_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Pattern: layer_norm(x, (768,), w, b, 1e-06) → single output
# ---------------------------------------------------------------------------
def pattern(x, ln_weight, ln_bias):
    return torch.nn.functional.layer_norm(x, (768,), ln_weight, ln_bias, 1e-06)


def replacement_args(x, ln_weight, ln_bias):
    return (x, ln_weight, ln_bias)


# ---------------------------------------------------------------------------
# @torch.fx.wrap kernel wrapper (only tensor allocation APIs used here)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_layer_norm_768(x, ln_weight, ln_bias):
    D = x.shape[-1]          # 768
    N = x.numel() // D       # number of rows (e.g. 981)
    out = torch.empty_like(x)
    _ln_kernel[(N,)](x, ln_weight, ln_bias, out, D)
    return out


def replacement_func():
    return triton_layer_norm_768