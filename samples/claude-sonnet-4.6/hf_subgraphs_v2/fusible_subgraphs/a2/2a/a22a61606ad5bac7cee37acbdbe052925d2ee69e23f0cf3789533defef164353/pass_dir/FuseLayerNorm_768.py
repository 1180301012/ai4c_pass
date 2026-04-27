"""
Pass: FuseLayerNorm_768
Replace layer_norm(x, (768,), weight, bias, 1e-12) with a Triton kernel.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ln_768(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x_orig = tl.load(X_ptr + row * stride + offsets, mask=mask, other=0.0)
    x = x_orig.to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    xn = xc * rstd

    w = tl.load(W_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    y = xn * w + b
    tl.store(Y_ptr + row * stride + offsets, y.to(x_orig.dtype), mask=mask)


def pattern(x, ln_weight, ln_bias):
    return torch.nn.functional.layer_norm(x, (768,), ln_weight, ln_bias, 1e-12)


@torch.fx.wrap
def triton_ln_768(x, ln_weight, ln_bias):
    N = 768
    M = x.numel() // N
    out = torch.empty_like(x)
    _ln_768[(M,)](x, ln_weight, ln_bias, out, N, N, 1e-12,
                  BLOCK_SIZE=1024, num_warps=8)
    return out


def replacement_args(x, ln_weight, ln_bias):
    return (x, ln_weight, ln_bias)


def replacement_func():
    return triton_ln_768