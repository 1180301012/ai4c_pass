"""
Minimal test pass: match ONLY layer_norm(x, (1024,), weight, bias, 1e-05)
and replace with a Triton kernel.  One pass, one pattern, no cross-file issues.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    stride, N, eps,
    BLOCK: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x  = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    xs = tl.where(mask, x, 0.0)
    mean   = tl.sum(xs, 0) / N
    diff   = tl.where(mask, x - mean, 0.0)
    var    = tl.sum(diff * diff, 0) / N
    istd   = 1.0 / tl.sqrt(var + eps)
    xn     = (x - mean) * istd

    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = xn * w + b

    if IS_BF16:
        tl.store(out_ptr + row * stride + offs, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row * stride + offs, out.to(tl.float16),  mask=mask)


@torch.fx.wrap
def fused_ln_h1024(x, weight, bias):
    N    = x.shape[-1]
    rows = x.numel() // N
    out  = torch.empty_like(x)
    BLOCK   = triton.next_power_of_2(N)
    BLOCK   = max(BLOCK, 16)
    IS_BF16 = (x.dtype == torch.bfloat16)
    nw = 8 if BLOCK >= 512 else (4 if BLOCK >= 128 else (2 if BLOCK >= 32 else 1))
    _ln_kernel[(rows,)](x, weight, bias, out, N, N, 1e-5,
                        BLOCK=BLOCK, IS_BF16=IS_BF16, num_warps=nw)
    return out


def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


def replacement_func():
    return fused_ln_h1024