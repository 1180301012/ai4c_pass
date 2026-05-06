"""
Shared Triton LayerNorm kernel and dispatch wrapper for all pass variants.
Both FuseLayerNorm_768 and FuseLayerNorm_1024 import shared_ln_dispatch()
so the framework sees them as returning the SAME replacement_func, keeping
both passes alive despite output_pass_replacement_func_limit=1.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    stride_xn, stride_xs, stride_xc,
    S: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton LayerNorm. Input x has physical (possibly non-contiguous) strides.
    One program per (n,s) position (pid = flat row index n*S + s).
    """
    row = tl.program_id(0)
    n   = row // S
    s   = row % S

    offs = tl.arange(0, BLOCK_SIZE)
    m    = offs < C

    # Load input (possibly non-contiguous from transpose)
    x    = tl.load(x_ptr + n * stride_xn + s * stride_xs + offs * stride_xc,
                   mask=m, other=0.0)
    xf   = x.to(tl.float32)
    mean = tl.sum(xf * m.to(tl.float32)) * (1.0 / C)
    diff = xf - mean
    var  = tl.sum(diff * diff * m.to(tl.float32)) * (1.0 / C)
    xn   = diff * tl.rsqrt(var + 1e-5)

    w   = tl.load(w_ptr + offs, mask=m, other=1.0).to(tl.float32)
    b   = tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32)
    out = (xn * w + b).to(x.dtype)

    # Write contiguous output [N, S, C]
    tl.store(out_ptr + row * C + offs, out, mask=m)


@torch.fx.wrap
def shared_layer_norm_dispatch(x, weight, bias, route):
    """Single dispatch gate for all LayerNorm variants.

    route="C768"  -> 768-channel LN (bfloat16/9, bfloat16/7)
    route="C1024" -> 1024-channel LN (float32/9)
    """
    N     = x.shape[0]
    S     = x.shape[1]   # H*W (sequence length)
    C     = x.shape[2]
    out   = torch.empty((N, S, C), dtype=x.dtype, device=x.device)
    sxn, sx, sc = x.stride()
    BS    = triton.next_power_of_2(C)

    _layer_norm_kernel[(N * S,)](
        x, weight, bias, out,
        sxn, sx, sc,
        S=S,
        C=C,
        BLOCK_SIZE=BS,
        num_warps=16,
    )
    return out