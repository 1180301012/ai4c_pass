import torch
import triton
import triton.language as tl


def pattern(x, ln_bias, ln_weight):
    y = torch.nn.functional.layer_norm(x, (768,), ln_weight, ln_bias, 1e-06)
    return y


def replacement_args(x, ln_bias, ln_weight):
    return (x, ln_bias, ln_weight)


@triton.jit
def layer_norm_768_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    x_s0,
    x_s1,
    x_s2,
    out_s0,
    out_s1,
    out_s2,
    S,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // S
    seq = row - batch * S

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    x = tl.load(x_ptr + batch * x_s0 + seq * x_s1 + offs * x_s2, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)

    mean = tl.sum(x_fp32, axis=0) / H
    centered = x_fp32 - mean
    var = tl.sum(centered * centered, axis=0) / H
    inv_std = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = centered * inv_std * w + b

    tl.store(out_ptr + batch * out_s0 + seq * out_s1 + offs * out_s2, y, mask=mask)


@torch.fx.wrap
def layer_norm_768(x, ln_bias, ln_weight):
    batch = x.shape[0]
    seq = x.shape[1]
    hidden = x.shape[2]
    out = torch.empty_like(x)

    layer_norm_768_kernel[(batch * seq,)](
        x,
        ln_weight,
        ln_bias,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        seq,
        hidden,
        1e-6,
        BLOCK_SIZE=1024,
    )
    return out


def replacement_func():
    return layer_norm_768