import triton
import triton.language as tl
import torch


@triton.jit
def _layernorm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    x_s0,
    x_s1,
    x_s2,
    o_s0,
    o_s1,
    o_s2,
    S,
    H,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // S
    seq = row % S
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x_ptrs = x_ptr + batch * x_s0 + seq * x_s1 + offs * x_s2
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / H
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / H
    inv_std = tl.rsqrt(var + eps)

    g = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xc * inv_std * g + b

    out_ptrs = out_ptr + batch * o_s0 + seq * o_s1 + offs * o_s2
    tl.store(out_ptrs, y, mask=mask)


@torch.fx.wrap
def fused_dropout_layernorm(x, gamma, beta, return_x):
    assert x.ndim == 3
    B, S, H = x.shape
    out = torch.empty_like(x)

    if H <= 32:
        block_h = 32
        num_warps = 1
    elif H <= 64:
        block_h = 64
        num_warps = 2
    elif H <= 128:
        block_h = 128
        num_warps = 4
    elif H <= 256:
        block_h = 256
        num_warps = 4
    elif H <= 512:
        block_h = 512
        num_warps = 8
    elif H <= 1024:
        block_h = 1024
        num_warps = 8
    else:
        raise RuntimeError(f"Unsupported hidden size {H}")

    _layernorm_kernel[(B * S,)](
        x,
        gamma,
        beta,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        S,
        H,
        1e-12,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )

    if return_x:
        return x, out
    return out