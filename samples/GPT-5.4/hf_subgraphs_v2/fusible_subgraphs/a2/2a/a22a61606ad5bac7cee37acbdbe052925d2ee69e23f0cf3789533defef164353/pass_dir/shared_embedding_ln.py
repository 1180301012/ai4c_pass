import triton
import triton.language as tl
import torch


@triton.jit
def _fused_add_ln_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    gamma_ptr,
    beta_ptr,
    out_sum_ptr,
    out_ln_ptr,
    a_s0,
    a_s1,
    a_s2,
    b_s0,
    b_s1,
    b_s2,
    c_s0,
    c_s1,
    c_s2,
    o_s0,
    o_s1,
    o_s2,
    S,
    H,
    eps,
    STORE_SUM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // S
    seq = row % S
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    a_ptrs = a_ptr + batch * a_s0 + seq * a_s1 + offs * a_s2
    b_ptrs = b_ptr + batch * b_s0 + seq * b_s1 + offs * b_s2
    c_ptrs = c_ptr + batch * c_s0 + seq * c_s1 + offs * c_s2

    av = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)
    cv = tl.load(c_ptrs, mask=mask, other=0.0).to(tl.float32)
    x = av + bv + cv

    mean = tl.sum(x, axis=0) / H
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / H
    inv_std = tl.rsqrt(var + eps)

    g = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    bt = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xc * inv_std * g + bt

    out_ln_ptrs = out_ln_ptr + batch * o_s0 + seq * o_s1 + offs * o_s2
    tl.store(out_ln_ptrs, y, mask=mask)

    if STORE_SUM:
        out_sum_ptrs = out_sum_ptr + batch * o_s0 + seq * o_s1 + offs * o_s2
        tl.store(out_sum_ptrs, x, mask=mask)


@torch.fx.wrap
def fused_add_dropout_layernorm(
    a,
    b,
    c,
    gamma,
    beta,
    return_sum,
):
    assert a.ndim == 3
    assert b.ndim == 3
    assert c.ndim == 3
    B, S, H = a.shape

    if b.shape[0] == 1 and B != 1:
        b = b.expand(B, b.shape[1], b.shape[2])
    if c.shape[0] == 1 and B != 1:
        c = c.expand(B, c.shape[1], c.shape[2])

    out_ln = torch.empty_like(a)
    out_sum = torch.empty_like(a) if return_sum else torch.empty((1,), device=a.device, dtype=a.dtype)

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

    _fused_add_ln_kernel[(B * S,)](
        a,
        b,
        c,
        gamma,
        beta,
        out_sum,
        out_ln,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        out_ln.stride(0),
        out_ln.stride(1),
        out_ln.stride(2),
        S,
        H,
        1e-12,
        STORE_SUM=return_sum,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )

    if return_sum:
        return out_sum, out_ln
    return out_ln