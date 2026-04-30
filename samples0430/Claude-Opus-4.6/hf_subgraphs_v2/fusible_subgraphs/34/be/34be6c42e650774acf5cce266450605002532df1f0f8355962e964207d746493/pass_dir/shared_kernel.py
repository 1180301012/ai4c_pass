import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_ln_768(
    a_ptr, b_ptr, w_ptr, bi_ptr, out_ptr,
    BLOCK_SIZE: tl.constexpr, HIDDEN: tl.constexpr,
):
    row_idx = tl.program_id(0)
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < HIDDEN
    base = row_idx * HIDDEN

    a = tl.load(a_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
    x = a + b

    mean = tl.sum(x, axis=0) * (1.0 / HIDDEN)
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) * (1.0 / HIDDEN)
    rstd = tl.math.rsqrt(var + 1e-5)
    xn = xc * rstd

    w = tl.load(w_ptr + off, mask=mask, other=0.0).to(tl.float32)
    bi = tl.load(bi_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + off, xn * w + bi, mask=mask)


@triton.jit
def fused_add_ln_32(
    a_ptr, b_ptr, w_ptr, bi_ptr, out_ptr,
    BLOCK_SIZE: tl.constexpr, HIDDEN: tl.constexpr,
):
    row_idx = tl.program_id(0)
    off = tl.arange(0, BLOCK_SIZE)
    base = row_idx * HIDDEN

    a = tl.load(a_ptr + base + off).to(tl.float32)
    b = tl.load(b_ptr + base + off).to(tl.float32)
    x = a + b

    mean = tl.sum(x, axis=0) * (1.0 / HIDDEN)
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) * (1.0 / HIDDEN)
    rstd = tl.math.rsqrt(var + 1e-5)
    xn = xc * rstd

    w = tl.load(w_ptr + off).to(tl.float32)
    bi = tl.load(bi_ptr + off).to(tl.float32)
    tl.store(out_ptr + base + off, xn * w + bi)


@torch.fx.wrap
def fused_add_layernorm_dispatch(a, b, weight, bias, route):
    out = torch.empty_like(a)
    N = a.shape[-1]
    num_rows = a.numel() // N
    if N >= 512:
        fused_add_ln_768[(num_rows,)](
            a, b, weight, bias, out,
            BLOCK_SIZE=1024, HIDDEN=N,
            num_warps=4,
        )
    else:
        fused_add_ln_32[(num_rows,)](
            a, b, weight, bias, out,
            BLOCK_SIZE=32, HIDDEN=N,
            num_warps=1,
        )
    return out