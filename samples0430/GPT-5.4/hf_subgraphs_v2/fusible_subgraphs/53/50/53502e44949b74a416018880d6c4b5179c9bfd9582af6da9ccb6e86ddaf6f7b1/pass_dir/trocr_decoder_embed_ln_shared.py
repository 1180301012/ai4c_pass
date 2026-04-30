import torch
import triton
import triton.language as tl


HIDDEN_SIZE = 256
TOKEN_SCALE = 16.0
LN_EPS = 1e-5


@triton.jit
def two_embed_mul_add_ln_kernel(
    tok_weight_ptr,
    pos_weight_ptr,
    gamma_ptr,
    beta_ptr,
    tok_ids_ptr,
    pos_ids_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    TOKEN_SCALE_CONST: tl.constexpr,
    EPS: tl.constexpr,
):
    offs = tl.arange(0, HIDDEN)
    tok_idx = tl.load(tok_ids_ptr)
    pos_idx = tl.load(pos_ids_ptr)

    tok = tl.load(tok_weight_ptr + tok_idx * HIDDEN + offs).to(tl.float32)
    pos = tl.load(pos_weight_ptr + pos_idx * HIDDEN + offs).to(tl.float32)
    v = tok * TOKEN_SCALE_CONST + pos

    mean = tl.sum(v, axis=0) / HIDDEN
    centered = v - mean
    var = tl.sum(centered * centered, axis=0) / HIDDEN
    inv_std = tl.rsqrt(var + EPS)

    gamma = tl.load(gamma_ptr + offs).to(tl.float32)
    beta = tl.load(beta_ptr + offs).to(tl.float32)
    y = centered * inv_std
    y = y * gamma + beta

    tl.store(out_ptr + offs, y)


@triton.jit
def embed_mul_add_ln_kernel(
    addend_ptr,
    tok_weight_ptr,
    gamma_ptr,
    beta_ptr,
    ids_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    TOKEN_SCALE_CONST: tl.constexpr,
    EPS: tl.constexpr,
):
    offs = tl.arange(0, HIDDEN)
    tok_idx = tl.load(ids_ptr)

    tok = tl.load(tok_weight_ptr + tok_idx * HIDDEN + offs).to(tl.float32)
    addend = tl.load(addend_ptr + offs).to(tl.float32)
    v = tok * TOKEN_SCALE_CONST + addend

    mean = tl.sum(v, axis=0) / HIDDEN
    centered = v - mean
    var = tl.sum(centered * centered, axis=0) / HIDDEN
    inv_std = tl.rsqrt(var + EPS)

    gamma = tl.load(gamma_ptr + offs).to(tl.float32)
    beta = tl.load(beta_ptr + offs).to(tl.float32)
    y = centered * inv_std
    y = y * gamma + beta

    tl.store(out_ptr + offs, y)


@triton.jit
def add_ln_kernel(
    x_ptr,
    addend_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
):
    offs = tl.arange(0, HIDDEN)

    x = tl.load(x_ptr + offs).to(tl.float32)
    addend = tl.load(addend_ptr + offs).to(tl.float32)
    v = x + addend

    mean = tl.sum(v, axis=0) / HIDDEN
    centered = v - mean
    var = tl.sum(centered * centered, axis=0) / HIDDEN
    inv_std = tl.rsqrt(var + EPS)

    gamma = tl.load(gamma_ptr + offs).to(tl.float32)
    beta = tl.load(beta_ptr + offs).to(tl.float32)
    y = centered * inv_std
    y = y * gamma + beta

    tl.store(out_ptr + offs, y)


@torch.fx.wrap
def trocr_decoder_dispatch(*args):
    route = args[-1]

    if route == "two_embed_mul_add_ln":
        tok_weight, pos_weight, gamma, beta, tok_ids, pos_ids = args[:-1]
        out = torch.empty((1, 1, HIDDEN_SIZE), device=tok_weight.device, dtype=tok_weight.dtype)
        two_embed_mul_add_ln_kernel[(1,)](
            tok_weight,
            pos_weight,
            gamma,
            beta,
            tok_ids,
            pos_ids,
            out,
            HIDDEN=HIDDEN_SIZE,
            TOKEN_SCALE_CONST=TOKEN_SCALE,
            EPS=LN_EPS,
            num_warps=1,
            num_stages=1,
        )
        return out

    if route == "embed_mul_add_ln":
        addend, tok_weight, gamma, beta, ids = args[:-1]
        out = torch.empty_like(addend)
        embed_mul_add_ln_kernel[(1,)](
            addend,
            tok_weight,
            gamma,
            beta,
            ids,
            out,
            HIDDEN=HIDDEN_SIZE,
            TOKEN_SCALE_CONST=TOKEN_SCALE,
            EPS=LN_EPS,
            num_warps=1,
            num_stages=1,
        )
        return out

    if route == "add_ln":
        x, addend, gamma, beta = args[:-1]
        out = torch.empty_like(x)
        add_ln_kernel[(1,)](
            x,
            addend,
            gamma,
            beta,
            out,
            HIDDEN=HIDDEN_SIZE,
            EPS=LN_EPS,
            num_warps=1,
            num_stages=1,
        )
        return out

    raise RuntimeError(f"unknown route: {route}")