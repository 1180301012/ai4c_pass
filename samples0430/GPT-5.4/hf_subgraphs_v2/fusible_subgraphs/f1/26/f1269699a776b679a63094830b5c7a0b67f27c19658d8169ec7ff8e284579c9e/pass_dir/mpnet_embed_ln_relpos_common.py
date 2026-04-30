import torch
import triton
import triton.language as tl


@triton.jit
def _mpnet_embed_ln_kernel(
    input_ids_ptr,
    position_ids_ptr,
    word_weight_ptr,
    position_weight_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    B,
    S,
    D,
    input_ids_s0,
    input_ids_s1,
    position_ids_s0,
    position_ids_s1,
    word_weight_s0,
    word_weight_s1,
    position_weight_s0,
    position_weight_s1,
    out_s0,
    out_s1,
    out_s2,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // S
    s = row - b * S

    token_id = tl.load(input_ids_ptr + b * input_ids_s0 + s * input_ids_s1).to(tl.int64)
    position_id = tl.load(position_ids_ptr + b * position_ids_s0 + s * position_ids_s1).to(tl.int64)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    word_vals = tl.load(
        word_weight_ptr + token_id * word_weight_s0 + offs * word_weight_s1,
        mask=mask,
        other=0,
    ).to(tl.float32)
    pos_vals = tl.load(
        position_weight_ptr + position_id * position_weight_s0 + offs * position_weight_s1,
        mask=mask,
        other=0,
    ).to(tl.float32)
    x = word_vals + pos_vals

    mean = tl.sum(x, axis=0) / D
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / D
    rstd = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1).to(tl.float32)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = centered * rstd
    y = y * gamma + beta

    tl.store(out_ptr + b * out_s0 + s * out_s1 + offs * out_s2, y, mask=mask)


@triton.jit
def _mpnet_relpos_bucket_kernel(
    out_ptr,
    S,
    out_s0,
    out_s1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < S) & (offs_n < S)

    rel = offs_m - offs_n
    neg = rel < 0
    abs_rel = tl.abs(rel)
    small = abs_rel < 8

    abs_rel_clamped = tl.maximum(abs_rel, 8)
    big = abs_rel_clamped.to(tl.float32) / 8.0
    big = tl.log(big)
    big = (big / 2.772588722239781) * 8.0
    big = 8 + big.to(tl.int32)
    big = tl.minimum(big, 15)

    bucket = tl.where(small, abs_rel, big)
    bucket = bucket + neg.to(tl.int32) * 16

    tl.store(out_ptr + offs_m * out_s0 + offs_n * out_s1, bucket.to(tl.int64), mask=mask)


def _mpnet_fused_impl(in_0, in_1, in_2, in_3, in_4, in_5, eps):
    batch = in_0.shape[0]
    seq = in_0.shape[1]
    hidden = in_2.shape[0]

    out = torch.empty((batch, seq, hidden), device=in_4.device, dtype=in_4.dtype)
    rel = torch.empty((seq, seq), device=in_4.device, dtype=torch.int64)

    if hidden <= 64:
        block_d = 64
        num_warps = 2
    else:
        block_d = 1024
        num_warps = 8

    _mpnet_embed_ln_kernel[(batch * seq,)](
        in_0,
        in_5,
        in_4,
        in_3,
        in_2,
        in_1,
        out,
        batch,
        seq,
        hidden,
        in_0.stride(0),
        in_0.stride(1),
        in_5.stride(0),
        in_5.stride(1),
        in_4.stride(0),
        in_4.stride(1),
        in_3.stride(0),
        in_3.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        eps,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )

    if seq <= 16:
        block_m = 16
        block_n = 16
    else:
        block_m = 32
        block_n = 32

    _mpnet_relpos_bucket_kernel[(triton.cdiv(seq, block_m), triton.cdiv(seq, block_n))](
        rel,
        seq,
        rel.stride(0),
        rel.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=1,
    )

    return out, rel


@torch.fx.wrap
def mpnet_embed_ln_relpos_dispatch(in_0, in_1, in_2, in_3, in_4, in_5, route):
    if route == "seq7_h768_eps1e5":
        return _mpnet_fused_impl(in_0, in_1, in_2, in_3, in_4, in_5, 1e-5)
    if route == "seq11_h768_eps1e5":
        return _mpnet_fused_impl(in_0, in_1, in_2, in_3, in_4, in_5, 1e-5)
    if route == "seq45_h64_eps1e12":
        return _mpnet_fused_impl(in_0, in_1, in_2, in_3, in_4, in_5, 1e-12)
    return _mpnet_fused_impl(in_0, in_1, in_2, in_3, in_4, in_5, 1e-5)


def replacement_func():
    return mpnet_embed_ln_relpos_dispatch