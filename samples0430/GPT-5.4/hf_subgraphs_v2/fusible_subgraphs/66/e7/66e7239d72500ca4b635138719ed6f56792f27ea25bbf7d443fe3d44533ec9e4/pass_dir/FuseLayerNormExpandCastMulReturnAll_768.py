import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(tmp_4)
    tmp_7 = tmp_6.float()
    tmp_8 = tmp_4 * tmp_7
    return (tmp_7, tmp_8, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_layernorm_mask_kernel(
    x_ptr,
    beta_ptr,
    gamma_ptr,
    mask_ptr,
    out_mask_ptr,
    out_mul_ptr,
    out_ln_ptr,
    B,
    S,
    H,
    x_s0,
    x_s1,
    x_s2,
    m_s0,
    m_s1,
    om_s0,
    om_s1,
    om_s2,
    ou_s0,
    ou_s1,
    ou_s2,
    ol_s0,
    ol_s1,
    ol_s2,
    EPS,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    rows = B * S
    if row >= rows:
        return

    b = row // S
    s = row % S

    offs = tl.arange(0, BLOCK_SIZE)
    col_mask = offs < H

    x_base = b * x_s0 + s * x_s1
    x = tl.load(x_ptr + x_base + offs * x_s2, mask=col_mask, other=0.0)
    x_f32 = x.to(tl.float32)

    mean = tl.sum(x_f32, axis=0) / H
    centered = tl.where(col_mask, x_f32 - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / H
    rstd = tl.rsqrt(var + EPS)

    gamma = tl.load(gamma_ptr + offs, mask=col_mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs, mask=col_mask, other=0.0).to(tl.float32)
    y_f32 = centered * rstd * gamma + beta

    if IS_BF16:
        y_lp = y_f32.to(tl.bfloat16)
    else:
        y_lp = y_f32.to(tl.float16)

    mask_scalar = tl.load(mask_ptr + b * m_s0 + s * m_s1).to(tl.float32)

    out_ln_base = b * ol_s0 + s * ol_s1
    out_mask_base = b * om_s0 + s * om_s1
    out_mul_base = b * ou_s0 + s * ou_s1

    tl.store(out_ln_ptr + out_ln_base + offs * ol_s2, y_lp, mask=col_mask)
    tl.store(out_mask_ptr + out_mask_base + offs * om_s2, mask_scalar, mask=col_mask)
    tl.store(out_mul_ptr + out_mul_base + offs * ou_s2, y_lp.to(tl.float32) * mask_scalar, mask=col_mask)


@torch.fx.wrap
def fused_layernorm_expand_cast_mul_runtime(in_0, in_1, in_2, in_3):
    out_mask = torch.empty_like(in_3, dtype=torch.float32)
    out_mul = torch.empty_like(in_3, dtype=torch.float32)
    out_ln = torch.empty_like(in_3)

    B = in_3.shape[0]
    S = in_3.shape[1]
    H = in_3.shape[2]

    BLOCK_SIZE = 1024
    grid = (B * S,)

    fused_layernorm_mask_kernel[grid](
        in_3,
        in_1,
        in_2,
        in_0,
        out_mask,
        out_mul,
        out_ln,
        B,
        S,
        H,
        in_3.stride(0),
        in_3.stride(1),
        in_3.stride(2),
        in_0.stride(0),
        in_0.stride(1),
        out_mask.stride(0),
        out_mask.stride(1),
        out_mask.stride(2),
        out_mul.stride(0),
        out_mul.stride(1),
        out_mul.stride(2),
        out_ln.stride(0),
        out_ln.stride(1),
        out_ln.stride(2),
        1e-12,
        IS_BF16=(in_3.dtype == torch.bfloat16),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )

    return (out_mask, out_mul, out_ln)


def replacement_impl(in_0, in_1, in_2, in_3):
    outs = fused_layernorm_expand_cast_mul_runtime(in_0, in_1, in_2, in_3)
    return (outs[0], outs[1], outs[2])


def replacement_func():
    return replacement_impl