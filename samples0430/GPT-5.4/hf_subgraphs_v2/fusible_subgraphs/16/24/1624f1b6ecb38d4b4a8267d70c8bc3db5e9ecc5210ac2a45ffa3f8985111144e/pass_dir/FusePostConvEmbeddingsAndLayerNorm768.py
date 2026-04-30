import torch
import triton
import triton.language as tl


def pattern(conv_out, cls_token, pos_embed, ln_bias, ln_weight):
    tmp_7 = conv_out.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_9 = cls_token.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + pos_embed
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), ln_weight, ln_bias, 1e-06)
    return tmp_12, tmp_13


def replacement_args(conv_out, cls_token, pos_embed, ln_bias, ln_weight):
    return (conv_out, cls_token, pos_embed, ln_bias, ln_weight)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=4),
    ],
    key=["H"],
)
@triton.jit
def fused_postconv_ln_kernel(
    conv_ptr,
    cls_ptr,
    pos_ptr,
    gamma_ptr,
    beta_ptr,
    out12_ptr,
    out13_ptr,
    conv_s0,
    conv_s1,
    conv_s2,
    conv_s3,
    conv_s4,
    cls_s0,
    cls_s2,
    pos_s1,
    pos_s2,
    out12_s0,
    out12_s1,
    out12_s2,
    out13_s0,
    out13_s1,
    out13_s2,
    OT,
    OH,
    OW,
    H,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    seq = OT * OH * OW + 1
    b = row // seq
    token = row - b * seq

    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    pos_vals = tl.load(pos_ptr + token * pos_s1 + offs * pos_s2, mask=mask, other=0.0)
    cls_vals = tl.load(cls_ptr + b * cls_s0 + offs * cls_s2, mask=mask, other=0.0)

    patch = tl.maximum(token - 1, 0)
    plane = OH * OW
    t = patch // plane
    rem = patch - t * plane
    h = rem // OW
    w = rem - h * OW

    conv_ptrs = conv_ptr + b * conv_s0 + offs * conv_s1 + t * conv_s2 + h * conv_s3 + w * conv_s4
    patch_vals = tl.load(conv_ptrs, mask=mask, other=0.0)
    x = tl.where(token == 0, cls_vals, patch_vals) + pos_vals

    out12_ptrs = out12_ptr + b * out12_s0 + token * out12_s1 + offs * out12_s2
    tl.store(out12_ptrs, x, mask=mask)

    x_fp32 = x.to(tl.float32)
    mean = tl.sum(x_fp32, axis=0) / H
    centered = x_fp32 - mean
    var = tl.sum(centered * centered, axis=0) / H
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = centered * inv_std * gamma + beta

    out13_ptrs = out13_ptr + b * out13_s0 + token * out13_s1 + offs * out13_s2
    tl.store(out13_ptrs, y, mask=mask)


@torch.fx.wrap
def fused_postconv_ln_impl(conv_out, cls_token, pos_embed, ln_bias, ln_weight):
    batch = conv_out.shape[0]
    hidden = conv_out.shape[1]
    ot = conv_out.shape[2]
    oh = conv_out.shape[3]
    ow = conv_out.shape[4]
    seq = ot * oh * ow + 1

    out12 = torch.empty((batch, seq, hidden), device=conv_out.device, dtype=conv_out.dtype)
    out13 = torch.empty((batch, seq, hidden), device=conv_out.device, dtype=conv_out.dtype)

    fused_postconv_ln_kernel[(batch * seq,)](
        conv_out,
        cls_token,
        pos_embed,
        ln_weight,
        ln_bias,
        out12,
        out13,
        conv_out.stride(0),
        conv_out.stride(1),
        conv_out.stride(2),
        conv_out.stride(3),
        conv_out.stride(4),
        cls_token.stride(0),
        cls_token.stride(2),
        pos_embed.stride(1),
        pos_embed.stride(2),
        out12.stride(0),
        out12.stride(1),
        out12.stride(2),
        out13.stride(0),
        out13.stride(1),
        out13.stride(2),
        ot,
        oh,
        ow,
        hidden,
        1e-6,
    )

    return out12, out13


def fused_postconv_ln(conv_out, cls_token, pos_embed, ln_bias, ln_weight):
    outs = fused_postconv_ln_impl(conv_out, cls_token, pos_embed, ln_bias, ln_weight)
    return outs[0], outs[1]


def replacement_func():
    return fused_postconv_ln