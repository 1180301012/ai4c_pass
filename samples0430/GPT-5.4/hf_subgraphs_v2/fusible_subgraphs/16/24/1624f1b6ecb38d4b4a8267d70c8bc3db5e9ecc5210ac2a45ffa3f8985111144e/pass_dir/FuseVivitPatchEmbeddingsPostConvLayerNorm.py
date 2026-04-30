import torch
import triton
import triton.language as tl


def pattern(conv_out, cls_token, pos_embed):
    tmp_7 = conv_out.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_9 = cls_token.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + pos_embed
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    return tmp_12


def replacement_args(conv_out, cls_token, pos_embed):
    return (conv_out, cls_token, pos_embed)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=4),
    ],
    key=["H"],
)
@triton.jit
def fused_vivit_patch_post_kernel(
    conv_ptr,
    cls_ptr,
    pos_ptr,
    out_ptr,
    conv_s0,
    conv_s1,
    conv_s2,
    conv_s3,
    conv_s4,
    cls_s0,
    cls_s1,
    cls_s2,
    pos_s1,
    pos_s2,
    out_s0,
    out_s1,
    out_s2,
    OT,
    OH,
    OW,
    H,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    seq = OT * OH * OW + 1
    b = row // seq
    token = row - b * seq

    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    cls_vals = tl.load(cls_ptr + b * cls_s0 + offs * cls_s2, mask=mask, other=0.0)
    pos_vals = tl.load(pos_ptr + token * pos_s1 + offs * pos_s2, mask=mask, other=0.0)

    patch = tl.maximum(token - 1, 0)
    plane = OH * OW
    t = patch // plane
    rem = patch - t * plane
    h = rem // OW
    w = rem - h * OW

    conv_ptrs = conv_ptr + b * conv_s0 + offs * conv_s1 + t * conv_s2 + h * conv_s3 + w * conv_s4
    conv_vals = tl.load(conv_ptrs, mask=mask, other=0.0)

    x = tl.where(token == 0, cls_vals, conv_vals) + pos_vals
    out_ptrs = out_ptr + b * out_s0 + token * out_s1 + offs * out_s2
    tl.store(out_ptrs, x, mask=mask)


@torch.fx.wrap
def fused_vivit_patch_post(conv_out, cls_token, pos_embed):
    batch = conv_out.shape[0]
    hidden = conv_out.shape[1]
    ot = conv_out.shape[2]
    oh = conv_out.shape[3]
    ow = conv_out.shape[4]
    seq = ot * oh * ow + 1

    out = torch.empty((batch, seq, hidden), device=conv_out.device, dtype=conv_out.dtype)

    fused_vivit_patch_post_kernel[(batch * seq,)](
        conv_out,
        cls_token,
        pos_embed,
        out,
        conv_out.stride(0),
        conv_out.stride(1),
        conv_out.stride(2),
        conv_out.stride(3),
        conv_out.stride(4),
        cls_token.stride(0),
        cls_token.stride(1),
        cls_token.stride(2),
        pos_embed.stride(1),
        pos_embed.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        ot,
        oh,
        ow,
        hidden,
    )

    return out


def replacement_func():
    return fused_vivit_patch_post