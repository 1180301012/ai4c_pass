import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), in_1, in_0, 1e-05)
    return (tmp_8, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_roll_crop_add_ln_kernel(
    bias_ptr,
    weight_ptr,
    in2_ptr,
    in3_ptr,
    out_sum_ptr,
    out_ln_ptr,
    in2_s1,
    in2_s2,
    in3_s1,
    in3_s2,
    in3_s3,
    in3_s4,
    in3_s5,
    out_sum_s1,
    out_sum_s2,
    out_ln_s1,
    out_ln_s2,
    H,
    OUT_H,
    C,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    h = row // OUT_H
    w = row - h * OUT_H
    rh = tl.where(h < 3, h + H - 3, h - 3)
    rw = tl.where(w < 3, w + H - 3, w - 3)

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    src_offs = (
        (rh // 7) * in3_s1
        + (rh % 7) * in3_s2
        + (rw // 7) * in3_s3
        + (rw % 7) * in3_s4
        + offs * in3_s5
    )
    in2_offs = row * in2_s1 + offs * in2_s2
    out_sum_offs = row * out_sum_s1 + offs * out_sum_s2
    out_ln_offs = row * out_ln_s1 + offs * out_ln_s2

    x = tl.load(in2_ptr + in2_offs, mask=mask, other=0.0)
    y = tl.load(in3_ptr + src_offs, mask=mask, other=0.0)
    s_raw = x + y
    tl.store(out_sum_ptr + out_sum_offs, s_raw, mask=mask)

    s = s_raw.to(tl.float32)
    mean = tl.sum(s, axis=0) / C
    diff = tl.where(mask, s - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C
    rstd = tl.rsqrt(var + 1e-5)

    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = diff * rstd * weight + bias
    tl.store(out_ln_ptr + out_ln_offs, out, mask=mask)


@torch.fx.wrap
def fused_roll_crop_add_layernorm_dispatch(bias, weight, in_2, in_3):
    out_sum = torch.empty_like(in_2)
    out_ln = torch.empty_like(in_2)

    H = in_3.shape[1] * in_3.shape[2]
    OUT_H = H - 5
    C = in_3.shape[5]
    M = in_2.shape[1]

    if C <= 128:
        block_c = 128
        num_warps = 4
    elif C <= 256:
        block_c = 256
        num_warps = 4
    else:
        block_c = 512
        num_warps = 8

    fused_roll_crop_add_ln_kernel[(M,)](
        bias,
        weight,
        in_2,
        in_3,
        out_sum,
        out_ln,
        in_2.stride(1),
        in_2.stride(2),
        in_3.stride(1),
        in_3.stride(2),
        in_3.stride(3),
        in_3.stride(4),
        in_3.stride(5),
        out_sum.stride(1),
        out_sum.stride(2),
        out_ln.stride(1),
        out_ln.stride(2),
        H,
        OUT_H,
        C,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
    return out_sum, out_ln


def replacement_func():
    return fused_roll_crop_add_layernorm_dispatch